from dataclasses import dataclass, field, replace
from enum import Enum
import time
import math
from typing import Tuple

import os
import arx5_interface as arx5
import numpy as np
import torch

from lerobot.cameras import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.utils.robot_utils import precise_sleep

from lerobot_arx5 import (
    ARXControlModel,
    ARX5ArmConfig,
    ARX5RobotConfig,
)

DOF = 6
MOTOR_NAMES = []

class ARX5Arm:
    """
    Class for controlling a single ARX arm.
    """

    def __init__(
        self,
        control_mode: ARXControlModel,
        config: ARX5ArmConfig,
        is_master: bool,
    ):
        self.control_mode = control_mode
        self.config = config
        self.is_connected = False
        self.is_master = is_master
        self.robot_controller = None

    def connect(self):
        if self.is_connected:
            raise DeviceAlreadyConnectedError(
                "ARX5Arm is already connected. Do not run `robot.connect()` twice."
            )
        self.is_connected = True

        robot_config = arx5.RobotConfigFactory.get_instance().get_config(self.config.model)
        robot_config.gripper_torque_max *= 2
        controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
            self.control_mode.value, robot_config.joint_dof
        )
        controller_config.controller_dt = 0.01  # Sets the internal communication frequency (in seconds).
                                                # Slower CPU + USB I/O processing requires lower frequency comms.
        controller_config.gravity_compensation = True   # TODO: may be default true
        controller_config.background_send_recv = True

        if self.control_mode == ARXControlModel.JOINT_CONTROLLER:
            self.robot_controller = arx5.Arx5JointController(robot_config, controller_config, self.config.interface_name)
            print("ARX5 Joint Controller created")
        elif self.control_mode == ARXControlModel.CARTESIAN_CONTROLLER:
            self.robot_controller = arx5.Arx5CartesianController(robot_config, controller_config, self.config.interface_name)
            print("ARX5 Cartesian Controller created")
        else:
            raise ValueError(f"Invalid arm control mode provided: expected {ARXControlModel.JOINT_CONTROLLER} or {ARXControlModel.CARTESIAN_CONTROLLER}, got {self.control_mode}")
        
        self.robot_controller.reset_to_home()
        # self.robot_controller.enable_gravity_compensation(self.config.urdf_path)
        # self.robot_controller.set_log_level(arx5.LogLevel.DEBUG)

        self.robot_config = robot_config
        print(f"Gripper max width: {self.robot_config.gripper_width}")

    def disconnect(self):
        if not self.is_connected:
            raise DeviceAlreadyConnectedError(
                "ARX5Arm is not connected. Do not run `robot.disconnect()` twice."
            )
        # notify the arm process of imminent shutdown
        self.is_connected = False
        # safely shut down the follower arm
        if not self.is_master:
            self.robot_controller.reset_to_home()
            self.robot_controller.set_to_damping()
        self.robot_controller = None

    def reset(self):
        if not self.is_connected:
            raise DeviceAlreadyConnectedError(
                "ARX5Arm is not connected. Do not run `robot.disconnect()` twice."
            )
        self.robot_controller.reset_to_home()

    def configure(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "ARX5Arm is not connected. You need to run `robot.connect()`."
            )
        if self.is_master:
            self.robot_controller.set_to_damping()
            gain = self.robot_controller.get_gain()
            gain.kd()[:3] *= 0.05
            gain.kd()[3:] *= 0.1
            self.robot_controller.set_gain(gain)
        else:
            # gripper - make it less aggressive
            gain = self.robot_controller.get_gain()
            gain.kd()[:3] /= 1.2
            gain.kd()[3:] *= 1.2
            gain.gripper_kp /= 1.8
            gain.gripper_kd *= 1.8
            self.robot_controller.set_gain(gain)

    def get_state(self) -> np.ndarray:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "ARX5Arm is not connected. You need to run `robot.connect()`."
            )
        if self.control_mode == ARXControlModel.JOINT_CONTROLLER:
            joint_state = self.robot_controller.get_joint_state()
            state = np.concatenate([joint_state.pos().copy(), np.array([joint_state.gripper_pos])])
        else:
            eef_state = self.robot_controller.get_eef_state()
            state = np.concatenate([eef_state.pose_6d().copy(), np.array([eef_state.gripper_pos])])
        if self.is_master:
            state[-1] *= 3.85

        # name = "master" if self.is_master else "puppet"
        # print(f"arm ({name}): {state[-3:-1]}")
        return state
    
    def get_obs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "ARX5Arm is not connected. You need to run `robot.connect()`."
            )
        joint_state = self.robot_controller.get_joint_state()
        pos = np.concatenate([joint_state.pos().copy(), np.array([joint_state.gripper_pos])])
        vel = np.concatenate([joint_state.vel().copy(), np.array([joint_state.gripper_vel])])
        effort = np.concatenate([joint_state.torque().copy(), np.array([joint_state.gripper_torque])])
        if self.is_master:
            pos[-1] *= 3.85
            vel[-1] *= 3.85
            effort[-1] *= 3.85

        eef_cartesians = self.robot_controller.get_eef_state()
        eef_state = eef_cartesians.pose_6d().copy()
        # eef_state = np.concatenate([eef_cartesians.pose_6d().copy(), np.array([eef_cartesians.gripper_pos])])
        return (pos, vel, effort, eef_state)
    
    def send_command(self, action: np.ndarray):
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "ARX5Arm is not connected. You need to run `robot.connect()`."
            )
        # Rescale gripper width
        if self.is_master:
            action[DOF] /= 3.85
        if action[DOF] > self.robot_config.gripper_width:
            action[DOF] = self.robot_config.gripper_width
        gripper_pos = action[DOF]
        
        if self.control_mode == ARXControlModel.JOINT_CONTROLLER:
            cmd = arx5.JointState(DOF)
            cmd.pos()[0:DOF] = action[0:DOF]
            cmd.gripper_pos = gripper_pos
            self.robot_controller.set_joint_cmd(cmd)
        else:
            cartesian_pos = action[0:DOF]
            eef_cmd = arx5.EEFState(cartesian_pos, gripper_pos)
            self.robot_controller.set_eef_cmd(eef_cmd)

    def interpolate_arm_position(self, action: np.ndarray):
        seconds = 3.5
        fps = 30
        num_steps = math.ceil(seconds * fps)
        current_pos = self.get_state()

        for i in range(num_steps + 1):  # +1 to include the target
            start_loop_t = time.perf_counter()

            t = i / num_steps
            interp_pos = current_pos * (1 - t) + action * t
            self.send_command(interp_pos)

            dt_s = time.perf_counter() - start_loop_t
            precise_sleep(1 / fps - dt_s)

class ARX5Robot:
    """
    A class for controlling a robot consisting of one or more ARX arms.
    """
    robot_type: str | None = "arx5"

    def __init__(
        self,
        config: ARX5RobotConfig | None = None,
    ):
        if config is None:
            raise ValueError("No robot config provided.")

        if config.control_mode == ARXControlModel.JOINT_CONTROLLER:
            self.joint_thresholds = np.array([0.25, 0.25, 0.25, 0.3, 0.3, 0.3, 0.10])
        else:
            self.joint_thresholds = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.10])

        self.leader_arms = {}
        self.follower_arms = {}
        for key, arm_config in self.config.leader_arms.items():
            self.leader_arms[key] = ARX5Arm(arm_config, True)
        for key, arm_config in self.config.follower_arms.items():
            self.follower_arms[key] = ARX5Arm(arm_config, False)

        self.cameras = make_cameras_from_configs(self.config.cameras)
        self.is_connected = False
        self.logs = {}

        # set teleop offset to 0
        self.teleop_offset = {}
        for key in self.config.follower_arms.keys():
            self.teleop_offset[key] = torch.zeros(DOF + 1)

    @property
    def has_camera(self):
        return len(self.cameras) > 0
    
    @property
    def num_cameras(self):
        return len(self.cameras)
    
    @property
    def _cameras_ft(self) -> dict:
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            key = f"observation.images.{cam_key}"
            cam_ft[key] = {
                "shape": (cam.height, cam.width, cam.channels),
                "names": ["height", "width", "channels"],
                "info": None,
            }
        return cam_ft
    
    @property
    def _motors_ft(self) -> dict:
        action_space = len(self.follower_arms) * (DOF + 1)
        return {
            "action": {
                "dtype": "float32",
                "shape": (action_space,),
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (action_space,),
            },
            "observation.velocity": {
                "dtype": "float32",
                "shape": (action_space,),
            },
            "observation.effort": {
                "dtype": "float32",
                "shape": (action_space,),
            },
            "observation.eef_6d_pose": {
                "dtype": "float32",
                "shape": (len(self.follower_arms)*6,),
            },
        }
    
    @property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft
    
    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self):
        if self.is_connected:
            raise DeviceAlreadyConnectedError(
                "ARX5Robot is already connected. Do not run `robot.connect()` twice."
            )

        if not self.leader_arms and not self.follower_arms and not self.cameras:
            raise ValueError(
                "ARX5Robot doesn't have any device to connect. See example of usage in docstring of the class."
            )

        # Connect the arms
        for name in self.follower_arms:
            print(f"Connecting {name} follower arm.")
            self.follower_arms[name].connect()
            time.sleep(0.2)
        for name in self.leader_arms:
            print(f"Connecting {name} leader arm.")
            self.leader_arms[name].connect()
            time.sleep(0.2)

        # Run calibration process which begins by resetting all arms
        self.configure()

        # Connect the cameras
        for name in self.cameras:
            self.cameras[name].connect()

        self.is_connected = True

    def configure(self):
        for name in self.follower_arms:
            print(f"Configuring {name} follower arm: {self.follower_arms[name]}")
            self.follower_arms[name].configure()
        for name in self.leader_arms:
            print(f"Configuring {name} leader arm: {self.leader_arms[name]}")
            self.leader_arms[name].configure()

    def reset(self):
        # reset leader arms
        for name in self.follower_arms:
            self.follower_arms[name].reset()

        # reset teleop offsets
        for name in self.teleop_offset:
            self.teleop_offset[name] = torch.zeros(DOF + 1)

    def teleop_step(
        self, record_data=False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "ARX5Robot is not connected. You need to run `robot.connect()`."
            )
        
        # Prepare to assign the position of the leader to the follower
        leader_pos = {}
        for name in self.leader_arms:
            before_lread_t = time.perf_counter()
            leader_pos[name] = self.leader_arms[name].get_state()
            self.logs[f"read_leader_{name}_pos_dt_s"] = time.perf_counter() - before_lread_t
        
        follower_goal_pos = {}
        for name in self.leader_arms:
            follower_current_pos = self.follower_arms[name].get_state()
            follower_goal_pos[name] = leader_pos[name]
            # Check if goal position is safe - if the difference in individual joints is greater than the threshold
            # We need to raise an error
            if np.any(np.abs(follower_goal_pos[name] - follower_current_pos) > self.joint_thresholds):
                raise ValueError(f"Goal position for {name} is not safe. The difference in individual joints is greater than the threshold.")

            # Send action
            if name in self.follower_arms:
                before_fwrite_t = time.perf_counter()

                action = np.add(leader_pos[name], self.teleop_offset[name]) # TODO: confirm they are the same shape and this is how to add tensors
                self.follower_arms[name].send_command(action[0:DOF + 1])

                self.logs[f"write_follower_{name}_goal_pos_dt_s"] = time.perf_counter() - before_fwrite_t

        # Early exit when recording data is not requested
        if not record_data:
            return

        # Read follower position
        follower_obs = {}
        for name in self.follower_arms:
            before_fread_t = time.perf_counter()
            follower_obs[name] = self.follower_arms[name].get_obs()
            self.logs[f"read_follower_{name}_pos_dt_s"] = time.perf_counter() - before_fread_t

        # Create state by concatenating follower current position
        state = []
        velocity = []
        effort = []
        eef_pose = []
        for name in self.follower_arms:
            if name in follower_obs:
                state.append(follower_obs[name][0])
                velocity.append(follower_obs[name][1])
                effort.append(follower_obs[name][2])
                eef_pose.append(follower_obs[name][3])
        state = np.concatenate(state)
        velocity = np.concatenate(velocity)
        effort = np.concatenate(effort)
        eef_pose = np.concatenate(eef_pose)

        # Create action by concatenating follower goal position
        action = []
        for name in self.leader_arms:
            if name in follower_goal_pos:
                action.append(follower_goal_pos[name])
        action = np.concatenate(action)

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # Populate output dictionnaries and format to pytorch
        obs_dict, action_dict = {}, {}
        obs_dict["observation.state"] = torch.from_numpy(state.astype(np.float32))
        obs_dict["observation.velocity"] = torch.from_numpy(velocity.astype(np.float32))
        obs_dict["observation.effort"] = torch.from_numpy(effort.astype(np.float32))
        obs_dict["observation.eef_6d_pose"] = torch.from_numpy(eef_pose.astype(np.float32))
        action_dict["action"] = torch.from_numpy(action.astype(np.float32))
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = torch.from_numpy(images[name])

        return obs_dict, action_dict

    def capture_observation(self):
        """The returned observations do not have a batch dimension."""
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "ARX5Robot is not connected. You need to run `robot.connect()`."
            )

        # Read follower observations
        follower_obs = {}
        for name in self.follower_arms:
            before_fread_t = time.perf_counter()
            follower_obs[name] = self.follower_arms[name].get_obs()
            self.logs[f"read_follower_{name}_pos_dt_s"] = time.perf_counter() - before_fread_t

        # Create state by concatenating follower current position
        state = []
        velocity = []
        effort = []
        eef_pose = []
        for name in self.follower_arms:
            if name in follower_obs:
                state.append(follower_obs[name][0])
                velocity.append(follower_obs[name][1])
                effort.append(follower_obs[name][2])
                eef_pose.append(follower_obs[name][3])
        state = np.concatenate(state)
        velocity = np.concatenate(velocity)
        effort = np.concatenate(effort)
        eef_pose = np.concatenate(eef_pose)

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # Populate output dictionaries
        obs_dict = {}
        obs_dict["observation.state"] = state.astype(np.float32)
        obs_dict["observation.velocity"] = velocity.astype(np.float32)
        obs_dict["observation.effort"] = effort.astype(np.float32)
        obs_dict["observation.eef_6d_pose"] = eef_pose.astype(np.float32)
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = torch.from_numpy(images[name])
        return obs_dict

    def send_action(self, action: torch.Tensor):
        """The provided action is expected to be a vector."""
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "ARX5Robot is not connected. You need to run `robot.connect()`."
            )
        action = action.numpy()

        from_idx = 0
        follower_goal_pos = {}
        for name in self.follower_arms:
            if name in self.follower_arms:
                to_idx = DOF
                follower_goal_pos[name] = action[from_idx:(from_idx + to_idx + 1)]
                from_idx = to_idx + 1

        for name in self.follower_arms:
            self.follower_arms[name].send_command(follower_goal_pos[name])

    def set_leader_arms_to_follower_positions(self):
        """
        Used during dAgger. Safely sets follower positions to match the current master position.
        """
        for name, leader_arm in self.leader_arms.items():
            leader_arm.reset()
            
            # capture follower arm's position
            follower_pos = self.follower_arms[name].get_state()
            
            print(f"*** Setting leader arm '{name}' to position {follower_pos} ***")
            leader_arm.interpolate_arm_position(follower_pos)
            print(f"*** Leader arm '{name}' set to position {follower_pos} ***")
            
            # unlock the leader arm
            leader_arm.configure()

    def lock_teleop_offset(self):
        """
        Used to lock the offset between leader and follower arms to their current positions
        """

        # first get current position of leader
        leader_pos = {}
        for name in self.leader_arms:
            leader_pos[name] = self.leader_arms[name].get_state()

        # then get current position of follower
        follower_pos = {}
        for name in self.follower_arms:
            follower_pos[name] = self.follower_arms[name].get_state()

        # set teleop position for each arm as follower - leader
        for name in follower_pos:
            self.teleop_offset[name] = follower_pos[name] - leader_pos[name]

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "ARX5Robot is not connected. You need to run `robot.connect()` before disconnecting."
            )

        for name in self.follower_arms:
            self.follower_arms[name].disconnect()

        for name in self.leader_arms:
            self.leader_arms[name].disconnect()

        for name in self.cameras:
            self.cameras[name].disconnect()

        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
