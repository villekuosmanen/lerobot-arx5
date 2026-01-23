"""
Low-level ARX5 arm controller.

This module provides the ARX5Arm class which handles direct communication
with the ARX5 hardware via the arx5_interface SDK.
"""

import math
import time
from typing import Tuple

import arx5_interface as arx5
import numpy as np

from .config import ARXControlMode, ARX5ArmConfig, DOF


class ARX5Arm:
    """
    Class for controlling a single ARX arm.
    
    This is a low-level interface used by both the Robot (follower) and 
    Teleoperator (leader) implementations.
    """

    def __init__(
        self,
        control_mode: ARXControlMode,
        config: ARX5ArmConfig,
        is_leader: bool = False,
    ):
        """
        Initialize an ARX5 arm controller.
        
        Args:
            control_mode: Joint or Cartesian control mode
            config: Arm configuration (model type, interface name)
            is_leader: True for leader/teleoperator arms, False for follower/robot arms
        """
        self.control_mode = control_mode
        self.config = config
        self.is_connected = False
        self.is_leader = is_leader
        self.robot_controller = None
        self.robot_config = None

    def connect(self) -> None:
        """Establish connection with the arm hardware."""
        if self.is_connected:
            raise RuntimeError(
                "ARX5Arm is already connected. Do not call connect() twice."
            )
        
        robot_config = arx5.RobotConfigFactory.get_instance().get_config(self.config.model.value)
        robot_config.gripper_torque_max *= 2
        
        controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
            self.control_mode.value, robot_config.joint_dof
        )
        # Sets the internal communication frequency (in seconds).
        # Slower CPU + USB I/O processing requires lower frequency comms.
        controller_config.controller_dt = 0.01
        controller_config.gravity_compensation = True
        controller_config.background_send_recv = True

        if self.control_mode == ARXControlMode.JOINT_CONTROLLER:
            self.robot_controller = arx5.Arx5JointController(
                robot_config, controller_config, self.config.interface_name
            )
        elif self.control_mode == ARXControlMode.CARTESIAN_CONTROLLER:
            self.robot_controller = arx5.Arx5CartesianController(
                robot_config, controller_config, self.config.interface_name
            )
        else:
            raise ValueError(
                f"Invalid arm control mode: expected {ARXControlMode.JOINT_CONTROLLER} "
                f"or {ARXControlMode.CARTESIAN_CONTROLLER}, got {self.control_mode}"
            )
        
        self.robot_controller.reset_to_home()
        self.robot_config = robot_config
        self.is_connected = True

    def disconnect(self) -> None:
        """Disconnect from the arm hardware."""
        if not self.is_connected:
            raise RuntimeError(
                "ARX5Arm is not connected. Cannot disconnect."
            )
        
        # Safely shut down follower arms
        if not self.is_leader:
            self.robot_controller.reset_to_home()
            self.robot_controller.set_to_damping()
        
        self.robot_controller = None
        time.sleep(0.5)
        self.is_connected = False

    def reset_to_home(self) -> None:
        """Reset the arm to its home position."""
        if not self.is_connected:
            raise RuntimeError("ARX5Arm is not connected.")
        self.robot_controller.reset_to_home()

    def configure(self) -> None:
        """
        Configure the arm for operation.
        
        Leader arms are set to damping mode with reduced gains for manual manipulation.
        Follower arms have their gains adjusted for smoother tracking.
        """
        if not self.is_connected:
            raise RuntimeError("ARX5Arm is not connected.")
        
        if self.is_leader:
            self.robot_controller.set_to_damping()
            gain = self.robot_controller.get_gain()
            gain.kd()[:3] *= 0.05
            gain.kd()[3:] *= 0.1
            self.robot_controller.set_gain(gain)
        else:
            # Follower arm - adjust gripper to be less aggressive
            gain = self.robot_controller.get_gain()
            gain.kd()[:3] /= 1.2
            gain.kd()[3:] *= 1.2
            gain.gripper_kp /= 1.8
            gain.gripper_kd *= 1.8
            self.robot_controller.set_gain(gain)

    def get_state(self) -> np.ndarray:
        """
        Get the current state of the arm.
        
        Returns:
            numpy array of shape (DOF+1,) containing joint positions and gripper position
        """
        if not self.is_connected:
            raise RuntimeError("ARX5Arm is not connected.")
        
        if self.control_mode == ARXControlMode.JOINT_CONTROLLER:
            joint_state = self.robot_controller.get_joint_state()
            state = np.concatenate([
                joint_state.pos().copy(), 
                np.array([joint_state.gripper_pos])
            ])
        else:
            eef_state = self.robot_controller.get_eef_state()
            state = np.concatenate([
                eef_state.pose_6d().copy(), 
                np.array([eef_state.gripper_pos])
            ])
        
        # Scale gripper for leader arms
        if self.is_leader:
            state[-1] *= 3.85
        
        return state

    def get_observation(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get full observation from the arm including position, velocity, effort, and EEF pose.
        
        Returns:
            Tuple of (position, velocity, effort, eef_pose) numpy arrays
        """
        if not self.is_connected:
            raise RuntimeError("ARX5Arm is not connected.")
        
        joint_state = self.robot_controller.get_joint_state()
        pos = np.concatenate([joint_state.pos().copy(), np.array([joint_state.gripper_pos])])
        vel = np.concatenate([joint_state.vel().copy(), np.array([joint_state.gripper_vel])])
        effort = np.concatenate([joint_state.torque().copy(), np.array([joint_state.gripper_torque])])
        
        if self.is_leader:
            pos[-1] *= 3.85
            vel[-1] *= 3.85
            effort[-1] *= 3.85

        eef_cartesians = self.robot_controller.get_eef_state()
        eef_pose = eef_cartesians.pose_6d().copy() # NOTE - we should append the gtipper position to this as well.
        
        return pos, vel, effort, eef_pose

    def send_command(self, action: np.ndarray) -> None:
        """
        Send a command to the arm.
        
        Args:
            action: numpy array of shape (DOF+1,) with joint positions and gripper position
        """
        if not self.is_connected:
            raise RuntimeError("ARX5Arm is not connected.")
        
        action = action.copy()
        
        # Rescale gripper width for leader arms
        if self.is_leader:
            action[DOF] /= 3.85
        
        # Clamp gripper to max width
        if action[DOF] > self.robot_config.gripper_width:
            action[DOF] = self.robot_config.gripper_width
        
        gripper_pos = action[DOF]
        
        if self.control_mode == ARXControlMode.JOINT_CONTROLLER:
            cmd = arx5.JointState(DOF)
            cmd.pos()[0:DOF] = action[0:DOF]
            cmd.gripper_pos = gripper_pos
            self.robot_controller.set_joint_cmd(cmd)
        else:
            cartesian_pos = action[0:DOF]
            eef_cmd = arx5.EEFState(cartesian_pos, gripper_pos)
            self.robot_controller.set_eef_cmd(eef_cmd)

    def interpolate_to_position(self, target: np.ndarray, duration: float = 3.5, fps: float = 30) -> None:
        """
        Smoothly interpolate the arm to a target position.
        
        Args:
            target: Target position array
            duration: Time in seconds for the interpolation
            fps: Frames per second for the interpolation
        """
        num_steps = math.ceil(duration * fps)
        current_pos = self.get_state()

        for i in range(num_steps + 1):
            start_loop_t = time.perf_counter()

            t = i / num_steps
            interp_pos = current_pos * (1 - t) + target * t
            self.send_command(interp_pos)

            dt_s = time.perf_counter() - start_loop_t
            sleep_time = 1 / fps - dt_s
            if sleep_time > 0:
                time.sleep(sleep_time)

