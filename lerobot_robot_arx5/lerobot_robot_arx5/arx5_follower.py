"""
ARX5 Follower Robot implementation.

This module provides the ARX5Follower class which implements the LeRobot Robot
interface for ARX5 robot arms acting as followers (receiving actions).
"""

from functools import cached_property
import logging
import time
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.robot import Robot

from arx5_common import ARX5Arm, ARXControlMode, DOF, MOTOR_NAMES, EEF_ACTION_KEYS

from .config_arx5_follower import ARX5FollowerConfig

logger = logging.getLogger(__name__)


class ARX5Follower(Robot):
    """
    ARX5 follower arm robot for LeRobot.
    
    This robot receives actions (typically from a policy or teleoperator) and
    executes them on the physical ARX5 arm hardware.
    """
    
    config_class = ARX5FollowerConfig
    name = "arx5_follower"

    def __init__(self, config: ARX5FollowerConfig):
        super().__init__(config)
        self.config = config
        
        # Create the low-level arm controller
        self.arm = ARX5Arm(
            control_mode=config.control_mode,
            config=config.arm_config,
            is_leader=False,
        )
        
        # Create cameras
        self.cameras = make_cameras_from_configs(config.cameras)


    @cached_property
    def _motor_pos_ft(self) -> dict[str, type]:
        """Motor position feature types (used for joint control actions)."""
        return {f"{name}.pos": float for name in MOTOR_NAMES}

    @cached_property
    def _motor_vel_ft(self) -> dict[str, type]:
        """Motor velocity feature types."""
        return {f"{name}.velocity": float for name in MOTOR_NAMES}

    @cached_property
    def _motor_eff_ft(self) -> dict[str, type]:
        """Motor effort/torque feature types."""
        return {f"{name}.effort": float for name in MOTOR_NAMES}

    @cached_property
    def _eef_ft(self) -> dict[str, type]:
        """End-effector pose feature types (x, y, z, roll, pitch, yaw, gripper)."""
        return {key: float for key in EEF_ACTION_KEYS}

    @cached_property
    def _cameras_ft(self) -> dict[str, tuple]:
        """Camera feature types."""
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def motor_names(self) -> dict[str, type]:
        return MOTOR_NAMES

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Features available in observations from this robot."""
        return {
            **self._motor_pos_ft,
            **self._motor_vel_ft,
            **self._motor_eff_ft,
            **self._eef_ft,
            **self._cameras_ft,
        }

    @cached_property
    def action_features(self) -> dict[str, type]:
        """
        Features expected in actions sent to this robot.
        
        - JOINT_CONTROLLER: Joint positions (shoulder_pan.pos, ..., gripper.pos)
        - CARTESIAN_CONTROLLER: EEF pose (eef.x, eef.y, eef.z, eef.roll, eef.pitch, eef.yaw, eef.gripper)
        """
        if self.config.control_mode == ARXControlMode.JOINT_CONTROLLER:
            return self._motor_pos_ft
        else:
            return self._eef_ft

    @property
    def is_connected(self) -> bool:
        """Whether the robot hardware is connected."""
        return self.arm.is_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        """
        Connect to the robot hardware.
        
        Args:
            calibrate: Whether to run calibration (ARX5 doesn't require manual calibration)
        """
        if self.is_connected:
            raise RuntimeError(f"{self} is already connected")
        
        logger.info(f"Connecting {self}...")
        self.arm.connect()
        time.sleep(0.2)
        
        # Connect cameras
        for cam in self.cameras.values():
            cam.connect()
        
        # Configure the arm
        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        """ARX5 arms are factory calibrated."""
        return True

    def calibrate(self) -> None:
        """ARX5 arms don't require manual calibration."""
        pass

    def configure(self) -> None:
        """Configure the arm for follower operation."""
        self.arm.configure()
    
    def reset(self):
        if not self.is_connected:
            raise RuntimeError(f"{self} is not connected.")
        self.arm.reset_to_home()

    def get_observation(self) -> dict[str, Any]:
        """
        Read current observation from the robot.
        
        Returns:
            Dictionary with motor positions, velocities, efforts, EEF pose, and camera images.
        """
        if not self.is_connected:
            raise RuntimeError(f"{self} is not connected.")
        
        # Read arm state
        start = time.perf_counter()
        pos, vel, effort, eef_pose = self.arm.get_observation()
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")
        
        # Build observation dict with named joints
        obs_dict = {}
        
        # Motor positions
        for i, name in enumerate(MOTOR_NAMES):
            obs_dict[f"{name}.pos"] = float(pos[i])
        
        # Motor velocities
        for i, name in enumerate(MOTOR_NAMES):
            obs_dict[f"{name}.velocity"] = float(vel[i])
        
        # Motor efforts (torques)
        for i, name in enumerate(MOTOR_NAMES):
            obs_dict[f"{name}.effort"] = float(effort[i])
        
        # End-effector pose (x, y, z, roll, pitch, yaw)
        obs_dict["eef.x"] = float(eef_pose[0])
        obs_dict["eef.y"] = float(eef_pose[1])
        obs_dict["eef.z"] = float(eef_pose[2])
        obs_dict["eef.roll"] = float(eef_pose[3])
        obs_dict["eef.pitch"] = float(eef_pose[4])
        obs_dict["eef.yaw"] = float(eef_pose[5])
        obs_dict["eef.gripper"] = float(pos[6])
        
        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")
        
        return obs_dict

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        """
        Send an action to the robot.
        
        Args:
            action: Dictionary mapping joint names (JOINT_CONTROLLER) or 
                    EEF pose keys (CARTESIAN_CONTROLLER) to target values.
            
        Returns:
            The action that was actually sent (may be clipped for safety).
        """
        if not self.is_connected:
            raise RuntimeError(f"{self} is not connected.")
        
        import numpy as np
        
        if self.config.control_mode == ARXControlMode.JOINT_CONTROLLER:
            # Extract joint positions from the action dict
            goal_pos = np.array([action[f"{name}.pos"] for name in MOTOR_NAMES])
            
            # Optional: Apply max_relative_target safety limit
            if self.config.max_relative_target is not None:
                current_pos, _, _, _ = self.arm.get_observation()
                delta = goal_pos - current_pos
                max_delta = self.config.max_relative_target
                delta = np.clip(delta, -max_delta, max_delta)
                goal_pos = current_pos + delta
            
            # Send to hardware
            self.arm.send_command(goal_pos)
            
            # Return what was sent
            return {f"{name}.pos": float(goal_pos[i]) for i, name in enumerate(MOTOR_NAMES)}
        else:
            # CARTESIAN_CONTROLLER: Extract EEF pose from the action dict
            goal_pos = np.array([action[key] for key in EEF_ACTION_KEYS])
            
            # Optional: Apply max_relative_target safety limit
            if self.config.max_relative_target is not None:
                pos, _, _, current_eef = self.arm.get_observation()
                current_pos = np.concatenate([current_eef, [pos[-1]]])  # EEF + gripper from joint state
                delta = goal_pos - current_pos
                max_delta = self.config.max_relative_target
                delta = np.clip(delta, -max_delta, max_delta)
                goal_pos = current_pos + delta
            
            # Send to hardware
            self.arm.send_command(goal_pos)
            
            # Return what was sent
            return {key: float(goal_pos[i]) for i, key in enumerate(EEF_ACTION_KEYS)}

    def disconnect(self) -> None:
        """Disconnect from the robot hardware."""
        if not self.is_connected:
            raise RuntimeError(f"{self} is not connected.")
        
        self.arm.disconnect()
        
        for cam in self.cameras.values():
            cam.disconnect()
        
        logger.info(f"{self} disconnected.")
