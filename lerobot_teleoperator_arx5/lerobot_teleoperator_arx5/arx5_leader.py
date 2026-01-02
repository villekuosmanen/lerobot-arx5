"""
ARX5 Leader Teleoperator implementation.

This module provides the ARX5Leader class which implements the LeRobot Teleoperator
interface for ARX5 robot arms acting as leaders (producing actions from human input).
"""

from functools import cached_property
import logging
import time

from lerobot.teleoperators.teleoperator import Teleoperator

from arx5_common import ARX5Arm, MOTOR_NAMES, EEF_ACTION_KEYS

from .config_arx5_leader import ARX5LeaderConfig

logger = logging.getLogger(__name__)


class ARX5Leader(Teleoperator):
    """
    ARX5 leader arm teleoperator for LeRobot.
    
    This teleoperator reads the position of a physical ARX5 arm that a human
    operator manipulates, producing actions that can be sent to a follower robot.
    """
    
    config_class = ARX5LeaderConfig
    name = "arx5_leader"
    
    def __init__(self, config: ARX5LeaderConfig):
        super().__init__(config)
        self.config = config
        
        # Create the low-level arm controller
        self.arm = ARX5Arm(
            control_mode=config.control_mode,
            config=config.arm_config,
            is_leader=True,
        )

    @cached_property
    def _motor_pos_ft(self) -> dict[str, type]:
        """Motor position feature types."""
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
        """End-effector pose feature types."""
        return {key: float for key in EEF_ACTION_KEYS}
    
    @cached_property
    def motor_names(self) -> dict[str, type]:
        return MOTOR_NAMES

    @cached_property
    def action_features(self) -> dict[str, type]:
        """
        Features produced by this teleoperator (sent to follower robot).
        
        Includes motor positions, velocities, efforts, and EEF pose.
        """
        return {
            **self._motor_pos_ft,
            **self._motor_vel_ft,
            **self._motor_eff_ft,
            **self._eef_ft,
        }

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        """Features that can be sent back to this teleoperator as feedback."""
        # ARX5 doesn't currently support force feedback
        return {}

    @property
    def is_connected(self) -> bool:
        """Whether the teleoperator hardware is connected."""
        return self.arm.is_connected

    def connect(self, calibrate: bool = True) -> None:
        """
        Connect to the teleoperator hardware.
        
        Args:
            calibrate: Whether to run calibration (ARX5 doesn't require manual calibration)
        """
        if self.is_connected:
            raise RuntimeError(f"{self} is already connected")
        
        logger.info(f"Connecting {self}...")
        self.arm.connect()
        time.sleep(0.2)
        
        # Configure the arm for leader operation (low resistance for manual manipulation)
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
        """Configure the arm for leader/teleoperator operation."""
        self.arm.configure()

    def reset(self):
        if not self.is_connected:
            raise RuntimeError(f"{self} is not connected.")
        self.arm.reset_to_home()

    def get_action(self) -> dict[str, float]:
        """
        Read the current state of the leader arm as an action.
        
        Returns:
            Dictionary with motor positions, velocities, efforts, and EEF pose
            that can be sent to a follower robot.
        """
        if not self.is_connected:
            raise RuntimeError(f"{self} is not connected.")
        
        start = time.perf_counter()
        pos, vel, effort, eef_pose = self.arm.get_observation()
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        
        action = {}
        
        # Motor positions
        for i, name in enumerate(MOTOR_NAMES):
            action[f"{name}.pos"] = float(pos[i])
        
        # Motor velocities
        for i, name in enumerate(MOTOR_NAMES):
            action[f"{name}.velocity"] = float(vel[i])
        
        # Motor efforts (torques)
        for i, name in enumerate(MOTOR_NAMES):
            action[f"{name}.effort"] = float(effort[i])
        
        # End-effector pose (x, y, z, roll, pitch, yaw, gripper)
        action["eef.x"] = float(eef_pose[0])
        action["eef.y"] = float(eef_pose[1])
        action["eef.z"] = float(eef_pose[2])
        action["eef.roll"] = float(eef_pose[3])
        action["eef.pitch"] = float(eef_pose[4])
        action["eef.yaw"] = float(eef_pose[5])
        action["eef.gripper"] = float(pos[6])  # Gripper from joint state
        
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        """
        Send feedback to the teleoperator (e.g., force feedback).
        
        ARX5 doesn't currently support force feedback.
        """
        raise NotImplementedError("ARX5 does not support force feedback")

    def disconnect(self) -> None:
        """Disconnect from the teleoperator hardware."""
        if not self.is_connected:
            raise RuntimeError(f"{self} is not connected.")
        
        self.arm.disconnect()
        logger.info(f"{self} disconnected.")
