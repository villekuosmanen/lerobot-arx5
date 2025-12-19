"""
Configuration for ARX5 follower robot.
"""

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig

from arx5_common import ARXArmModel, ARXControlMode, ARX5ArmConfig


@RobotConfig.register_subclass("arx5_follower")
@dataclass
class ARX5FollowerConfig(RobotConfig):
    """
    Configuration for a single ARX5 follower arm robot.
    
    Usage:
        lerobot-control --robot.type=arx5_follower --robot.interface_name=enxa0cec881b947
    """
    # Arm hardware configuration
    arm_model: ARXArmModel = ARXArmModel.L5
    interface_name: str = "can0"
    control_mode: ARXControlMode = ARXControlMode.JOINT_CONTROLLER
    
    # Cameras attached to this robot
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    
    # Safety: max relative target limits the magnitude of position changes per step
    max_relative_target: float | None = None

    @property
    def arm_config(self) -> ARX5ArmConfig:
        """Generate ARX5ArmConfig from this robot config."""
        return ARX5ArmConfig(
            model=self.arm_model,
            interface_name=self.interface_name,
        )
