from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig

from lerobot_arx5 import ARXArmModel


# Configs for ARX5 robot
@dataclass
class ARX5ArmConfig:
    model: str
    interface_name: str

@dataclass
class ARX5RobotConfig:
    leader_arms: dict[str, ARX5ArmConfig]
    follower_arms: dict[str, ARX5ArmConfig]
    cameras: dict[str, CameraConfig]

@RobotConfig.register_subclass("arx5")
@dataclass
class ARX5SingleArmRobotConfig(RobotConfig):
    """Single-armed ARX5 configuration with both leader and follower arms."""
    leader_arms: dict[str, ARX5ArmConfig] = field(
        default_factory=lambda: {
            "main": ARX5ArmConfig(
                model=ARXArmModel.L5,
                interface_name="enx5c5310ecc0ec",
            ),
        }
    )
    follower_arms: dict[str, ARX5ArmConfig] = field(
        default_factory=lambda: {
            "main": ARX5ArmConfig(
                model=ARXArmModel.L5,
                interface_name="enx5c5310ecc38c",
            ),
        }
    )
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "front": CameraConfig(
                camera_index=0,
                fps=20,
                width=640,
                height=480,
            ),
            "wrist": CameraConfig(
                camera_index=2,
                fps=20,
                width=640,
                height=480,
            ),
        }
    )

@RobotConfig.register_subclass("arx5_bimanual")
@dataclass
class ARX5BimanualRobotConfig(RobotConfig):
    """Bimanual ARX5 configuration with both leader and follower arms."""
    leader_arms: dict[str, ARX5ArmConfig] = field(
        default_factory=lambda: {
            "left": ARX5ArmConfig(
                model=ARXArmModel.L5,
                interface_name="enx5c5310ecc0ec",
            ),
            "right": ARX5ArmConfig(
                model=ARXArmModel.L5,
                interface_name="enx5c5310ec6299",
            ),
        }
    )
    follower_arms: dict[str, ARX5ArmConfig] = field(
        default_factory=lambda: {
            "left": ARX5ArmConfig(
                model=ARXArmModel.L5,
                interface_name="enx5c5310ecc38c",
            ),
            "right": ARX5ArmConfig(
                model=ARXArmModel.L5,
                interface_name="enx5c5310ecc90b",
            ),
        }
    )
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "front": CameraConfig(
                camera_index=0,
                fps=20,
                width=640,
                height=480,
            ),
            "left_wrist": CameraConfig(
                camera_index=2,
                fps=20,
                width=640,
                height=480,
            ),
            "right_wrist": CameraConfig(
                camera_index=4,  
                fps=20,
                width=640,
                height=480,
            ),
        }
    )

@RobotConfig.register_subclass("arx5_follow")
@dataclass
class ARX5SingleArmFollowOnlyConfig(RobotConfig):
    """Single-armed ARX5 configuration with follower arm only (for inference)."""
    leader_arms: dict[str, ARX5ArmConfig] = field(default_factory=dict)
    follower_arms: dict[str, ARX5ArmConfig] = field(
        default_factory=lambda: {
            "main": ARX5ArmConfig(
                model=ARXArmModel.L5,
                interface_name="enx5c5310ecc38c",
            ),
        }
    )
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "front": CameraConfig(
                camera_index=0,
                fps=20,
                width=640,
                height=480,
            ),
            "wrist": CameraConfig(
                camera_index=2,
                fps=20,
                width=640,
                height=480,
            ),
        }
    )

@RobotConfig.register_subclass("arx5_bimanual_follow")
@dataclass
class ARX5BimanualFollowOnlyConfig(RobotConfig):
    """Bimanual ARX5 configuration with follower arms only (for inference)."""
    leader_arms: dict[str, ARX5ArmConfig] = field(default_factory=dict)
    follower_arms: dict[str, ARX5ArmConfig] = field(
        default_factory=lambda: {
            "left": ARX5ArmConfig(
                model=ARXArmModel.L5,
                interface_name="enx5c5310ecc38c",
            ),
            "right": ARX5ArmConfig(
                model=ARXArmModel.L5,
                interface_name="enx5c5310ecc90b",  
            ),
        }
    )
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "front": CameraConfig(
                camera_index=0,
                fps=20,
                width=640,
                height=480,
            ),
            "left_wrist": CameraConfig(
                camera_index=2,
                fps=20,
                width=640,
                height=480,
            ),
            "right_wrist": CameraConfig(
                camera_index=4,  
                fps=20,
                width=640,
                height=480,
            ),
        }
    )
