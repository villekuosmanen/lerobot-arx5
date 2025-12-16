from dataclasses import dataclass, field
from enum import Enum

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig

from lerobot_arx5 import ARXArmModel


class ARXArmModel(Enum):
    """
    Two models of ARX arms are supported: the X5 and L5.
    The main difference between the two is the type of motor used in the three base joints.
    Ensure you are using the right arm model before starting the robot, as choosing the wrong one can lead to dangerous movements.
    """
    X5 = "X5"
    L5 = "L5"

class ARXControlModel(Enum):
    """
    The ARX5 SDK supports two control modes: joint-based or cartesian space based.
    In this library, joint-based control refers to position control,
    where we specify the position in radians for the six joints of the arm, as well as the gripper's width.
    Cartesian control refers to the position and orientation of the end-effector in 6d space, along with the gripper's width.

    It is essential models trained one control mode are only rolled out with the same control mode. Mixing them can result in dangerous behaviour!
    """
    JOINT_CONTROLLER = "joint_controller"
    CARTESIAN_CONTROLLER = "cartesian_controller"

# Configs for ARX5 robot
@dataclass
class ARX5ArmConfig:
    model: ARXArmModel
    interface_name: str

@dataclass
class ARX5RobotConfig:
    control_mode: ARXControlModel
    leader_arms: dict[str, ARX5ArmConfig]
    follower_arms: dict[str, ARX5ArmConfig]
    cameras: dict[str, CameraConfig]

@RobotConfig.register_subclass("arx5")
@dataclass
class ARX5SingleArmRobotConfig(RobotConfig):
    """Single-armed ARX5 configuration with both leader and follower arms."""
    control_mode = ARXControlModel.JOINT_CONTROLLER
    leader_arms: dict[str, ARX5ArmConfig] = field(
        default_factory=lambda: {
            "main": ARX5ArmConfig(
                model=ARXArmModel.L5,
                interface_name="enx6c6e0711f4e2",
                urdf_path="lerobot/models/arx5.urdf",
            ),
        }
    )
    follower_arms: dict[str, ARX5ArmConfig] = field(
        default_factory=lambda: {
            "main": ARX5ArmConfig(
                model=ARXArmModel.L5,
                interface_name="enxa0cec881b947",
                urdf_path="lerobot/models/arx5.urdf",
            ),
        }
    )
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "front": CameraConfig(
                camera_index=2,
                fps=20,
                width=640,
                height=480,
            ),
            "wrist": CameraConfig(
                camera_index=0,
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
    control_mode = ARXControlModel.JOINT_CONTROLLER
    leader_arms: dict[str, ARX5ArmConfig] = field(
        default_factory=lambda: {
            "left": ARX5ArmConfig(
                model=ARXArmModel.L5,
                interface_name="enx6c6e0711f4e2",  
                urdf_path="lerobot/models/arx5.urdf",
            ),
            "right": ARX5ArmConfig(
                model=ARXArmModel.L5,
                interface_name="enxa0cec889459b",  
                urdf_path="lerobot/models/arx5.urdf",
            ),
        }
    )
    follower_arms: dict[str, ARX5ArmConfig] = field(
        default_factory=lambda: {
            "left": ARX5ArmConfig(
                model=ARXArmModel.L5,
                interface_name="enxa0cec881b947",  
                urdf_path="lerobot/models/arx5.urdf",
            ),
            "right": ARX5ArmConfig(
                model=ARXArmModel.L5,
                interface_name="enx6c6e0711f127",  
                urdf_path="lerobot/models/arx5.urdf",
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
            # "right_wrist": CameraConfig(
            #     camera_index=4,  
            #     fps=20,
            #     width=640,
            #     height=480,
            # ),
        }
    )

@RobotConfig.register_subclass("arx5_follow")
@dataclass
class ARX5SingleArmFollowOnlyConfig(RobotConfig):
    """Single-armed ARX5 configuration with follower arm only (for inference)."""
    control_mode = ARXControlModel.JOINT_CONTROLLER
    leader_arms: dict[str, ARX5ArmConfig] = field(default_factory=dict)
    follower_arms: dict[str, ARX5ArmConfig] = field(
        default_factory=lambda: {
            "main": ARX5ArmConfig(
                model=ARXArmModel.L5,
                interface_name="enxa0cec881b947",
                urdf_path="lerobot/models/arx5.urdf",
            ),
        }
    )
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "front": CameraConfig(
                camera_index=2,
                fps=20,
                width=640,
                height=480,
            ),
            "wrist": CameraConfig(
                camera_index=0,
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
    control_mode = ARXControlModel.JOINT_CONTROLLER
    leader_arms: dict[str, ARX5ArmConfig] = field(default_factory=dict)
    follower_arms: dict[str, ARX5ArmConfig] = field(
        default_factory=lambda: {
            "left": ARX5ArmConfig(
                model=ARXArmModel.L5,
                interface_name="enxa0cec881b947",  
                urdf_path="lerobot/models/arx5.urdf",
            ),
            "right": ARX5ArmConfig(
                model=ARXArmModel.L5,
                interface_name="enx6c6e0711f127",  
                urdf_path="lerobot/models/arx5.urdf",
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
