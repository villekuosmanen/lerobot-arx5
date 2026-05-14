from dataclasses import dataclass, field

from lerobot.robots.config import RobotConfig

from .config_arx5_follower import ARX5FollowerConfig


@RobotConfig.register_subclass("bi_arx5_follower")
@dataclass
class BiARX5FollowerConfig(RobotConfig):
    """Configuration for bimanual ARX5 follower robot (two arms)."""
    left_arm_config: ARX5FollowerConfig = field(default_factory=ARX5FollowerConfig)
    right_arm_config: ARX5FollowerConfig = field(default_factory=ARX5FollowerConfig)
