from dataclasses import dataclass, field

from lerobot.robots.config import RobotConfig

from .config_arx5_follower import ARX5FollowerBaseConfig


@RobotConfig.register_subclass("bi_arx5_follower")
@dataclass
class BiARX5FollowerConfig(RobotConfig):
    """Configuration for bimanual ARX5 follower robot (two arms)."""

    # Use the non-registered base config for nested arms. If these fields use
    # the registered `ARX5FollowerConfig` choice class, draccus recursively
    # expands RobotConfig -> bi_arx5_follower -> ARX5FollowerConfig ->
    # RobotConfig -> ... while building parser actions.
    left_arm_config: ARX5FollowerBaseConfig = field(default_factory=ARX5FollowerBaseConfig)
    right_arm_config: ARX5FollowerBaseConfig = field(default_factory=ARX5FollowerBaseConfig)
