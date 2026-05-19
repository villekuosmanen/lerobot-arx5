from dataclasses import dataclass, field

from lerobot.teleoperators.config import TeleoperatorConfig

from .config_arx5_leader import ARX5LeaderConfig


@TeleoperatorConfig.register_subclass("bi_arx5_leader")
@dataclass
class BiARX5LeaderConfig(TeleoperatorConfig):
    """Configuration for bimanual ARX5 leader teleoperator (two arms)."""
    left_arm_config: ARX5LeaderConfig = field(default_factory=ARX5LeaderConfig)
    right_arm_config: ARX5LeaderConfig = field(default_factory=ARX5LeaderConfig)
