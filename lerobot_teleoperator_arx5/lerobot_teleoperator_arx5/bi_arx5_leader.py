from functools import cached_property
import logging

from lerobot.teleoperators.teleoperator import Teleoperator

from .config_bi_arx5_leader import BiARX5LeaderConfig
from .arx5_leader import ARX5Leader

logger = logging.getLogger(__name__)


class BiARX5Leader(Teleoperator):
    """Bimanual ARX5 leader teleoperator — two arms with left_/right_ key prefixing."""

    config_class = BiARX5LeaderConfig
    name = "bi_arx5_leader"

    def __init__(self, config: BiARX5LeaderConfig):
        super().__init__(config)
        self.config = config
        self.left_arm = ARX5Leader(config.left_arm_config)
        self.right_arm = ARX5Leader(config.right_arm_config)

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {
            **{f"left_{k}": v for k, v in self.left_arm.action_features.items()},
            **{f"right_{k}": v for k, v in self.right_arm.action_features.items()},
        }

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise RuntimeError(f"{self} is already connected")
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def reset(self):
        if not self.is_connected:
            raise RuntimeError(f"{self} is not connected.")
        self.left_arm.reset()
        self.right_arm.reset()

    def get_action(self):
        if not self.is_connected:
            raise RuntimeError(f"{self} is not connected.")
        action = {}
        action.update({f"left_{k}": v for k, v in self.left_arm.get_action().items()})
        action.update({f"right_{k}": v for k, v in self.right_arm.get_action().items()})
        return action

    def send_feedback(self, feedback):
        raise NotImplementedError("ARX5 does not support force feedback")

    def disconnect(self) -> None:
        if not self.is_connected:
            raise RuntimeError(f"{self} is not connected.")
        self.left_arm.disconnect()
        self.right_arm.disconnect()
