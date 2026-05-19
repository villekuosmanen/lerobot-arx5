from functools import cached_property
import logging

from lerobot.robots.robot import Robot

from .config_bi_arx5_follower import BiARX5FollowerConfig
from .arx5_follower import ARX5Follower

logger = logging.getLogger(__name__)


class BiARX5Follower(Robot):
    """Bimanual ARX5 follower robot — two arms with left_/right_ key prefixing."""

    config_class = BiARX5FollowerConfig
    name = "bi_arx5_follower"

    def __init__(self, config: BiARX5FollowerConfig):
        super().__init__(config)
        self.config = config
        self.left_arm = ARX5Follower(config.left_arm_config)
        self.right_arm = ARX5Follower(config.right_arm_config)
        self.cameras = {**self.left_arm.cameras, **self.right_arm.cameras}

    @cached_property
    def motor_names(self) -> list[str]:
        return (
            [f"left_{n}" for n in self.left_arm.motor_names]
            + [f"right_{n}" for n in self.right_arm.motor_names]
        )

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {
            **{f"left_{k}": v for k, v in self.left_arm.observation_features.items()},
            **{f"right_{k}": v for k, v in self.right_arm.observation_features.items()},
        }

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {
            **{f"left_{k}": v for k, v in self.left_arm.action_features.items()},
            **{f"right_{k}": v for k, v in self.right_arm.action_features.items()},
        }

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

    def get_observation(self):
        if not self.is_connected:
            raise RuntimeError(f"{self} is not connected.")
        obs = {}
        obs.update({f"left_{k}": v for k, v in self.left_arm.get_observation().items()})
        obs.update({f"right_{k}": v for k, v in self.right_arm.get_observation().items()})
        return obs

    def send_action(self, action):
        if not self.is_connected:
            raise RuntimeError(f"{self} is not connected.")
        left_action = {k.removeprefix("left_"): v for k, v in action.items() if k.startswith("left_")}
        right_action = {k.removeprefix("right_"): v for k, v in action.items() if k.startswith("right_")}
        sent_left = self.left_arm.send_action(left_action)
        sent_right = self.right_arm.send_action(right_action)
        return {
            **{f"left_{k}": v for k, v in sent_left.items()},
            **{f"right_{k}": v for k, v in sent_right.items()},
        }

    def disconnect(self) -> None:
        if not self.is_connected:
            raise RuntimeError(f"{self} is not connected.")
        self.left_arm.disconnect()
        self.right_arm.disconnect()
