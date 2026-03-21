from collections.abc import Sequence
from typing import Any

from lerobot.processor.converters import (
    observation_to_transition,
    robot_action_observation_to_transition,
    robot_action_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.types import RobotAction, RobotObservation
from lerobot.processor.pipeline import RobotProcessorPipeline
from lerobot.configs.types import PipelineFeatureType
from lerobot.processor import DataProcessorPipeline
from lerobot.datasets.pipeline_features import should_keep, strip_prefix, PREFIXES_TO_STRIP
from lerobot.datasets.feature_utils import _validate_feature_names
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STR


from .processor import (
    ARX5RobotObservationAggregatorProcessorStep,
    ARX5RobotActionAggregatorProcessorStep,
    ARX5TeleopActionAggregatorProcessorStep,
)


def make_arx5_teleop_action_processor(teleop) -> RobotProcessorPipeline[
    tuple[RobotAction, RobotObservation], RobotAction
]:
    teleop_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            ARX5TeleopActionAggregatorProcessorStep(
                motor_names=list(teleop.motor_names),
                include_velocity=True,
                include_effort=True,
                include_eef_pose=True,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    return teleop_action_processor


def make_arx5_robot_action_processor(robot) -> RobotProcessorPipeline[
    tuple[RobotAction, RobotObservation], RobotAction
]:
    robot_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            ARX5RobotActionAggregatorProcessorStep(
                motor_names=list(robot.motor_names),
                include_velocity=False,
                include_effort=False,
                include_eef_pose=False,
            ),
        ],
        to_transition=robot_action_to_transition,
        to_output=transition_to_robot_action,
    )
    return robot_action_processor


def make_arx5_robot_observation_processor(robot) -> RobotProcessorPipeline[RobotObservation, RobotObservation]:
    robot_observation_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[
            ARX5RobotObservationAggregatorProcessorStep(
                motor_names=list(robot.motor_names),
                include_velocity=True,
                include_effort=True,
                include_eef_pose=True,
            ),
        ],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )
    return robot_observation_processor


def make_arx_processors(teleop, robot):
    teleop_action_processor = make_arx5_teleop_action_processor(teleop)
    robot_action_processor = make_arx5_robot_action_processor(robot)
    robot_observation_processor = make_arx5_robot_observation_processor(robot)
    return (teleop_action_processor, robot_action_processor, robot_observation_processor)


def hw_to_dataset_features(
    hw_features: dict[str, type | tuple], prefix: str, use_video: bool = True
) -> dict[str, dict]:
    """Convert hardware-specific features to a LeRobot dataset feature dictionary.

    This function takes a dictionary describing hardware outputs (like joint states
    or camera image shapes) and formats it into the standard LeRobot feature
    specification.

    Args:
        hw_features (dict): Dictionary mapping feature names to their type (float for
            joints) or shape (tuple for images).
        prefix (str): The prefix to add to the feature keys (e.g., "observation"
            or "action").
        use_video (bool): If True, image features are marked as "video", otherwise "image".

    Returns:
        dict: A LeRobot features dictionary.
    """
    features = {}
    joint_fts = {
        key: ftype
        for key, ftype in hw_features.items()
        if ftype is float or (isinstance(ftype, PolicyFeature) and ftype.type != FeatureType.VISUAL)
    }
    cam_fts = {key: shape for key, shape in hw_features.items() if isinstance(shape, tuple)}

    for name, ft in joint_fts.items():
        if prefix == ACTION:
            features[f"{prefix}.{name}"] = {
                "dtype": "float32",
                "shape": ft.shape,
                "names": [f"joint_{i}" for i in range(ft.shape[-1])],
            }
        elif prefix == OBS_STR:
            features[f"{prefix}.state.{name}"] = {
                "dtype": "float32",
                "shape": ft.shape,
                "names": [f"joint_{i}" for i in range(ft.shape[-1])],
            }

    for key, shape in cam_fts.items():
        features[f"{prefix}.images.{key}"] = {
            "dtype": "video" if use_video else "image",
            "shape": shape,
            "names": ["height", "width", "channels"],
        }

    _validate_feature_names(features)
    return features

def aggregate_pipeline_dataset_features(
    pipeline: DataProcessorPipeline,
    initial_features: dict[PipelineFeatureType, dict[str, Any]],
    *,
    use_videos: bool = True,
    patterns: Sequence[str] | None = None,
) -> dict[str, dict]:
    """
    Aggregates and filters pipeline features to create a dataset-ready features dictionary.

    This function transforms initial features using the pipeline, categorizes them as action or observations
    (image or state), filters them based on `use_videos` and `patterns`, and finally
    formats them for use with a Hugging Face LeRobot Dataset.

    Args:
        pipeline: The DataProcessorPipeline to apply.
        initial_features: A dictionary of raw feature specs for actions and observations.
        use_videos: If False, image features are excluded.
        patterns: A sequence of regex patterns to filter action and state features.
                  Image features are not affected by this filter.

    Returns:
        A dictionary of features formatted for a Hugging Face LeRobot Dataset.
    """
    all_features = pipeline.transform_features(initial_features)

    # Intermediate storage for categorized and filtered features.
    processed_features: dict[str, dict[str, Any]] = {
        ACTION: {},
        OBS_STR: {},
    }
    images_token = OBS_IMAGES.split(".")[-1]

    # Iterate through all features transformed by the pipeline.
    for ptype, feats in all_features.items():
        if ptype not in [PipelineFeatureType.ACTION, PipelineFeatureType.OBSERVATION]:
            continue

        for key, value in feats.items():
            # 1. Categorize the feature.
            is_action = ptype == PipelineFeatureType.ACTION
            # Observations are classified as images if their key matches image-related tokens or if the shape of the feature is 3.
            # All other observations are treated as state.
            is_image = not is_action and (
                (isinstance(value, tuple) and len(value) == 3)
                or (
                    key.startswith(f"{OBS_IMAGES}.")
                    or key.startswith(f"{images_token}.")
                    or f".{images_token}." in key
                )
            )

            # 2. Apply filtering rules.
            if is_image and not use_videos:
                continue
            if not is_image and not should_keep(key, patterns):
                continue

            # 3. Add the feature to the appropriate group with a clean name.
            name = strip_prefix(key, PREFIXES_TO_STRIP)
            if is_action:
                processed_features[ACTION][name] = value
            else:
                processed_features[OBS_STR][name] = value

    # Convert the processed features into the final dataset format.
    dataset_features = {}
    if processed_features[ACTION]:
        dataset_features.update(hw_to_dataset_features(processed_features[ACTION], ACTION, use_videos))
    if processed_features[OBS_STR]:

        dataset_features.update(hw_to_dataset_features(processed_features[OBS_STR], OBS_STR, use_videos))

    return dataset_features
