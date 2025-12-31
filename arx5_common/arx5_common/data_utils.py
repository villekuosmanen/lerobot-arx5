from typing import Any

import numpy as np
from lerobot.datasets.utils import DEFAULT_FEATURES


def build_dataset_frame(
    ds_features: dict[str, dict], values: dict[str, Any], prefix: str
) -> dict[str, np.ndarray]:
    """Construct a single data frame from raw values based on dataset features.

    A "frame" is a dictionary containing all the data for a single timestep,
    formatted as numpy arrays according to the feature specification.

    Args:
        ds_features (dict): The LeRobot dataset features dictionary.
        values (dict): A dictionary of raw values from the hardware/environment.
        prefix (str): The prefix to filter features by (e.g., "observation"
            or "action").

    Returns:
        dict: A dictionary representing a single frame of data.
    """
    frame = {}
    for key, ft in ds_features.items():
        if key in DEFAULT_FEATURES or not key.startswith(prefix):
            continue
        elif ft["dtype"] == "float32" and len(ft["shape"]) == 1:
            frame[key] = values[key]
        elif ft["dtype"] in ["image", "video"]:
            frame[key] = values[key.removeprefix(f"{prefix}.images.")]

    return frame
