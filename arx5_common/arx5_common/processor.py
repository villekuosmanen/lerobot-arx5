from dataclasses import dataclass

import torch

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    ActionProcessorStep,
    ObservationProcessorStep,
    ProcessorStepRegistry,
    RobotAction,
    RobotActionProcessorStep,
)
from lerobot.utils.constants import ACTION, OBS_STATE


@dataclass
@ProcessorStepRegistry.register("arx5_select_control_mode")
class ARX5SelectControlModeProcessorStep(ObservationProcessorStep):
    """
    Selects a control mode namespace and maps it to observation.state for policy training.
    
    Args:
        control_mode: Which control mode to use ("pos", "velocity", "eef_pose")
    """
    control_mode: str = "pos"  # or "velocity", "eef_pose", etc.
    
    def observation(self, observation: dict) -> dict:
        processed_obs = observation.copy()
        
        # Map the selected mode to observation.state
        if self.control_mode == "pos":
            if "observation.pos" in observation:
                processed_obs[OBS_STATE] = observation["observation.pos"]
        elif self.control_mode == "velocity":
            if "observation.velocity" in observation:
                processed_obs[OBS_STATE] = observation["observation.velocity"]
        elif self.control_mode == "eef_pose":
            if "observation.eef_pose" in observation:
                processed_obs[OBS_STATE] = observation["observation.eef_pose"]
        # Could also concatenate multiple modes:
        elif self.control_mode == "pos_velocity":
            pos = observation.get("observation.pos")
            vel = observation.get("observation.velocity")
            if pos is not None and vel is not None:
                processed_obs[OBS_STATE] = torch.cat([pos, vel], dim=-1)
        
        return processed_obs
    
    def transform_features(self, features):
        # Update the observation.state feature based on selected mode
        obs_features = features[PipelineFeatureType.OBSERVATION]
        
        if self.control_mode == "pos" and "observation.pos" in obs_features:
            obs_features[OBS_STATE] = obs_features["observation.pos"]
        elif self.control_mode == "velocity" and "observation.velocity" in obs_features:
            obs_features[OBS_STATE] = obs_features["observation.velocity"]
        elif self.control_mode == "eef_pose" and "observation.eef_pose" in obs_features:
            obs_features[OBS_STATE] = obs_features["observation.eef_pose"]
        # etc.
        
        return features

@dataclass
@ProcessorStepRegistry.register("arx5_select_action_mode")
class ARX5SelectActionModeProcessorStep(ActionProcessorStep):
    """Maps action namespace to the standard 'action' key for policy training."""
    action_mode: str = "pos"
    
    def action(self, action: dict) -> dict:
        processed_action = action.copy()
        
        if self.action_mode == "pos" and "action.pos" in action:
            processed_action[ACTION] = action["action.pos"]
        elif self.action_mode == "eef_pose" and "action.eef_pose" in action:
            processed_action[ACTION] = action["action.eef_pose"]
        
        return processed_action

@dataclass
@ProcessorStepRegistry.register("arx5_robot_observation_aggregator")
class ARX5RobotObservationAggregatorProcessorStep(ObservationProcessorStep):
    """
    Aggregates robot observations into namespaced features.
    This should have been part of the processor PR!
    """
    motor_names: list[str]
    include_velocity: bool = False
    include_effort: bool = False
    include_eef_pose: bool = False
    
    def observation(self, observation: dict) -> dict:
        processed_obs = observation.copy()
        
        # Extract positions
        positions = torch.tensor([
            observation[f"{name}.pos"] for name in self.motor_names
        ], dtype=torch.float32).unsqueeze(0)
        processed_obs["observation.pos"] = positions
        
        # Extract velocities if available
        if self.include_velocity:
            velocities = torch.tensor([
                observation[f"{name}.velocity"] for name in self.motor_names
            ], dtype=torch.float32).unsqueeze(0)
            processed_obs["observation.velocity"] = velocities
        
        # Extract efforts if available
        if self.include_effort:
            efforts = torch.tensor([
                observation[f"{name}.effort"] for name in self.motor_names
            ], dtype=torch.float32).unsqueeze(0)
            processed_obs["observation.effort"] = efforts
        
        # Extract EEF pose if available
        if self.include_eef_pose:
            eef_pose = torch.tensor([
                observation["eef.x"],
                observation["eef.y"],
                observation["eef.z"],
                observation["eef.roll"],
                observation["eef.pitch"],
                observation["eef.yaw"],
                observation["eef.gripper"],
            ], dtype=torch.float32).unsqueeze(0)
            processed_obs["observation.eef_pose"] = eef_pose
        
        return processed_obs
    
    def transform_features(self, features):
        obs_features = features[PipelineFeatureType.OBSERVATION]
        
        # Define the new namespaced features
        obs_features["observation.pos"] = PolicyFeature(
            type=FeatureType.STATE,
            shape=(len(self.motor_names),)
        )
        
        if self.include_velocity:
            obs_features["observation.velocity"] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(len(self.motor_names),)
            )
        
        if self.include_effort:
            obs_features["observation.effort"] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(len(self.motor_names),)
            )
        
        if self.include_eef_pose:
            obs_features["observation.eef_pose"] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(7,)
            )
        
        # Remove the individual motor keys
        for name in self.motor_names:
            obs_features.pop(f"{name}.pos", None)
            if self.include_velocity:
                obs_features.pop(f"{name}.velocity", None)
            if self.include_effort:
                obs_features.pop(f"{name}.effort", None)
        
        # Remove EEF individual components
        if self.include_eef_pose:
            for key in ["eef.x", "eef.y", "eef.z", "eef.roll", "eef.pitch", "eef.yaw", "eef.gripper"]:
                obs_features.pop(key, None)
        
        return features

@dataclass
@ProcessorStepRegistry.register("arx5_teleop_action_aggregator")
class ARX5TeleopActionAggregatorProcessorStep(RobotActionProcessorStep):
    """
    Aggregates teleop actions into namespaced features.
    Converts {motor_name}.pos, {motor_name}.velocity, etc. → action.pos, action.velocity
    """
    motor_names: list[str]
    include_velocity: bool = True
    include_effort: bool = True
    include_eef_pose: bool = True
    
    def action(self, action: RobotAction) -> RobotAction:
        processed_action = action.copy()
        
        # Extract positions
        positions = torch.tensor([
            action[f"{name}.pos"] for name in self.motor_names
        ], dtype=torch.float32).unsqueeze(0)
        processed_action["action.pos"] = positions
        
        if self.include_velocity:
            velocities = torch.tensor([
                action[f"{name}.velocity"] for name in self.motor_names
            ], dtype=torch.float32).unsqueeze(0)
            processed_action["action.velocity"] = velocities
        
        if self.include_effort:
            efforts = torch.tensor([
                action[f"{name}.effort"] for name in self.motor_names
            ], dtype=torch.float32).unsqueeze(0)
            processed_action["action.effort"] = efforts
        
        if self.include_eef_pose:
            eef_pose = torch.tensor([
                action["eef.x"], action["eef.y"], action["eef.z"],
                action["eef.roll"], action["eef.pitch"], action["eef.yaw"],
                action["eef.gripper"],
            ], dtype=torch.float32).unsqueeze(0)
            processed_action["action.eef_pose"] = eef_pose
        
        return processed_action
    
    def transform_features(self, features):
        action_features = features[PipelineFeatureType.ACTION]
        
        action_features["action.pos"] = PolicyFeature(
            type=FeatureType.ACTION, shape=(len(self.motor_names),)
        )
        
        if self.include_velocity:
            action_features["action.velocity"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(len(self.motor_names),)
            )
        
        if self.include_effort:
            action_features["action.effort"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(len(self.motor_names),)
            )
        
        if self.include_eef_pose:
            action_features["action.eef_pose"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(7,)
            )
        
        # Remove individual motor keys
        for name in self.motor_names:
            action_features.pop(f"{name}.pos", None)
            if self.include_velocity:
                action_features.pop(f"{name}.velocity", None)
            if self.include_effort:
                action_features.pop(f"{name}.effort", None)
        
        if self.include_eef_pose:
            for key in ["eef.x", "eef.y", "eef.z", "eef.roll", "eef.pitch", "eef.yaw", "eef.gripper"]:
                action_features.pop(key, None)
        
        return features
