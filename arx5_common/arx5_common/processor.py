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
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


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
        
        # Hack to handle images
        for key, val in observation.items():
            if len(val.shape) == 3:
                # assume it is an image
                processed_obs.pop(key)
                processed_obs[f"{OBS_IMAGES}.{key}"] = val

        # Map the selected mode to observation.state
        if self.control_mode == "pos":
            if "observation.state.pos" in observation:
                processed_obs[OBS_STATE] = observation["observation.state.pos"]
        elif self.control_mode == "velocity":
            if "observation.state.velocity" in observation:
                processed_obs[OBS_STATE] = observation["observation.state.velocity"]
        elif self.control_mode == "eef_pose":
            if "observation.state.eef_pose" in observation:
                processed_obs[OBS_STATE] = observation["observation.state.eef_pose"]
        

        return processed_obs
    
    def transform_features(self, features):
        # Update the observation.state feature based on selected mode
        obs_features = features[PipelineFeatureType.OBSERVATION]
        
        if self.control_mode == "pos" and "observation.state.pos" in obs_features:
            obs_features[OBS_STATE] = obs_features["observation.state.pos"]
        elif self.control_mode == "velocity" and "observation.state.velocity" in obs_features:
            obs_features[OBS_STATE] = obs_features["observation.state.velocity"]
        elif self.control_mode == "effort" and "observation.state.effort" in obs_features:
            obs_features[OBS_STATE] = obs_features["observation.state.effort"]
        elif self.control_mode == "eef_pose" and "observation.state.eef_pose" in obs_features:
            obs_features[OBS_STATE] = obs_features["observation.state.eef_pose"]
        # etc.

        # Hack to handle images
        for key, val in obs_features.items():
            if len(val.shape) == 3:
                # assume it is an image
                obs_features[f"{OBS_IMAGES}.{key}"] = val
        
        return {PipelineFeatureType.OBSERVATION: obs_features}

@dataclass
@ProcessorStepRegistry.register("arx5_select_action_mode")
class ARX5SelectActionModeProcessorStep(ActionProcessorStep):
    """Maps action namespace to the standard 'action' key for policy training."""
    action_mode: str = "pos"
    
    def action(self, action: dict) -> dict:
        processed_action = {}
        processed_action[f"{ACTION}.{self.action_mode}"] = action["action"].squeeze(0)
        # as our model only predicts current action mode, we need to set
        # everything else to zeroes
        processed_action[f"{ACTION}.velocity"] = torch.zeros_like(processed_action[f"{ACTION}.{self.action_mode}"])
        processed_action[f"{ACTION}.effort"] = torch.zeros_like(processed_action[f"{ACTION}.{self.action_mode}"])
        processed_action[f"{ACTION}.eef_pose"] = torch.zeros_like(processed_action[f"{ACTION}.{self.action_mode}"])
        return processed_action
    
    def transform_features(self, features):
        # Update the action feature based on selected mode
        action_features = features[PipelineFeatureType.ACTION]
        action_features [f"{ACTION}.{self.action_mode}"] = features[ACTION]
                
        return {PipelineFeatureType.ACTION: action_features}

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
        for name in self.motor_names:
            del(processed_obs[f"{name}.pos"])
        positions = torch.tensor([
            observation[f"{name}.pos"] for name in self.motor_names
        ], dtype=torch.float32)
        processed_obs["observation.state.pos"] = positions
        
        # Extract velocities if available
        for name in self.motor_names:
            del(processed_obs[f"{name}.velocity"])
        if self.include_velocity:
            velocities = torch.tensor([
                observation[f"{name}.velocity"] for name in self.motor_names
            ], dtype=torch.float32)
            processed_obs["observation.state.velocity"] = velocities
        
        # Extract efforts if available
        for name in self.motor_names:
            del(processed_obs[f"{name}.effort"])
        if self.include_effort:
            efforts = torch.tensor([
                observation[f"{name}.effort"] for name in self.motor_names
            ], dtype=torch.float32)
            processed_obs["observation.state.effort"] = efforts
        
        del(processed_obs["eef.x"])
        del(processed_obs["eef.y"])
        del(processed_obs["eef.z"])
        del(processed_obs["eef.roll"])
        del(processed_obs["eef.pitch"])
        del(processed_obs["eef.yaw"])
        del(processed_obs["eef.gripper"])
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
            ], dtype=torch.float32)
            processed_obs["observation.state.eef_pose"] = eef_pose
        
        return processed_obs
    
    def transform_features(self, features):
        obs_features = features[PipelineFeatureType.OBSERVATION]
        
        # Define the new namespaced features
        obs_features["pos"] = PolicyFeature(
            type=FeatureType.STATE,
            shape=(len(self.motor_names),)
        )
        
        if self.include_velocity:
            obs_features["velocity"] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(len(self.motor_names),)
            )
        
        if self.include_effort:
            obs_features["effort"] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(len(self.motor_names),)
            )
        
        if self.include_eef_pose:
            obs_features["eef_pose"] = PolicyFeature(
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
        
        return {PipelineFeatureType.OBSERVATION: obs_features}

@dataclass
@ProcessorStepRegistry.register("arx5_robot_action_aggregator")
class ARX5RobotActionAggregatorProcessorStep(RobotActionProcessorStep):
    """
    Dis-aggregates robot actions from namespaced features back into motor features.
    Converts action.pos, action.velocity → {motor_name}.pos, {motor_name}.velocity, etc.
    """
    motor_names: list[str]
    include_velocity: bool = True
    include_effort: bool = True
    include_eef_pose: bool = True
    
    def action(self, action: RobotAction) -> RobotAction:
        processed_action = {}
        
        # Extract positions
        for i, motor_name in enumerate(self.motor_names):
            processed_action[f"{motor_name}.pos"] = action["action.pos"].squeeze()[i]
            if self.include_velocity:
                processed_action[f"{motor_name}.velocity"] = action["action.velocity"].squeeze()[i]
            if self.include_effort:
                processed_action[f"{motor_name}.effort"] = action["action.effort"].squeeze()[i]
        
        if self.include_eef_pose:           
            processed_action["eef.x"] = action["action.eef_pose"].squeeze()[0]
            processed_action["eef.y"] = action["action.eef_pose"].squeeze()[1]
            processed_action["eef.z"] = action["action.eef_pose"].squeeze()[2]
            processed_action["eef.roll"] = action["action.eef_pose"].squeeze()[3]
            processed_action["eef.pitch"] = action["action.eef_pose"].squeeze()[4]
            processed_action["eef.yaw"] = action["action.eef_pose"].squeeze()[5]
            processed_action["eef.gripper"] = action["action.eef_pose"].squeeze()[6]
        
        return processed_action
    
    def transform_features(self, features):
        action_features = features[PipelineFeatureType.ACTION]
        
        action_features["pos"] = PolicyFeature(
            type=FeatureType.ACTION, shape=(len(self.motor_names),)
        )
        
        if self.include_velocity:
            action_features["velocity"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(len(self.motor_names),)
            )
        
        if self.include_effort:
            action_features["effort"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(len(self.motor_names),)
            )
        
        if self.include_eef_pose:
            action_features["eef_pose"] = PolicyFeature(
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
        
        return {PipelineFeatureType.ACTION: action_features}

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
        for name in self.motor_names:
            del(processed_action[f"{name}.pos"])
        positions = torch.tensor([
            action[f"{name}.pos"] for name in self.motor_names
        ], dtype=torch.float32)
        processed_action["action.pos"] = positions
        
        for name in self.motor_names:
            del(processed_action[f"{name}.velocity"])
        if self.include_velocity:
            velocities = torch.tensor([
                action[f"{name}.velocity"] for name in self.motor_names
            ], dtype=torch.float32)
            processed_action["action.velocity"] = velocities
        
        for name in self.motor_names:
            del(processed_action[f"{name}.effort"])
        if self.include_effort:
            efforts = torch.tensor([
                action[f"{name}.effort"] for name in self.motor_names
            ], dtype=torch.float32)
            processed_action["action.effort"] = efforts
        
        del(processed_action["eef.x"])
        del(processed_action["eef.y"])
        del(processed_action["eef.z"])
        del(processed_action["eef.roll"])
        del(processed_action["eef.pitch"])
        del(processed_action["eef.yaw"])
        del(processed_action["eef.gripper"])
        if self.include_eef_pose:
            eef_pose = torch.tensor([
                action["eef.x"], action["eef.y"], action["eef.z"],
                action["eef.roll"], action["eef.pitch"], action["eef.yaw"],
                action["eef.gripper"],
            ], dtype=torch.float32)
            processed_action["action.eef_pose"] = eef_pose
        
        return processed_action
    
    def transform_features(self, features):
        action_features = features[PipelineFeatureType.ACTION]
        
        action_features["pos"] = PolicyFeature(
            type=FeatureType.ACTION, shape=(len(self.motor_names),)
        )
        
        if self.include_velocity:
            action_features["velocity"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(len(self.motor_names),)
            )
        
        if self.include_effort:
            action_features["effort"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(len(self.motor_names),)
            )
        
        if self.include_eef_pose:
            action_features["eef_pose"] = PolicyFeature(
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
        
        return {PipelineFeatureType.ACTION: action_features}
