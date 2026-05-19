# LeRobot ARX5 Plugins

This repository provides [LeRobot](https://github.com/huggingface/lerobot) plugins for [ARX5 robot arms](https://arx-x.com/).

## Package Structure

This monorepo contains three packages:

| Package | Description |
|---------|-------------|
| `arx5-common` | Shared low-level arm control and configuration |
| `lerobot_robot_arx5` | Follower robot plugin (`arx5_follower`, `bi_arx5_follower`) |
| `lerobot_teleoperator_arx5` | Leader teleoperator plugin (`arx5_leader`, `bi_arx5_leader`) |

```
lerobot-arx5/
├── arx5_common/                          # Shared utilities
│   └── arx5_common/
│       ├── config.py                     # ARXArmModel, ARXControlMode, etc.
│       └── arm.py                        # ARX5Arm low-level controller
│
├── lerobot_robot_arx5/                   # Robot (follower) plugin
│   └── lerobot_robot_arx5/
│       ├── config_arx5_follower.py       # ARX5FollowerConfig
│       ├── arx5_follower.py              # ARX5Follower
│       ├── config_bi_arx5_follower.py    # BiARX5FollowerConfig
│       └── bi_arx5_follower.py           # BiARX5Follower
│
└── lerobot_teleoperator_arx5/            # Teleoperator (leader) plugin
    └── lerobot_teleoperator_arx5/
        ├── config_arx5_leader.py         # ARX5LeaderConfig
        ├── arx5_leader.py                # ARX5Leader
        ├── config_bi_arx5_leader.py      # BiARX5LeaderConfig
        └── bi_arx5_leader.py             # BiARX5Leader
```

## Installation

### For Development (editable install of all packages)

```bash
# Clone the repository
git clone https://github.com/yourusername/lerobot-arx5.git
cd lerobot-arx5

# Install all packages in development mode
pip install -e ./arx5_common
pip install -e ./lerobot_robot_arx5
pip install -e ./lerobot_teleoperator_arx5
```

### For Users (from PyPI)

```bash
# Install the follower robot plugin
pip install lerobot_robot_arx5

# Install the leader teleoperator plugin
pip install lerobot_teleoperator_arx5
```

## Usage

### Single-arm teleoperation

```bash
lerobot-teleoperate \
    --robot.type=arx5_follower \
    --robot.interface_name=enxa0cec881b947 \
    --teleop.type=arx5_leader \
    --teleop.interface_name=enxa0cec889459b
```

### Bimanual teleoperation

```bash
lerobot-teleoperate \
    --robot.type=bi_arx5_follower \
    --robot.left_arm_config.interface_name=enxa0cec881b947 \
    --robot.right_arm_config.interface_name=enx6c6e0711f127 \
    --teleop.type=bi_arx5_leader \
    --teleop.left_arm_config.interface_name=enxa0cec889459b \
    --teleop.right_arm_config.interface_name=enx6c6e0711f4e2
```

### Recording a Dataset

```bash
lerobot-record \
    --robot.type=arx5_follower \
    --robot.interface_name=enxa0cec881b947 \
    --teleop.type=arx5_leader \
    --teleop.interface_name=enx6c6e0711f4e2 \
    --repo-id=your-username/arx5-dataset
```

### Policy Inference (without leader arm)

```bash
lerobot-control \
    --robot.type=arx5_follower \
    --robot.interface_name=enxa0cec881b947 \
    --policy.path=your-username/arx5-policy
```

## Usage of Processors in pipelines:

During recording

```
# Define your observation processing pipeline
robot_obs_pipeline = RobotProcessorPipeline(
    steps=[
        ARX5RobotObservationAggregatorProcessorStep(
            motor_names=list(robot.bus.motors.keys()),
            include_velocity=True,
            include_effort=True,
            include_eef_pose=True,
        ),
        # Could add other steps like FK, image processing, etc.
    ],
    to_transition=observation_to_transition,  # Converter function
    to_output=transition_to_observation,
)

# Define your action processing pipeline  
teleop_action_pipeline = RobotProcessorPipeline(
    steps=[
        ARX5TeleopActionAggregatorProcessorStep(
            motor_names=list(teleop.bus.motors.keys()),
            include_velocity=True,
            include_effort=True,
            include_eef_pose=True,
        ),
        # Could add steps to convert teleop space to robot space
    ],
    to_transition=robot_action_to_transition,
    to_output=transition_to_robot_action,
)

# Compute what the dataset features will be AFTER pipeline processing
dataset_features = combine_feature_dicts(
    aggregate_pipeline_dataset_features(
        pipeline=robot_obs_pipeline,
        initial_features=create_initial_features(observation=robot.observation_features),
        use_videos=True,
    ),
    aggregate_pipeline_dataset_features(
        pipeline=teleop_action_pipeline,
        initial_features=create_initial_features(action=teleop.action_features),
        use_videos=False,
    ),
)

# Create dataset with the computed features
dataset = LeRobotDataset.create(
    repo_id="your_username/your_dataset",
    fps=30,
    features=dataset_features,  # This now has observation.pos, observation.eef_pose, action.pos, etc.
    robot_type=robot.name,
    use_videos=True,
)

# Recording loop
robot.connect()
teleop.connect()

while recording:
    # Get raw data
    raw_obs = robot.get_observation()  # {"shoulder.pos": 0.5, "shoulder.velocity": 0.1, ...}
    raw_action = teleop.get_action()    # {"shoulder.pos": 0.6, ...}
    
    # Process through pipelines
    processed_obs = robot_obs_pipeline(raw_obs)      # {"observation.pos": [0.5, ...], "observation.velocity": [...], ...}
    processed_action = teleop_action_pipeline(raw_action)  # {"action.pos": [0.6, ...], ...}
    
    # Add to dataset (dataset knows the schema from features)
    dataset.add_frame({
        **processed_obs,
        **processed_action,
        "reward": 0.0,
        "done": False,
    })

dataset.save_episode()
dataset.push_to_hub()
```

During training

```
# Create policy preprocessing pipeline
# This selects which namespace to use and maps to standard keys
policy_preprocessor = PolicyProcessorPipeline(
    steps=[
        ARX5SelectControlModeProcessorStep(control_mode="eef_pose"),  # observation.eef_pose → observation.state
        ARX5SelectActionModeProcessorStep(action_mode="eef_pose"),    # action.eef_pose → action
        NormalizerProcessorStep(
            features={
                "observation.state": dataset.stats["observation.eef_pose"],  # Use EEF stats
                "action": dataset.stats["action.eef_pose"],
            },
        ),
        DeviceProcessorStep(device="cuda"),
    ],
    name="preprocessor",
)

# Create policy postprocessing pipeline
policy_postprocessor = PolicyProcessorPipeline(
    steps=[
        UnnormalizerProcessorStep(
            features={
                "action": dataset.stats["action.eef_pose"],
            },
        ),
        DeviceProcessorStep(device="cpu"),
    ],
    name="postprocessor",
)
```


## Configuration Options

### ARX5FollowerConfig (Robot)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `arm_model` | `ARXArmModel` | `L5` | Arm model (X5 or L5) |
| `interface_name` | `str` | `"can0"` | Network interface for the arm |
| `control_mode` | `ARXControlMode` | `JOINT_CONTROLLER` | Control mode |
| `cameras` | `dict` | `{}` | Camera configurations |
| `max_relative_target` | `float \| None` | `None` | Safety limit for position changes |

### ARX5LeaderConfig (Teleoperator)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `arm_model` | `ARXArmModel` | `L5` | Arm model (X5 or L5) |
| `interface_name` | `str` | `"can0"` | Network interface for the arm |
| `control_mode` | `ARXControlMode` | `JOINT_CONTROLLER` | Control mode |

## Control Modes

- **JOINT_CONTROLLER**: Position control specifying 6 joint angles (radians) + gripper width
- **CARTESIAN_CONTROLLER**: End-effector pose in 6D space + gripper width

> ⚠️ **Important**: Models trained with one control mode must only be deployed with the same control mode!

## Arm Models

- **X5**: Uses different motors in the base joints
- **L5**: Uses different motors in the base joints

> ⚠️ **Important**: Ensure you select the correct arm model for your hardware!

## Requirements

- Python ≥ 3.10
- [LeRobot](https://github.com/huggingface/lerobot) ≥ 0.4
- [arx5-sdk](https://github.com/real-stanford/arx5-sdk) (automatically installed)

## License

Apache 2.0 - see [LICENSE](LICENSE) for details.
