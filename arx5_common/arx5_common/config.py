"""
Shared configuration types for ARX5 robot arms.
"""

from dataclasses import dataclass
from enum import Enum


DOF = 6  # Degrees of freedom for ARX5 arms

# Motor names for the 6-DOF arm + gripper
MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "wrist_rotate", "gripper"]

# EEF action keys for cartesian control mode
EEF_ACTION_KEYS = ["eef.x", "eef.y", "eef.z", "eef.roll", "eef.pitch", "eef.yaw", "eef.gripper"]


class ARXArmModel(Enum):
    """
    Two models of ARX arms are supported: the X5 and L5.
    
    The main difference between the two is the type of motor used in the three base joints.
    Ensure you are using the right arm model before starting the robot, as choosing the 
    wrong one can lead to dangerous movements.
    """
    X5 = "X5"
    L5 = "L5"


class ARXControlMode(Enum):
    """
    The ARX5 SDK supports two control modes: joint-based or cartesian space based.
    
    - JOINT_CONTROLLER: Position control where we specify the position in radians 
      for the six joints of the arm, as well as the gripper's width.
    - CARTESIAN_CONTROLLER: Position and orientation of the end-effector in 6D space, 
      along with the gripper's width.

    It is essential that models trained with one control mode are only rolled out 
    with the same control mode. Mixing them can result in dangerous behaviour!
    """
    JOINT_CONTROLLER = "joint_controller"
    CARTESIAN_CONTROLLER = "cartesian_controller"


@dataclass
class ARX5ArmConfig:
    """Configuration for a single ARX5 arm."""
    model: ARXArmModel
    interface_name: str

