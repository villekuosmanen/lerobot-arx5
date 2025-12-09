from enum import Enum


class ARXArmModel(Enum):
    """
    Two models of ARX arms are supported: the X5 and L5.
    The main difference between the two is the type of motor used in the three base joints.
    Ensure you are using the right arm model before starting the robot, as choosing the wrong one can lead to dangerous movements.
    """
    X5 = "X5"
    L5 = "L5"
