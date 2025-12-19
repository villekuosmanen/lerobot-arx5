"""
Configuration for ARX5 leader teleoperator.
"""

from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig

from arx5_common import ARXArmModel, ARXControlMode, ARX5ArmConfig


@TeleoperatorConfig.register_subclass("arx5_leader")
@dataclass
class ARX5LeaderConfig(TeleoperatorConfig):
    """
    Configuration for a single ARX5 leader arm teleoperator.
    
    Usage:
        lerobot-teleoperate --teleop.type=arx5_leader --teleop.interface_name=enx6c6e0711f4e2
    """
    # Arm hardware configuration
    arm_model: ARXArmModel = ARXArmModel.L5
    interface_name: str = "can0"
    control_mode: ARXControlMode = ARXControlMode.JOINT_CONTROLLER

    @property
    def arm_config(self) -> ARX5ArmConfig:
        """Generate ARX5ArmConfig from this teleoperator config."""
        return ARX5ArmConfig(
            model=self.arm_model,
            interface_name=self.interface_name,
        )
