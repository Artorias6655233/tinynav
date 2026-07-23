from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class RobotConfig:
    """Robot geometry. Body frame uses +x forward, +y left, +z up."""

    name: str = "go2"
    shape: str = "square"
    length: float = 0.7
    width: float = 0.3
    radius: float = 0.3
    camera_x: float = 0.35
    camera_y: float = 0.0
    control_x: float = 0.0
    control_y: float = 0.0
    safety_radius: float = 0.1

    @property
    def cam_offset_3d(self) -> np.ndarray:
        """Offset [left, up, forward] from control center to camera in body frame."""
        return np.array(
            [self.camera_y - self.control_y, 0.0, self.camera_x - self.control_x],
            dtype=np.float32,
        )

    @property
    def half_size(self) -> tuple[float, float]:
        if self.shape == "circle":
            return (self.radius, self.radius)
        return (self.length / 2.0, self.width / 2.0)

    def footprint_from_control(self) -> tuple[float, float, float]:
        hl, hw = self.half_size
        return float(hl - self.control_x), float(hl + self.control_x), float(hw)


GO2_CONFIG = RobotConfig(
    name="go2",
    shape="square",
    length=0.4,
    width=0.3,
    camera_x=0.2,
    camera_y=0.0,
    control_x=0.0,
    control_y=0.0,
    safety_radius=0.2,
)


B2_CONFIG = RobotConfig(
    name="b2",
    shape="square",
    length=1.0,
    width=0.5,
    camera_x=0.5,
    camera_y=0.0,
    control_x=-0.5,
    control_y=0.0,
    safety_radius=0.1,
)


def camera_pose_to_control_center_pose(T_camera_in_world: np.ndarray, robot: RobotConfig) -> np.ndarray:
    T_control_in_world = np.array(T_camera_in_world, copy=True)
    T_control_in_world[:3, 3] = T_camera_in_world[:3, 3] - T_camera_in_world[:3, :3] @ robot.cam_offset_3d
    return T_control_in_world

