import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class EnvConfig:
    # Map
    map_size: float = 10.0
    n_static_obstacles: int = 6       # reduced from 8 — less crowded for early learning

    # Sensor — TurtleBot3 LDS-01 real specs
    n_laser_beams: int = 36           # subsampled from 360 for training speed
    max_range: float = 3.5
    min_range: float = 0.12
    fov_deg: float = 360.0

    # Noise
    laser_noise_std: float = 0.01
    occlusion_prob: float = 0.0
    dropout_prob: float = 0.0

    # Task
    goal_radius: float = 0.4          # slightly larger — easier to reach
    collision_radius: float = 0.18    # TurtleBot3 Waffle Pi ≈ 0.178 m
    obstacle_radius: float = 0.3      # half-size of obstacle for collision check
    max_steps: int = 500
    dt: float = 0.1

    # TurtleBot3 Waffle Pi velocity limits
    max_linear_vel: float = 0.26
    max_angular_vel: float = 1.82

    # Reward shaping
    goal_reward: float = 100.0
    collision_penalty: float = -10.0  # reduced: less catastrophic, easier to learn from
    step_penalty: float = -0.05       # reduced: agent not punished too hard for exploring
    progress_reward_scale: float = 1.0  # reward for moving toward goal

    # Interior walls for perceptual aliasing (list of (x1,y1,x2,y2) segments)
    interior_walls: tuple = ()

    # Dynamic obstacles
    n_dynamic_obstacles: int = 0
    dynamic_speed: float = 0.15       # m/s


ENV_A = EnvConfig(
    laser_noise_std=0.05,
    occlusion_prob=0.05,
    fov_deg=180.0,
    n_static_obstacles=8,
    dropout_prob=0.0,
)

class PartialObsNavEnv(gym.Env):
    def __init__(self, config: EnvConfig = ENV_A, seed: Optional[int] = None):
        super().__init__()
