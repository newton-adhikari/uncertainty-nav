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

ENV_B = EnvConfig(
    map_size=12.0,                    
    laser_noise_std=0.12,             
    occlusion_prob=0.20,              
    fov_deg=120.0,                    
    n_static_obstacles=10,            # moderate clutter
    dropout_prob=0.08,                # occasional dropout
    max_steps=600,                    # enough steps for larger map
    
    # Interior walls create corridors and perceptual aliasing
    interior_walls=(
        (-2.0, -4.0, -2.0, 2.0),     # vertical wall left
        (2.0, -2.0, 2.0, 4.0),       # vertical wall right
    ),
    n_dynamic_obstacles=2,            # 2 moving obstacles
    dynamic_speed=0.10,               # slower dynamics
)

# --- Distribution shift spectrum (Pillar 2) ---

# Env C: Sensor-only shift — Env A layout + Env B sensor degradation
# Isolates the effect of sensor noise/occlusion from layout change
ENV_C = EnvConfig(
    map_size=10.0,                    # same as Env A
    laser_noise_std=0.12,             # Env B noise
    occlusion_prob=0.20,              # Env B occlusion
    fov_deg=120.0,                    # Env B FoV
    n_static_obstacles=8,             # same as Env A
    dropout_prob=0.08,                # Env B dropout
    max_steps=500,                    # same as Env A
    interior_walls=(),                # no interior walls (Env A)
    n_dynamic_obstacles=0,            # no dynamic obstacles (Env A)
)

# Env D: Layout-only shift — Env B layout + Env A sensor parameters
# Isolates the effect of structural novelty from sensor degradation
ENV_D = EnvConfig(
    map_size=12.0,                    # Env B size
    laser_noise_std=0.05,             # Env A noise
    occlusion_prob=0.05,              # Env A occlusion
    fov_deg=180.0,                    # Env A FoV
    n_static_obstacles=10,            # Env B obstacles
    dropout_prob=0.0,                 # no dropout (Env A)
    max_steps=600,                    # Env B steps
    interior_walls=(                  # Env B walls
        (-2.0, -4.0, -2.0, 2.0),
        (2.0, -2.0, 2.0, 4.0),
    ),
    n_dynamic_obstacles=2,            # Env B dynamics
    dynamic_speed=0.10,
)

class PartialObsNavEnv(gym.Env):
    def __init__(self, config: EnvConfig = ENV_A, seed: Optional[int] = None):
        super().__init__()
        self.cfg = config
        self.rng = np.random.default_rng(seed)

        obs_dim = config.n_laser_beams + 3
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self._obstacles = []
        self._dynamic_obstacles = []  # list of (pos, velocity) pairs
        self._step = 0
        self._robot_pose = np.zeros(3)
        self._goal = np.zeros(2)
        self._prev_dist_to_goal = 0.0
        self._episode_path_length = 0.0
        self._optimal_path_length = 0.0

        # Parse interior walls into segments
        self._interior_walls = []
        for wall in config.interior_walls:
            self._interior_walls.append(
                (np.array([wall[0], wall[1]]), np.array([wall[2], wall[3]]))
            )

        # Precompute beam angles (relative to robot heading = 0)
        self._beam_angles_rel = np.linspace(
            0, 2 * np.pi, config.n_laser_beams, endpoint=False
        )
