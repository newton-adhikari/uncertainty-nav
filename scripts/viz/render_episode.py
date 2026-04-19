# Renders a clean matplotlib animation showing:
# Robot trajectory color-coded by epistemic uncertainty (green→red)

from uncertainty_nav.models import PolicyNetwork, DeepEnsemble
from uncertainty_nav.baselines import VanillaMLP
from uncertainty_nav.nav_env import PartialObsNavEnv, ENV_A, ENV_B, ENV_C, ENV_D, EnvConfig

CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "experiments/videos"
ENV_MAP = {"A": ENV_A, "B": ENV_B, "C": ENV_C, "D": ENV_D}
