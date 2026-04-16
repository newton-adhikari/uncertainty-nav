# we try to implement Epistemic Uncertainty via Deep Ensembles
# by training Train N independent policies from different random seeds


import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()

class DeepEnsemble(nn.Module):
    def __init__(self):
        super().__init__()

class EpistemicEnsemble(DeepEnsemble):
    def __init__(self):
        super().__init__()

class ValueNetwork(nn.Module):
    # Critic for PPO.

    def __init__(self):
        super().__init__()
