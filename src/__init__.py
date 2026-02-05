"""Evolutionary Reinforcement Learning Package."""

__version__ = "1.0.0"
__author__ = "Evolutionary RL Team"
__email__ = "contact@evolutionary-rl.com"

from .models import PolicyNetwork, ContinuousPolicyNetwork, ValueNetwork
from .algorithms import SimpleEvolutionStrategy, CMAES, DifferentialEvolution
from .policies import EvolutionaryRLAgent
from .eval import Evaluator
from .utils import set_seed, get_device, setup_logging

__all__ = [
    "PolicyNetwork",
    "ContinuousPolicyNetwork", 
    "ValueNetwork",
    "SimpleEvolutionStrategy",
    "CMAES",
    "DifferentialEvolution",
    "EvolutionaryRLAgent",
    "Evaluator",
    "set_seed",
    "get_device",
    "setup_logging"
]
