"""Evolutionary algorithms for reinforcement learning."""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Any, Union
import logging
from abc import ABC, abstractmethod
import copy

from ..utils.utils import get_device
from ..models.networks import PolicyNetwork, ContinuousPolicyNetwork


class EvolutionaryAlgorithm(ABC):
    """Abstract base class for evolutionary algorithms."""
    
    def __init__(self, population_size: int, device: Optional[torch.device] = None):
        """Initialize evolutionary algorithm.
        
        Args:
            population_size: Size of the population
            device: Device to run on
        """
        self.population_size = population_size
        self.device = device or get_device()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def evolve(self, population: List[nn.Module], fitness_scores: List[float]) -> List[nn.Module]:
        """Evolve the population based on fitness scores.
        
        Args:
            population: Current population of models
            fitness_scores: Fitness scores for each model
            
        Returns:
            New evolved population
        """
        pass


class SimpleEvolutionStrategy(EvolutionaryAlgorithm):
    """Simple Evolution Strategy (ES) implementation.
    
    Uses mutation and selection to evolve policies.
    """
    
    def __init__(
        self,
        population_size: int,
        mutation_rate: float = 0.1,
        mutation_strength: float = 0.1,
        elite_size: int = 1,
        device: Optional[torch.device] = None
    ):
        """Initialize Simple ES.
        
        Args:
            population_size: Size of the population
            mutation_rate: Probability of mutating each parameter
            mutation_strength: Strength of mutations
            elite_size: Number of elite individuals to preserve
            device: Device to run on
        """
        super().__init__(population_size, device)
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.elite_size = elite_size
    
    def mutate(self, model: nn.Module) -> nn.Module:
        """Apply mutation to a model.
        
        Args:
            model: Model to mutate
            
        Returns:
            Mutated model
        """
        mutated_model = copy.deepcopy(model)
        
        with torch.no_grad():
            for param in mutated_model.parameters():
                if np.random.random() < self.mutation_rate:
                    noise = torch.randn_like(param) * self.mutation_strength
                    param.add_(noise)
        
        return mutated_model
    
    def evolve(self, population: List[nn.Module], fitness_scores: List[float]) -> List[nn.Module]:
        """Evolve the population.
        
        Args:
            population: Current population
            fitness_scores: Fitness scores
            
        Returns:
            New population
        """
        # Sort by fitness (descending)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        
        new_population = []
        
        # Keep elite individuals
        for i in range(min(self.elite_size, len(population))):
            elite_idx = sorted_indices[i]
            new_population.append(copy.deepcopy(population[elite_idx]))
        
        # Generate rest of population through mutation
        while len(new_population) < self.population_size:
            # Select parent (prefer better individuals)
            parent_idx = self._select_parent(sorted_indices, fitness_scores)
            parent = population[parent_idx]
            
            # Create offspring through mutation
            offspring = self.mutate(parent)
            new_population.append(offspring)
        
        return new_population
    
    def _select_parent(self, sorted_indices: np.ndarray, fitness_scores: List[float]) -> int:
        """Select parent for reproduction using fitness-proportional selection.
        
        Args:
            sorted_indices: Indices sorted by fitness
            fitness_scores: Fitness scores
            
        Returns:
            Selected parent index
        """
        # Use top 50% for parent selection
        top_half = sorted_indices[:len(sorted_indices)//2]
        
        # Fitness-proportional selection
        top_fitness = [fitness_scores[i] for i in top_half]
        min_fitness = min(top_fitness)
        
        # Shift fitness to be positive
        shifted_fitness = [f - min_fitness + 1e-8 for f in top_fitness]
        probabilities = np.array(shifted_fitness) / sum(shifted_fitness)
        
        selected_idx = np.random.choice(len(top_half), p=probabilities)
        return top_half[selected_idx]


class CMAES(EvolutionaryAlgorithm):
    """Covariance Matrix Adaptation Evolution Strategy (CMA-ES).
    
    Advanced ES variant that adapts the mutation distribution.
    """
    
    def __init__(
        self,
        population_size: int,
        initial_std: float = 0.1,
        device: Optional[torch.device] = None
    ):
        """Initialize CMA-ES.
        
        Args:
            population_size: Size of the population
            initial_std: Initial standard deviation
            device: Device to run on
        """
        super().__init__(population_size, device)
        self.initial_std = initial_std
        self.dim = None  # Will be set when we see the first model
        self.mean = None
        self.cov = None
        self.pc = None  # Evolution path for covariance
        self.ps = None  # Evolution path for step size
        self.sigma = initial_std
        
        # CMA-ES parameters
        self.mu = population_size // 2  # Number of parents
        self.lambda_ = population_size  # Number of offspring
        
        # Learning rates
        self.cc = 4.0 / (self.dim + 4) if self.dim else 0.0
        self.cs = 4.0 / (self.dim + 4) if self.dim else 0.0
        self.c1 = 2.0 / ((self.dim + 1.3) ** 2) if self.dim else 0.0
        self.cmu = min(1.0 - self.c1, 2.0 * (self.mu - 2 + 1.0/self.mu) / ((self.dim + 2) ** 2)) if self.dim else 0.0
        
        # Damping for step size
        self.damps = 1.0 + 2.0 * max(0, np.sqrt((self.mu - 1) / (self.dim + 1)) - 1) + self.cs if self.dim else 1.0
    
    def _extract_parameters(self, model: nn.Module) -> np.ndarray:
        """Extract parameters from model as flat array.
        
        Args:
            model: PyTorch model
            
        Returns:
            Flattened parameters
        """
        params = []
        for param in model.parameters():
            params.append(param.data.cpu().numpy().flatten())
        return np.concatenate(params)
    
    def _set_parameters(self, model: nn.Module, params: np.ndarray) -> None:
        """Set parameters in model from flat array.
        
        Args:
            model: PyTorch model
            params: Flattened parameters
        """
        idx = 0
        for param in model.parameters():
            param_size = param.numel()
            param.data = torch.from_numpy(params[idx:idx+param_size].reshape(param.shape)).to(self.device)
            idx += param_size
    
    def _initialize(self, model: nn.Module) -> None:
        """Initialize CMA-ES with model parameters.
        
        Args:
            model: Model to extract parameter dimensions from
        """
        params = self._extract_parameters(model)
        self.dim = len(params)
        
        # Initialize mean and covariance
        self.mean = np.zeros(self.dim)
        self.cov = np.eye(self.dim) * (self.sigma ** 2)
        
        # Initialize evolution paths
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        
        # Update learning rates
        self.cc = 4.0 / (self.dim + 4)
        self.cs = 4.0 / (self.dim + 4)
        self.c1 = 2.0 / ((self.dim + 1.3) ** 2)
        self.cmu = min(1.0 - self.c1, 2.0 * (self.mu - 2 + 1.0/self.mu) / ((self.dim + 2) ** 2))
        self.damps = 1.0 + 2.0 * max(0, np.sqrt((self.mu - 1) / (self.dim + 1)) - 1) + self.cs
    
    def evolve(self, population: List[nn.Module], fitness_scores: List[float]) -> List[nn.Module]:
        """Evolve population using CMA-ES.
        
        Args:
            population: Current population
            fitness_scores: Fitness scores
            
        Returns:
            New population
        """
        if self.dim is None:
            self._initialize(population[0])
        
        # Sort by fitness (descending)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        
        # Select parents (top mu individuals)
        parents = [population[i] for i in sorted_indices[:self.mu]]
        
        # Extract parameters from parents
        parent_params = [self._extract_parameters(parent) for parent in parents]
        
        # Update mean (weighted average of parents)
        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        weights = weights / np.sum(weights)
        
        new_mean = np.zeros(self.dim)
        for i, params in enumerate(parent_params):
            new_mean += weights[i] * params
        
        # Update evolution paths
        y = new_mean - self.mean
        self.mean = new_mean
        
        # Update step size evolution path
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mu) * y / np.sqrt(np.diag(self.cov))
        
        # Update covariance evolution path
        self.pc = (1 - self.cc) * self.pc + np.sqrt(self.cc * (2 - self.cc) * self.mu) * y
        
        # Update covariance matrix
        self.cov = (1 - self.c1 - self.cmu) * self.cov + self.c1 * np.outer(self.pc, self.pc)
        
        # Add rank-mu update
        for i, params in enumerate(parent_params):
            y_i = params - self.mean
            self.cov += self.cmu * weights[i] * np.outer(y_i, y_i)
        
        # Update step size
        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / np.sqrt(self.dim) - 1))
        
        # Generate new population
        new_population = []
        for _ in range(self.lambda_):
            # Sample from multivariate normal
            sample = np.random.multivariate_normal(self.mean, self.cov)
            
            # Create new model
            new_model = copy.deepcopy(population[0])
            self._set_parameters(new_model, sample)
            new_population.append(new_model)
        
        return new_population


class DifferentialEvolution(EvolutionaryAlgorithm):
    """Differential Evolution (DE) algorithm.
    
    Uses difference vectors for mutation and crossover.
    """
    
    def __init__(
        self,
        population_size: int,
        mutation_factor: float = 0.8,
        crossover_rate: float = 0.9,
        device: Optional[torch.device] = None
    ):
        """Initialize Differential Evolution.
        
        Args:
            population_size: Size of the population
            mutation_factor: Mutation factor (F)
            crossover_rate: Crossover rate (CR)
            device: Device to run on
        """
        super().__init__(population_size, device)
        self.mutation_factor = mutation_factor
        self.crossover_rate = crossover_rate
    
    def _extract_parameters(self, model: nn.Module) -> np.ndarray:
        """Extract parameters from model as flat array."""
        params = []
        for param in model.parameters():
            params.append(param.data.cpu().numpy().flatten())
        return np.concatenate(params)
    
    def _set_parameters(self, model: nn.Module, params: np.ndarray) -> None:
        """Set parameters in model from flat array."""
        idx = 0
        for param in model.parameters():
            param_size = param.numel()
            param.data = torch.from_numpy(params[idx:idx+param_size].reshape(param.shape)).to(self.device)
            idx += param_size
    
    def evolve(self, population: List[nn.Module], fitness_scores: List[float]) -> List[nn.Module]:
        """Evolve population using Differential Evolution.
        
        Args:
            population: Current population
            fitness_scores: Fitness scores
            
        Returns:
            New population
        """
        new_population = []
        
        for i in range(self.population_size):
            # Select three different individuals (not including current)
            candidates = list(range(self.population_size))
            candidates.remove(i)
            a, b, c = np.random.choice(candidates, 3, replace=False)
            
            # Extract parameters
            x_i = self._extract_parameters(population[i])
            x_a = self._extract_parameters(population[a])
            x_b = self._extract_parameters(population[b])
            x_c = self._extract_parameters(population[c])
            
            # Mutation: v = x_a + F * (x_b - x_c)
            v = x_a + self.mutation_factor * (x_b - x_c)
            
            # Crossover
            u = x_i.copy()
            crossover_mask = np.random.random(len(x_i)) < self.crossover_rate
            u[crossover_mask] = v[crossover_mask]
            
            # Create trial individual
            trial_model = copy.deepcopy(population[i])
            self._set_parameters(trial_model, u)
            
            # Selection: keep better individual
            trial_fitness = self._evaluate_model(trial_model)
            if trial_fitness > fitness_scores[i]:
                new_population.append(trial_model)
            else:
                new_population.append(population[i])
        
        return new_population
    
    def _evaluate_model(self, model: nn.Module) -> float:
        """Evaluate a single model (placeholder - should be implemented by user)."""
        # This is a placeholder - in practice, this would evaluate the model
        # on the environment and return the fitness score
        return np.random.random()
