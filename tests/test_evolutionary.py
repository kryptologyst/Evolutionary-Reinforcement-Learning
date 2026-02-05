"""Unit tests for evolutionary algorithms."""

import pytest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import Mock

from src.models.networks import PolicyNetwork, ContinuousPolicyNetwork
from src.algorithms.evolutionary import (
    SimpleEvolutionStrategy,
    CMAES,
    DifferentialEvolution
)


class TestSimpleEvolutionStrategy:
    """Test Simple Evolution Strategy."""
    
    def test_initialization(self):
        """Test ES initialization."""
        es = SimpleEvolutionStrategy(
            population_size=10,
            mutation_rate=0.1,
            mutation_strength=0.1
        )
        
        assert es.population_size == 10
        assert es.mutation_rate == 0.1
        assert es.mutation_strength == 0.1
    
    def test_mutation(self):
        """Test mutation operation."""
        es = SimpleEvolutionStrategy(population_size=10)
        
        # Create a simple model
        model = PolicyNetwork(input_size=4, output_size=2)
        original_params = [p.clone() for p in model.parameters()]
        
        # Apply mutation
        mutated_model = es.mutate(model)
        
        # Check that parameters changed
        for orig_param, mut_param in zip(original_params, mutated_model.parameters()):
            assert not torch.equal(orig_param, mut_param)
    
    def test_evolve(self):
        """Test evolution process."""
        es = SimpleEvolutionStrategy(population_size=5)
        
        # Create population
        population = [
            PolicyNetwork(input_size=4, output_size=2) for _ in range(5)
        ]
        fitness_scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        # Evolve population
        new_population = es.evolve(population, fitness_scores)
        
        assert len(new_population) == 5
        assert all(isinstance(model, PolicyNetwork) for model in new_population)


class TestCMAES:
    """Test CMA-ES algorithm."""
    
    def test_initialization(self):
        """Test CMA-ES initialization."""
        cmaes = CMAES(population_size=10)
        
        assert cmaes.population_size == 10
        assert cmaes.dim is None  # Will be set when we see first model
    
    def test_parameter_extraction(self):
        """Test parameter extraction and setting."""
        cmaes = CMAES(population_size=10)
        model = PolicyNetwork(input_size=4, output_size=2)
        
        # Extract parameters
        params = cmaes._extract_parameters(model)
        assert isinstance(params, np.ndarray)
        assert len(params) > 0
        
        # Set parameters
        new_params = params + 0.1
        cmaes._set_parameters(model, new_params)
        
        # Verify parameters changed
        updated_params = cmaes._extract_parameters(model)
        assert not np.array_equal(params, updated_params)
    
    def test_evolve(self):
        """Test CMA-ES evolution."""
        cmaes = CMAES(population_size=5)
        
        # Create population
        population = [
            PolicyNetwork(input_size=4, output_size=2) for _ in range(5)
        ]
        fitness_scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        # Evolve population
        new_population = cmaes.evolve(population, fitness_scores)
        
        assert len(new_population) == 5
        assert cmaes.dim is not None  # Should be initialized after first evolution


class TestDifferentialEvolution:
    """Test Differential Evolution algorithm."""
    
    def test_initialization(self):
        """Test DE initialization."""
        de = DifferentialEvolution(
            population_size=10,
            mutation_factor=0.8,
            crossover_rate=0.9
        )
        
        assert de.population_size == 10
        assert de.mutation_factor == 0.8
        assert de.crossover_rate == 0.9
    
    def test_evolve(self):
        """Test DE evolution."""
        de = DifferentialEvolution(population_size=5)
        
        # Create population
        population = [
            PolicyNetwork(input_size=4, output_size=2) for _ in range(5)
        ]
        fitness_scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        # Mock evaluation function
        de._evaluate_model = Mock(return_value=2.5)
        
        # Evolve population
        new_population = de.evolve(population, fitness_scores)
        
        assert len(new_population) == 5


class TestPolicyNetworks:
    """Test policy network implementations."""
    
    def test_policy_network_initialization(self):
        """Test PolicyNetwork initialization."""
        model = PolicyNetwork(input_size=4, output_size=2)
        
        assert model.input_size == 4
        assert model.output_size == 2
        assert len(model.layers) == 3  # 2 hidden + 1 output
    
    def test_policy_network_forward(self):
        """Test PolicyNetwork forward pass."""
        model = PolicyNetwork(input_size=4, output_size=2)
        x = torch.randn(1, 4)
        
        output = model(x)
        
        assert output.shape == (1, 2)
        assert torch.allclose(output.sum(dim=-1), torch.ones(1), atol=1e-6)  # Probabilities sum to 1
    
    def test_policy_network_get_action(self):
        """Test PolicyNetwork action selection."""
        model = PolicyNetwork(input_size=4, output_size=2)
        state = np.random.randn(4)
        
        action = model.get_action(state)
        
        assert isinstance(action, int)
        assert 0 <= action < 2
    
    def test_continuous_policy_network_initialization(self):
        """Test ContinuousPolicyNetwork initialization."""
        model = ContinuousPolicyNetwork(input_size=4, output_size=2)
        
        assert model.input_size == 4
        assert model.output_size == 2
    
    def test_continuous_policy_network_forward(self):
        """Test ContinuousPolicyNetwork forward pass."""
        model = ContinuousPolicyNetwork(input_size=4, output_size=2)
        x = torch.randn(1, 4)
        
        mean, log_std = model(x)
        
        assert mean.shape == (1, 2)
        assert log_std.shape == (1, 2)
    
    def test_continuous_policy_network_get_action(self):
        """Test ContinuousPolicyNetwork action selection."""
        model = ContinuousPolicyNetwork(input_size=4, output_size=2)
        state = np.random.randn(4)
        
        # Deterministic action
        action = model.get_action(state, deterministic=True)
        assert isinstance(action, torch.Tensor)
        assert action.shape == (2,)
        
        # Stochastic action
        action, log_prob = model.get_action(state, deterministic=False)
        assert isinstance(action, torch.Tensor)
        assert isinstance(log_prob, torch.Tensor)
        assert action.shape == (2,)
        assert log_prob.shape == (1,)


if __name__ == "__main__":
    pytest.main([__file__])
