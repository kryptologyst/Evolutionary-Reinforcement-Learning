"""Unit tests for evolutionary RL agent."""

import pytest
import numpy as np
import torch
import gymnasium as gym
from unittest.mock import Mock, patch

from src.models.networks import PolicyNetwork
from src.algorithms.evolutionary import SimpleEvolutionStrategy
from src.policies.evolutionary_agent import EvolutionaryRLAgent


class TestEvolutionaryRLAgent:
    """Test Evolutionary RL Agent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.env = gym.make("CartPole-v1")
        self.model = PolicyNetwork(input_size=4, output_size=2)
        self.evolutionary_algorithm = SimpleEvolutionStrategy(population_size=5)
        self.agent = EvolutionaryRLAgent(
            model=self.model,
            evolutionary_algorithm=self.evolutionary_algorithm,
            eval_episodes=2
        )
    
    def test_initialization(self):
        """Test agent initialization."""
        assert self.agent.model == self.model
        assert self.agent.evolutionary_algorithm == self.evolutionary_algorithm
        assert self.agent.eval_episodes == 2
        assert self.agent.best_fitness == float('-inf')
        assert self.agent.best_model is None
    
    def test_evaluate_model(self):
        """Test model evaluation."""
        # Mock the environment to return predictable rewards
        with patch.object(self.env, 'reset', return_value=(np.zeros(4), {})):
            with patch.object(self.env, 'step', return_value=(np.zeros(4), 1.0, False, False, {})):
                fitness = self.agent.evaluate_model(self.model, self.env)
                
                assert isinstance(fitness, float)
                assert fitness > 0  # Should get positive reward
    
    def test_create_population(self):
        """Test population creation."""
        population = self.agent.create_population(self.model, 3)
        
        assert len(population) == 3
        assert all(isinstance(model, PolicyNetwork) for model in population)
        assert all(model.input_size == 4 for model in population)
        assert all(model.output_size == 2 for model in population)
    
    def test_get_action(self):
        """Test action selection."""
        state = np.random.randn(4)
        
        # Test with no best model (should use base model)
        action = self.agent.get_action(state)
        assert isinstance(action, int)
        assert 0 <= action < 2
        
        # Test with best model
        self.agent.best_model = self.model
        action = self.agent.get_action(state, deterministic=True)
        assert isinstance(action, int)
        assert 0 <= action < 2
    
    def test_training_history_tracking(self):
        """Test training history tracking."""
        # Simulate training data
        self.agent.training_history['generations'].append(1)
        self.agent.training_history['best_fitness'].append(10.0)
        self.agent.training_history['mean_fitness'].append(8.0)
        self.agent.training_history['std_fitness'].append(1.0)
        
        assert len(self.agent.training_history['generations']) == 1
        assert self.agent.training_history['best_fitness'][0] == 10.0
    
    def test_best_model_tracking(self):
        """Test best model tracking."""
        # Simulate finding a better model
        better_fitness = 15.0
        self.agent.best_fitness = 10.0
        
        # Create a mock better model
        better_model = PolicyNetwork(input_size=4, output_size=2)
        
        # Simulate updating best model
        if better_fitness > self.agent.best_fitness:
            self.agent.best_fitness = better_fitness
            self.agent.best_model = better_model
        
        assert self.agent.best_fitness == 15.0
        assert self.agent.best_model == better_model


class TestAgentIntegration:
    """Integration tests for the agent."""
    
    def test_agent_with_cartpole(self):
        """Test agent with CartPole environment."""
        env = gym.make("CartPole-v1")
        model = PolicyNetwork(input_size=4, output_size=2)
        evolutionary_algorithm = SimpleEvolutionStrategy(population_size=3)
        
        agent = EvolutionaryRLAgent(
            model=model,
            evolutionary_algorithm=evolutionary_algorithm,
            eval_episodes=1
        )
        
        # Test that agent can be created and initialized
        assert agent is not None
        assert agent.model is not None
        assert agent.evolutionary_algorithm is not None
    
    def test_agent_device_handling(self):
        """Test agent device handling."""
        device = torch.device("cpu")
        model = PolicyNetwork(input_size=4, output_size=2)
        evolutionary_algorithm = SimpleEvolutionStrategy(population_size=3)
        
        agent = EvolutionaryRLAgent(
            model=model,
            evolutionary_algorithm=evolutionary_algorithm,
            device=device
        )
        
        assert agent.device == device
        assert next(agent.model.parameters()).device == device


if __name__ == "__main__":
    pytest.main([__file__])
