"""Unit tests for evaluation utilities."""

import pytest
import numpy as np
import torch
import gymnasium as gym
from unittest.mock import Mock, patch

from src.models.networks import PolicyNetwork
from src.algorithms.evolutionary import SimpleEvolutionStrategy
from src.policies.evolutionary_agent import EvolutionaryRLAgent
from src.eval.evaluator import Evaluator


class TestEvaluator:
    """Test Evaluator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = Evaluator(log_dir="test_logs")
        self.env = gym.make("CartPole-v1")
        self.model = PolicyNetwork(input_size=4, output_size=2)
        self.evolutionary_algorithm = SimpleEvolutionStrategy(population_size=1)
        self.agent = EvolutionaryRLAgent(
            model=self.model,
            evolutionary_algorithm=self.evolutionary_algorithm,
            eval_episodes=1
        )
    
    def test_initialization(self):
        """Test evaluator initialization."""
        assert self.evaluator.log_dir.name == "test_logs"
        assert hasattr(self.evaluator, 'logger')
    
    def test_evaluate_agent(self):
        """Test agent evaluation."""
        # Mock the environment to return predictable results
        with patch.object(self.env, 'reset', return_value=(np.zeros(4), {})):
            with patch.object(self.env, 'step', return_value=(np.zeros(4), 1.0, True, False, {})):
                results = self.evaluator.evaluate_agent(
                    agent=self.agent,
                    env=self.env,
                    num_episodes=3,
                    save_results=False
                )
                
                assert 'episode_rewards' in results
                assert 'mean_reward' in results
                assert 'std_reward' in results
                assert 'success_rate' in results
                assert len(results['episode_rewards']) == 3
                assert isinstance(results['mean_reward'], float)
    
    def test_compare_agents(self):
        """Test agent comparison."""
        # Create multiple agents
        agent1 = EvolutionaryRLAgent(
            model=PolicyNetwork(input_size=4, output_size=2),
            evolutionary_algorithm=SimpleEvolutionStrategy(population_size=1),
            eval_episodes=1
        )
        agent2 = EvolutionaryRLAgent(
            model=PolicyNetwork(input_size=4, output_size=2),
            evolutionary_algorithm=SimpleEvolutionStrategy(population_size=1),
            eval_episodes=1
        )
        
        agents = {"agent1": agent1, "agent2": agent2}
        
        # Mock the environment
        with patch.object(self.env, 'reset', return_value=(np.zeros(4), {})):
            with patch.object(self.env, 'step', return_value=(np.zeros(4), 1.0, True, False, {})):
                results = self.evaluator.compare_agents(
                    agents=agents,
                    env=self.env,
                    num_episodes=2
                )
                
                assert 'agent1' in results
                assert 'agent2' in results
                assert 'statistical_tests' in results
                assert 'agent1_vs_agent2' in results['statistical_tests']
    
    def test_plot_training_curves(self):
        """Test training curve plotting."""
        training_history = {
            'generations': [1, 2, 3],
            'best_fitness': [1.0, 2.0, 3.0],
            'mean_fitness': [0.8, 1.8, 2.8],
            'std_fitness': [0.1, 0.2, 0.3],
            'evaluation_time': [0.1, 0.1, 0.1]
        }
        
        # Test that plotting doesn't raise errors
        try:
            self.evaluator.plot_training_curves(training_history)
        except Exception as e:
            pytest.fail(f"plot_training_curves raised {e}")
    
    def test_statistical_analysis(self):
        """Test statistical analysis functionality."""
        # Create sample data
        rewards1 = np.random.normal(10, 1, 100)
        rewards2 = np.random.normal(12, 1, 100)
        
        # Test that statistical analysis works
        from scipy import stats
        statistic, p_value = stats.mannwhitneyu(rewards1, rewards2, alternative='two-sided')
        
        assert isinstance(statistic, float)
        assert isinstance(p_value, float)
        assert 0 <= p_value <= 1


class TestEvaluationMetrics:
    """Test evaluation metrics calculation."""
    
    def test_confidence_interval_calculation(self):
        """Test confidence interval calculation."""
        evaluator = Evaluator()
        
        # Create sample data
        rewards = np.random.normal(10, 2, 100)
        
        # Calculate confidence interval
        from scipy import stats
        ci_lower, ci_upper = stats.t.interval(
            0.95, len(rewards) - 1,
            loc=np.mean(rewards), scale=stats.sem(rewards)
        )
        
        assert ci_lower < ci_upper
        assert ci_lower < np.mean(rewards) < ci_upper
    
    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        # Test with CartPole success criteria
        episode_lengths = [100, 200, 150, 500, 195]
        success_threshold = 195
        
        success_rate = sum(1 for length in episode_lengths if length >= success_threshold) / len(episode_lengths)
        
        assert success_rate == 0.4  # 2 out of 5 episodes successful


if __name__ == "__main__":
    pytest.main([__file__])
