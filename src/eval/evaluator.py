"""Evaluation utilities for evolutionary reinforcement learning."""

import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional, Union
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json


class Evaluator:
    """Comprehensive evaluator for evolutionary RL agents."""
    
    def __init__(self, log_dir: Optional[str] = None):
        """Initialize evaluator.
        
        Args:
            log_dir: Directory for saving evaluation results
        """
        self.log_dir = Path(log_dir) if log_dir else Path("evaluation_results")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def evaluate_agent(
        self,
        agent,
        env,
        num_episodes: int = 100,
        deterministic: bool = True,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """Evaluate an agent comprehensively.
        
        Args:
            agent: Agent to evaluate
            env: Environment to evaluate on
            num_episodes: Number of evaluation episodes
            deterministic: Whether to use deterministic actions
            save_results: Whether to save results to file
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.logger.info(f"Evaluating agent for {num_episodes} episodes")
        
        episode_rewards = []
        episode_lengths = []
        success_rate = 0
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0
            steps = 0
            max_steps = 1000
            
            while steps < max_steps:
                action = agent.get_action(state, deterministic=deterministic)
                next_state, reward, terminated, truncated, info = env.step(action)
                
                total_reward += reward
                steps += 1
                state = next_state
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            
            # Define success criteria (environment-specific)
            if hasattr(env, 'spec') and env.spec.id == 'CartPole-v1':
                success_rate += 1 if steps >= 195 else 0
            else:
                success_rate += 1 if total_reward > 0 else 0
        
        success_rate /= num_episodes
        
        # Calculate statistics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        median_reward = np.median(episode_rewards)
        
        # Confidence interval
        ci_lower, ci_upper = stats.t.interval(
            0.95, len(episode_rewards) - 1,
            loc=mean_reward, scale=stats.sem(episode_rewards)
        )
        
        mean_length = np.mean(episode_lengths)
        std_length = np.std(episode_lengths)
        
        results = {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'median_reward': median_reward,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'success_rate': success_rate,
            'mean_length': mean_length,
            'std_length': std_length,
            'num_episodes': num_episodes,
            'deterministic': deterministic
        }
        
        self.logger.info(f"Evaluation Results:")
        self.logger.info(f"  Mean Reward: {mean_reward:.4f} ± {std_reward:.4f}")
        self.logger.info(f"  Median Reward: {median_reward:.4f}")
        self.logger.info(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        self.logger.info(f"  Success Rate: {success_rate:.4f}")
        self.logger.info(f"  Mean Episode Length: {mean_length:.2f} ± {std_length:.2f}")
        
        if save_results:
            self._save_evaluation_results(results)
        
        return results
    
    def compare_agents(
        self,
        agents: Dict[str, Any],
        env,
        num_episodes: int = 100,
        deterministic: bool = True
    ) -> Dict[str, Any]:
        """Compare multiple agents.
        
        Args:
            agents: Dictionary of agent_name -> agent
            env: Environment to evaluate on
            num_episodes: Number of evaluation episodes per agent
            deterministic: Whether to use deterministic actions
            
        Returns:
            Comparison results
        """
        self.logger.info(f"Comparing {len(agents)} agents")
        
        comparison_results = {}
        
        for name, agent in agents.items():
            self.logger.info(f"Evaluating agent: {name}")
            results = self.evaluate_agent(
                agent, env, num_episodes, deterministic, save_results=False
            )
            comparison_results[name] = results
        
        # Statistical comparison
        agent_names = list(agents.keys())
        rewards_matrix = np.array([
            comparison_results[name]['episode_rewards'] for name in agent_names
        ])
        
        # Perform statistical tests
        statistical_tests = {}
        for i, name1 in enumerate(agent_names):
            for j, name2 in enumerate(agent_names):
                if i < j:
                    # Mann-Whitney U test (non-parametric)
                    statistic, p_value = stats.mannwhitneyu(
                        comparison_results[name1]['episode_rewards'],
                        comparison_results[name2]['episode_rewards'],
                        alternative='two-sided'
                    )
                    statistical_tests[f"{name1}_vs_{name2}"] = {
                        'statistic': statistic,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
        
        comparison_results['statistical_tests'] = statistical_tests
        
        # Create comparison plots
        self._plot_comparison(comparison_results, agent_names)
        
        # Save comparison results
        self._save_comparison_results(comparison_results)
        
        return comparison_results
    
    def plot_training_curves(
        self,
        training_history: Dict[str, List[float]],
        save_path: Optional[str] = None
    ) -> None:
        """Plot training curves.
        
        Args:
            training_history: Training history dictionary
            save_path: Path to save plot
        """
        plt.figure(figsize=(12, 8))
        
        # Plot fitness curves
        plt.subplot(2, 2, 1)
        generations = training_history['generations']
        best_fitness = training_history['best_fitness']
        mean_fitness = training_history['mean_fitness']
        std_fitness = training_history['std_fitness']
        
        plt.plot(generations, best_fitness, label='Best Fitness', linewidth=2)
        plt.plot(generations, mean_fitness, label='Mean Fitness', linewidth=2)
        plt.fill_between(
            generations,
            np.array(mean_fitness) - np.array(std_fitness),
            np.array(mean_fitness) + np.array(std_fitness),
            alpha=0.3, label='±1 Std'
        )
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot evaluation time
        plt.subplot(2, 2, 2)
        eval_times = training_history['evaluation_time']
        plt.plot(generations, eval_times, 'g-', linewidth=2)
        plt.xlabel('Generation')
        plt.ylabel('Evaluation Time (s)')
        plt.title('Evaluation Time per Generation')
        plt.grid(True, alpha=0.3)
        
        # Plot fitness distribution (last generation)
        plt.subplot(2, 2, 3)
        plt.hist(best_fitness, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Best Fitness')
        plt.ylabel('Frequency')
        plt.title('Distribution of Best Fitness Values')
        plt.grid(True, alpha=0.3)
        
        # Plot improvement over time
        plt.subplot(2, 2, 4)
        improvements = np.diff(best_fitness)
        plt.plot(generations[1:], improvements, 'r-', linewidth=2)
        plt.xlabel('Generation')
        plt.ylabel('Fitness Improvement')
        plt.title('Fitness Improvement per Generation')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Training curves saved to {save_path}")
        
        plt.show()
    
    def _plot_comparison(self, results: Dict[str, Any], agent_names: List[str]) -> None:
        """Plot comparison between agents."""
        plt.figure(figsize=(15, 10))
        
        # Reward comparison
        plt.subplot(2, 3, 1)
        rewards_data = [results[name]['episode_rewards'] for name in agent_names]
        plt.boxplot(rewards_data, labels=agent_names)
        plt.ylabel('Episode Reward')
        plt.title('Reward Distribution Comparison')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Mean reward comparison
        plt.subplot(2, 3, 2)
        mean_rewards = [results[name]['mean_reward'] for name in agent_names]
        std_rewards = [results[name]['std_reward'] for name in agent_names]
        
        plt.bar(agent_names, mean_rewards, yerr=std_rewards, capsize=5)
        plt.ylabel('Mean Episode Reward')
        plt.title('Mean Reward Comparison')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Success rate comparison
        plt.subplot(2, 3, 3)
        success_rates = [results[name]['success_rate'] for name in agent_names]
        plt.bar(agent_names, success_rates)
        plt.ylabel('Success Rate')
        plt.title('Success Rate Comparison')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Episode length comparison
        plt.subplot(2, 3, 4)
        lengths_data = [results[name]['episode_lengths'] for name in agent_names]
        plt.boxplot(lengths_data, labels=agent_names)
        plt.ylabel('Episode Length')
        plt.title('Episode Length Distribution')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Learning curves (if available)
        plt.subplot(2, 3, 5)
        for name in agent_names:
            if 'training_history' in results[name]:
                history = results[name]['training_history']
                if 'best_fitness' in history:
                    plt.plot(history['generations'], history['best_fitness'], label=name)
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.title('Learning Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Statistical significance heatmap
        plt.subplot(2, 3, 6)
        n_agents = len(agent_names)
        significance_matrix = np.ones((n_agents, n_agents))
        
        for i, name1 in enumerate(agent_names):
            for j, name2 in enumerate(agent_names):
                if i != j:
                    test_key = f"{name1}_vs_{name2}" if i < j else f"{name2}_vs_{name1}"
                    if test_key in results['statistical_tests']:
                        p_value = results['statistical_tests'][test_key]['p_value']
                        significance_matrix[i, j] = p_value
        
        sns.heatmap(significance_matrix, annot=True, fmt='.3f', 
                   xticklabels=agent_names, yticklabels=agent_names,
                   cmap='RdYlBu_r', vmin=0, vmax=1)
        plt.title('Statistical Significance (p-values)')
        
        plt.tight_layout()
        
        comparison_path = self.log_dir / "agent_comparison.png"
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Comparison plot saved to {comparison_path}")
        
        plt.show()
    
    def _save_evaluation_results(self, results: Dict[str, Any]) -> None:
        """Save evaluation results to file."""
        results_path = self.log_dir / "evaluation_results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, (np.floating, np.integer)):
                serializable_results[key] = float(value)
            else:
                serializable_results[key] = value
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Evaluation results saved to {results_path}")
    
    def _save_comparison_results(self, results: Dict[str, Any]) -> None:
        """Save comparison results to file."""
        comparison_path = self.log_dir / "comparison_results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if key == 'statistical_tests':
                serializable_results[key] = value
            else:
                if isinstance(value, dict):
                    serializable_results[key] = {}
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, np.ndarray):
                            serializable_results[key][subkey] = subvalue.tolist()
                        elif isinstance(subvalue, (np.floating, np.integer)):
                            serializable_results[key][subkey] = float(subvalue)
                        else:
                            serializable_results[key][subkey] = subvalue
                else:
                    serializable_results[key] = value
        
        with open(comparison_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Comparison results saved to {comparison_path}")
