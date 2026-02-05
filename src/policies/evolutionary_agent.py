"""Evolutionary Reinforcement Learning Agent."""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from pathlib import Path
import json
import time

from ..utils.utils import get_device, save_checkpoint, EarlyStopping
from ..models.networks import PolicyNetwork, ContinuousPolicyNetwork
from ..algorithms.evolutionary import EvolutionaryAlgorithm, SimpleEvolutionStrategy, CMAES, DifferentialEvolution


class EvolutionaryRLAgent:
    """Evolutionary Reinforcement Learning Agent.
    
    Uses evolutionary algorithms to evolve neural network policies for RL tasks.
    Supports multiple evolutionary strategies and environments.
    """
    
    def __init__(
        self,
        model: Union[PolicyNetwork, ContinuousPolicyNetwork],
        evolutionary_algorithm: EvolutionaryAlgorithm,
        device: Optional[torch.device] = None,
        eval_episodes: int = 10,
        max_episode_steps: int = 1000,
        save_frequency: int = 10,
        log_dir: Optional[str] = None
    ):
        """Initialize Evolutionary RL Agent.
        
        Args:
            model: Policy network to evolve
            evolutionary_algorithm: Evolutionary algorithm to use
            device: Device to run on
            eval_episodes: Number of episodes for evaluation
            max_episode_steps: Maximum steps per episode
            save_frequency: Frequency to save checkpoints
            log_dir: Directory for logging
        """
        self.model = model
        self.evolutionary_algorithm = evolutionary_algorithm
        self.device = device or get_device()
        self.eval_episodes = eval_episodes
        self.max_episode_steps = max_episode_steps
        self.save_frequency = save_frequency
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.training_history = {
            'generations': [],
            'best_fitness': [],
            'mean_fitness': [],
            'std_fitness': [],
            'evaluation_time': []
        }
        
        # Best model tracking
        self.best_fitness = float('-inf')
        self.best_model = None
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=20, min_delta=1e-4)
    
    def evaluate_model(self, model: nn.Module, env, deterministic: bool = False) -> float:
        """Evaluate a model on the environment.
        
        Args:
            model: Model to evaluate
            env: Environment to evaluate on
            deterministic: Whether to use deterministic actions
            
        Returns:
            Average reward over evaluation episodes
        """
        model.eval()
        total_rewards = []
        
        for _ in range(self.eval_episodes):
            state, _ = env.reset()
            total_reward = 0
            steps = 0
            
            while steps < self.max_episode_steps:
                with torch.no_grad():
                    if isinstance(model, PolicyNetwork):
                        action = model.get_action(state, deterministic=deterministic)
                    else:  # ContinuousPolicyNetwork
                        action = model.get_action(state, deterministic=deterministic)
                        if isinstance(action, tuple):
                            action = action[0]  # Get action, not log_prob
                        action = action.cpu().numpy()
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                state = next_state
                steps += 1
                
                if terminated or truncated:
                    break
            
            total_rewards.append(total_reward)
        
        return np.mean(total_rewards)
    
    def create_population(self, base_model: nn.Module, population_size: int) -> List[nn.Module]:
        """Create initial population from base model.
        
        Args:
            base_model: Base model to create population from
            population_size: Size of population
            
        Returns:
            List of models forming the population
        """
        population = []
        
        for _ in range(population_size):
            # Clone the base model
            individual = type(base_model)(
                input_size=base_model.input_size,
                output_size=base_model.output_size,
                hidden_sizes=getattr(base_model, 'hidden_sizes', (64, 64))
            )
            individual.load_state_dict(base_model.state_dict())
            individual.to(self.device)
            population.append(individual)
        
        return population
    
    def train(
        self,
        env,
        num_generations: int = 100,
        population_size: Optional[int] = None,
        verbose: bool = True,
        save_best: bool = True
    ) -> Dict[str, Any]:
        """Train the agent using evolutionary reinforcement learning.
        
        Args:
            env: Environment to train on
            num_generations: Number of generations to evolve
            population_size: Size of population (overrides algorithm setting)
            verbose: Whether to print progress
            save_best: Whether to save best model
            
        Returns:
            Training results dictionary
        """
        start_time = time.time()
        
        # Set population size
        if population_size is not None:
            self.evolutionary_algorithm.population_size = population_size
        
        # Create initial population
        population = self.create_population(self.model, self.evolutionary_algorithm.population_size)
        
        self.logger.info(f"Starting evolutionary training for {num_generations} generations")
        self.logger.info(f"Population size: {len(population)}")
        self.logger.info(f"Device: {self.device}")
        
        for generation in range(num_generations):
            generation_start = time.time()
            
            # Evaluate population
            fitness_scores = []
            for i, individual in enumerate(population):
                fitness = self.evaluate_model(individual, env)
                fitness_scores.append(fitness)
                
                if verbose and i % 10 == 0:
                    self.logger.info(f"Generation {generation+1}, Individual {i+1}/{len(population)}, Fitness: {fitness:.4f}")
            
            # Track best individual
            best_idx = np.argmax(fitness_scores)
            best_fitness = fitness_scores[best_idx]
            
            if best_fitness > self.best_fitness:
                self.best_fitness = best_fitness
                self.best_model = population[best_idx]
            
            # Calculate statistics
            mean_fitness = np.mean(fitness_scores)
            std_fitness = np.std(fitness_scores)
            
            # Update training history
            self.training_history['generations'].append(generation + 1)
            self.training_history['best_fitness'].append(best_fitness)
            self.training_history['mean_fitness'].append(mean_fitness)
            self.training_history['std_fitness'].append(std_fitness)
            self.training_history['evaluation_time'].append(time.time() - generation_start)
            
            # Log progress
            if verbose:
                self.logger.info(
                    f"Generation {generation+1}/{num_generations} - "
                    f"Best: {best_fitness:.4f}, Mean: {mean_fitness:.4f} ± {std_fitness:.4f}"
                )
            
            # Save checkpoint
            if (generation + 1) % self.save_frequency == 0:
                self._save_checkpoint(generation + 1, best_fitness)
            
            # Early stopping check
            if self.early_stopping(best_fitness, self.best_model):
                self.logger.info(f"Early stopping at generation {generation+1}")
                break
            
            # Evolve population (except for last generation)
            if generation < num_generations - 1:
                population = self.evolutionary_algorithm.evolve(population, fitness_scores)
        
        # Final evaluation
        final_time = time.time() - start_time
        self.logger.info(f"Training completed in {final_time:.2f} seconds")
        self.logger.info(f"Best fitness achieved: {self.best_fitness:.4f}")
        
        # Save final results
        results = {
            'best_fitness': self.best_fitness,
            'training_time': final_time,
            'generations_completed': len(self.training_history['generations']),
            'training_history': self.training_history
        }
        
        if save_best and self.best_model is not None:
            self._save_best_model()
        
        self._save_training_results(results)
        
        return results
    
    def _save_checkpoint(self, generation: int, fitness: float) -> None:
        """Save training checkpoint."""
        checkpoint_path = self.log_dir / f"checkpoint_gen_{generation}.pt"
        torch.save({
            'generation': generation,
            'fitness': fitness,
            'model_state_dict': self.best_model.state_dict() if self.best_model else self.model.state_dict(),
            'training_history': self.training_history
        }, checkpoint_path)
    
    def _save_best_model(self) -> None:
        """Save the best model."""
        if self.best_model is not None:
            model_path = self.log_dir / "best_model.pt"
            torch.save(self.best_model.state_dict(), model_path)
            self.logger.info(f"Best model saved to {model_path}")
    
    def _save_training_results(self, results: Dict[str, Any]) -> None:
        """Save training results to JSON."""
        results_path = self.log_dir / "training_results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if key == 'training_history':
                serializable_results[key] = {
                    k: [float(x) for x in v] if isinstance(v, np.ndarray) else v
                    for k, v in value.items()
                }
            else:
                serializable_results[key] = float(value) if isinstance(value, (np.floating, np.integer)) else value
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Training results saved to {results_path}")
    
    def load_best_model(self, model_path: str) -> None:
        """Load the best model from file.
        
        Args:
            model_path: Path to saved model
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        self.logger.info(f"Model loaded from {model_path}")
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Union[int, np.ndarray]:
        """Get action from the current model.
        
        Args:
            state: Current state
            deterministic: Whether to use deterministic action
            
        Returns:
            Action to take
        """
        model = self.best_model if self.best_model is not None else self.model
        return model.get_action(state, deterministic=deterministic)
