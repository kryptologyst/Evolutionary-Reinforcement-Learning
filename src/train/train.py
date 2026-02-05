"""Training script for evolutionary reinforcement learning."""

import argparse
import logging
import gymnasium as gym
import numpy as np
import torch
from pathlib import Path
import yaml
from typing import Dict, Any

from src.utils.utils import set_seed, get_device, setup_logging
from src.models.networks import PolicyNetwork, ContinuousPolicyNetwork
from src.algorithms.evolutionary import SimpleEvolutionStrategy, CMAES, DifferentialEvolution
from src.policies.evolutionary_agent import EvolutionaryRLAgent
from src.eval.evaluator import Evaluator


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Evolutionary RL Agent")
    
    # Environment arguments
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Environment name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Training arguments
    parser.add_argument("--generations", type=int, default=100, help="Number of generations")
    parser.add_argument("--population-size", type=int, default=20, help="Population size")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Evaluation episodes per individual")
    
    # Algorithm arguments
    parser.add_argument("--algorithm", type=str, default="simple_es", 
                       choices=["simple_es", "cmaes", "differential_evolution"],
                       help="Evolutionary algorithm to use")
    parser.add_argument("--mutation-rate", type=float, default=0.1, help="Mutation rate")
    parser.add_argument("--mutation-strength", type=float, default=0.1, help="Mutation strength")
    
    # Model arguments
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[64, 64], 
                       help="Hidden layer sizes")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    
    # Logging and saving
    parser.add_argument("--log-dir", type=str, default="logs", help="Log directory")
    parser.add_argument("--save-frequency", type=int, default=10, help="Save frequency")
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level")
    
    # Evaluation
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate, don't train")
    parser.add_argument("--model-path", type=str, help="Path to model for evaluation")
    
    return parser.parse_args()


def create_environment(env_name: str, seed: int) -> gym.Env:
    """Create and configure environment.
    
    Args:
        env_name: Name of the environment
        seed: Random seed
        
    Returns:
        Configured environment
    """
    env = gym.make(env_name)
    env.reset(seed=seed)
    return env


def create_model(env: gym.Env, hidden_sizes: tuple, dropout: float) -> torch.nn.Module:
    """Create policy network model.
    
    Args:
        env: Environment to get input/output sizes from
        hidden_sizes: Hidden layer sizes
        dropout: Dropout rate
        
    Returns:
        Policy network model
    """
    input_size = env.observation_space.shape[0]
    
    if isinstance(env.action_space, gym.spaces.Discrete):
        output_size = env.action_space.n
        model = PolicyNetwork(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            dropout=dropout
        )
    else:  # Continuous action space
        output_size = env.action_space.shape[0]
        model = ContinuousPolicyNetwork(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            dropout=dropout
        )
    
    return model


def create_evolutionary_algorithm(
    algorithm_name: str,
    population_size: int,
    mutation_rate: float,
    mutation_strength: float,
    device: torch.device
) -> Any:
    """Create evolutionary algorithm.
    
    Args:
        algorithm_name: Name of the algorithm
        population_size: Size of population
        mutation_rate: Mutation rate
        mutation_strength: Mutation strength
        device: Device to run on
        
    Returns:
        Evolutionary algorithm instance
    """
    if algorithm_name == "simple_es":
        return SimpleEvolutionStrategy(
            population_size=population_size,
            mutation_rate=mutation_rate,
            mutation_strength=mutation_strength,
            device=device
        )
    elif algorithm_name == "cmaes":
        return CMAES(
            population_size=population_size,
            device=device
        )
    elif algorithm_name == "differential_evolution":
        return DifferentialEvolution(
            population_size=population_size,
            mutation_factor=mutation_strength,
            crossover_rate=mutation_rate,
            device=device
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")


def train_agent(args: argparse.Namespace) -> Dict[str, Any]:
    """Train the evolutionary RL agent.
    
    Args:
        args: Command line arguments
        
    Returns:
        Training results
    """
    # Setup
    set_seed(args.seed)
    device = get_device()
    logger = setup_logging(args.log_level, f"{args.log_dir}/training.log")
    
    logger.info("Starting Evolutionary RL Training")
    logger.info(f"Environment: {args.env}")
    logger.info(f"Algorithm: {args.algorithm}")
    logger.info(f"Device: {device}")
    logger.info(f"Seed: {args.seed}")
    
    # Create environment
    env = create_environment(args.env, args.seed)
    logger.info(f"Environment created: {env.observation_space.shape} -> {env.action_space}")
    
    # Create model
    model = create_model(env, tuple(args.hidden_sizes), args.dropout)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create evolutionary algorithm
    evolutionary_algorithm = create_evolutionary_algorithm(
        args.algorithm,
        args.population_size,
        args.mutation_rate,
        args.mutation_strength,
        device
    )
    
    # Create agent
    agent = EvolutionaryRLAgent(
        model=model,
        evolutionary_algorithm=evolutionary_algorithm,
        device=device,
        eval_episodes=args.eval_episodes,
        save_frequency=args.save_frequency,
        log_dir=args.log_dir
    )
    
    # Train agent
    results = agent.train(
        env=env,
        num_generations=args.generations,
        population_size=args.population_size,
        verbose=True,
        save_best=True
    )
    
    logger.info("Training completed successfully")
    return results


def evaluate_agent(args: argparse.Namespace) -> Dict[str, Any]:
    """Evaluate a trained agent.
    
    Args:
        args: Command line arguments
        
    Returns:
        Evaluation results
    """
    # Setup
    set_seed(args.seed)
    device = get_device()
    logger = setup_logging(args.log_level, f"{args.log_dir}/evaluation.log")
    
    logger.info("Starting Agent Evaluation")
    logger.info(f"Environment: {args.env}")
    logger.info(f"Model path: {args.model_path}")
    
    # Create environment
    env = create_environment(args.env, args.seed)
    
    # Create model
    model = create_model(env, tuple(args.hidden_sizes), args.dropout)
    
    # Load trained model
    if args.model_path:
        checkpoint = torch.load(args.model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        logger.info(f"Model loaded from {args.model_path}")
    
    # Create dummy evolutionary algorithm (not used for evaluation)
    evolutionary_algorithm = SimpleEvolutionStrategy(
        population_size=1,
        device=device
    )
    
    # Create agent
    agent = EvolutionaryRLAgent(
        model=model,
        evolutionary_algorithm=evolutionary_algorithm,
        device=device,
        eval_episodes=args.eval_episodes,
        log_dir=args.log_dir
    )
    
    # Evaluate agent
    evaluator = Evaluator(log_dir=args.log_dir)
    results = evaluator.evaluate_agent(
        agent=agent,
        env=env,
        num_episodes=100,
        deterministic=True,
        save_results=True
    )
    
    logger.info("Evaluation completed successfully")
    return results


def main():
    """Main function."""
    args = parse_args()
    
    # Create log directory
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = Path(args.log_dir) / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    
    if args.eval_only:
        results = evaluate_agent(args)
    else:
        results = train_agent(args)
        
        # Evaluate the trained agent
        logger = logging.getLogger(__name__)
        logger.info("Evaluating trained agent...")
        
        evaluator = Evaluator(log_dir=args.log_dir)
        eval_results = evaluator.evaluate_agent(
            agent=results.get('agent'),
            env=create_environment(args.env, args.seed),
            num_episodes=100,
            deterministic=True,
            save_results=True
        )
        
        # Plot training curves
        if 'training_history' in results:
            evaluator.plot_training_curves(
                results['training_history'],
                save_path=Path(args.log_dir) / "training_curves.png"
            )


if __name__ == "__main__":
    main()
