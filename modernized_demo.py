#!/usr/bin/env python3
"""
Modernized Evolutionary Reinforcement Learning Example

This script demonstrates the modernized evolutionary RL implementation
compared to the original 0615.py script.

Key improvements:
- Modern tech stack (Gymnasium, PyTorch 2.x)
- Type hints and comprehensive documentation
- Multiple evolutionary algorithms
- Comprehensive evaluation and logging
- Device support (CUDA/MPS/CPU)
- Reproducible results with proper seeding
"""

import argparse
import logging
import gymnasium as gym
import numpy as np
import torch
from pathlib import Path

from src.utils.utils import set_seed, get_device, setup_logging
from src.models.networks import PolicyNetwork
from src.algorithms.evolutionary import SimpleEvolutionStrategy, CMAES
from src.policies.evolutionary_agent import EvolutionaryRLAgent
from src.eval.evaluator import Evaluator


def main():
    """Main function demonstrating modernized evolutionary RL."""
    parser = argparse.ArgumentParser(description="Modernized Evolutionary RL Demo")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Environment name")
    parser.add_argument("--algorithm", type=str, default="simple_es", 
                       choices=["simple_es", "cmaes"], help="Evolutionary algorithm")
    parser.add_argument("--generations", type=int, default=50, help="Number of generations")
    parser.add_argument("--population-size", type=int, default=20, help="Population size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-dir", type=str, default="demo_logs", help="Log directory")
    
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    device = get_device()
    logger = setup_logging("INFO", f"{args.log_dir}/demo.log")
    
    logger.info("Starting Modernized Evolutionary RL Demo")
    logger.info(f"Environment: {args.env}")
    logger.info(f"Algorithm: {args.algorithm}")
    logger.info(f"Device: {device}")
    logger.info(f"Seed: {args.seed}")
    
    # Create environment
    env = gym.make(args.env)
    env.reset(seed=args.seed)
    logger.info(f"Environment created: {env.observation_space.shape} -> {env.action_space}")
    
    # Create model
    model = PolicyNetwork(
        input_size=env.observation_space.shape[0],
        output_size=env.action_space.n,
        hidden_sizes=(64, 64),
        dropout=0.0
    )
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create evolutionary algorithm
    if args.algorithm == "simple_es":
        evolutionary_algorithm = SimpleEvolutionStrategy(
            population_size=args.population_size,
            mutation_rate=0.1,
            mutation_strength=0.1,
            device=device
        )
    elif args.algorithm == "cmaes":
        evolutionary_algorithm = CMAES(
            population_size=args.population_size,
            device=device
        )
    
    # Create agent
    agent = EvolutionaryRLAgent(
        model=model,
        evolutionary_algorithm=evolutionary_algorithm,
        device=device,
        eval_episodes=10,
        save_frequency=10,
        log_dir=args.log_dir
    )
    
    # Train agent
    logger.info("Starting training...")
    results = agent.train(
        env=env,
        num_generations=args.generations,
        population_size=args.population_size,
        verbose=True,
        save_best=True
    )
    
    logger.info("Training completed successfully!")
    logger.info(f"Best fitness achieved: {results['best_fitness']:.4f}")
    logger.info(f"Training time: {results['training_time']:.2f} seconds")
    
    # Evaluate the trained agent
    logger.info("Evaluating trained agent...")
    evaluator = Evaluator(log_dir=args.log_dir)
    eval_results = evaluator.evaluate_agent(
        agent=agent,
        env=env,
        num_episodes=100,
        deterministic=True,
        save_results=True
    )
    
    logger.info("Evaluation Results:")
    logger.info(f"  Mean Reward: {eval_results['mean_reward']:.4f} ± {eval_results['std_reward']:.4f}")
    logger.info(f"  Success Rate: {eval_results['success_rate']:.4f}")
    logger.info(f"  95% CI: [{eval_results['ci_lower']:.4f}, {eval_results['ci_upper']:.4f}]")
    
    # Plot training curves
    if 'training_history' in results:
        evaluator.plot_training_curves(
            results['training_history'],
            save_path=Path(args.log_dir) / "training_curves.png"
        )
    
    logger.info("Demo completed successfully!")
    logger.info(f"Results saved to: {args.log_dir}")


if __name__ == "__main__":
    main()
