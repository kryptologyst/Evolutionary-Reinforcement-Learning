"""Streamlit demo for Evolutionary Reinforcement Learning."""

import streamlit as st
import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from pathlib import Path
import json
import time

from src.utils.utils import set_seed, get_device
from src.models.networks import PolicyNetwork, ContinuousPolicyNetwork
from src.algorithms.evolutionary import SimpleEvolutionStrategy, CMAES, DifferentialEvolution
from src.policies.evolutionary_agent import EvolutionaryRLAgent
from src.eval.evaluator import Evaluator


def create_environment(env_name: str, seed: int) -> gym.Env:
    """Create environment."""
    env = gym.make(env_name)
    env.reset(seed=seed)
    return env


def create_model(env: gym.Env, hidden_sizes: tuple, dropout: float) -> torch.nn.Module:
    """Create policy network model."""
    input_size = env.observation_space.shape[0]
    
    if isinstance(env.action_space, gym.spaces.Discrete):
        output_size = env.action_space.n
        model = PolicyNetwork(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            dropout=dropout
        )
    else:
        output_size = env.action_space.shape[0]
        model = ContinuousPolicyNetwork(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            dropout=dropout
        )
    
    return model


def create_evolutionary_algorithm(algorithm_name: str, population_size: int, **kwargs):
    """Create evolutionary algorithm."""
    device = get_device()
    
    if algorithm_name == "Simple Evolution Strategy":
        return SimpleEvolutionStrategy(
            population_size=population_size,
            mutation_rate=kwargs.get('mutation_rate', 0.1),
            mutation_strength=kwargs.get('mutation_strength', 0.1),
            device=device
        )
    elif algorithm_name == "CMA-ES":
        return CMAES(
            population_size=population_size,
            device=device
        )
    elif algorithm_name == "Differential Evolution":
        return DifferentialEvolution(
            population_size=population_size,
            mutation_factor=kwargs.get('mutation_strength', 0.8),
            crossover_rate=kwargs.get('mutation_rate', 0.9),
            device=device
        )


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Evolutionary Reinforcement Learning Demo",
        page_icon="🧬",
        layout="wide"
    )
    
    st.title("🧬 Evolutionary Reinforcement Learning Demo")
    st.markdown("""
    This demo showcases evolutionary algorithms for reinforcement learning. 
    Watch as populations of neural networks evolve to solve RL tasks!
    
    **⚠️ Disclaimer**: This is a research/educational tool. Not for production control of real systems.
    """)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Environment selection
    env_name = st.sidebar.selectbox(
        "Environment",
        ["CartPole-v1", "MountainCar-v0", "Acrobot-v1"],
        index=0
    )
    
    # Algorithm selection
    algorithm_name = st.sidebar.selectbox(
        "Evolutionary Algorithm",
        ["Simple Evolution Strategy", "CMA-ES", "Differential Evolution"],
        index=0
    )
    
    # Training parameters
    st.sidebar.subheader("Training Parameters")
    generations = st.sidebar.slider("Generations", 10, 200, 50)
    population_size = st.sidebar.slider("Population Size", 10, 100, 20)
    eval_episodes = st.sidebar.slider("Evaluation Episodes", 5, 20, 10)
    
    # Algorithm-specific parameters
    st.sidebar.subheader("Algorithm Parameters")
    mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.1)
    mutation_strength = st.sidebar.slider("Mutation Strength", 0.01, 0.5, 0.1)
    
    # Model parameters
    st.sidebar.subheader("Model Parameters")
    hidden_size_1 = st.sidebar.slider("Hidden Layer 1", 16, 128, 64)
    hidden_size_2 = st.sidebar.slider("Hidden Layer 2", 16, 128, 64)
    dropout = st.sidebar.slider("Dropout", 0.0, 0.5, 0.0)
    
    # Random seed
    seed = st.sidebar.number_input("Random Seed", value=42, min_value=0)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Training Progress")
        
        # Training button
        if st.button("🚀 Start Training", type="primary"):
            # Initialize training
            set_seed(seed)
            
            # Create environment
            env = create_environment(env_name, seed)
            
            # Create model
            model = create_model(env, (hidden_size_1, hidden_size_2), dropout)
            
            # Create evolutionary algorithm
            evolutionary_algorithm = create_evolutionary_algorithm(
                algorithm_name, population_size,
                mutation_rate=mutation_rate,
                mutation_strength=mutation_strength
            )
            
            # Create agent
            agent = EvolutionaryRLAgent(
                model=model,
                evolutionary_algorithm=evolutionary_algorithm,
                eval_episodes=eval_episodes,
                log_dir="demo_logs"
            )
            
            # Training progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Placeholder for plots
            plot_placeholder = st.empty()
            
            # Training loop
            training_data = {
                'generations': [],
                'best_fitness': [],
                'mean_fitness': [],
                'std_fitness': []
            }
            
            for generation in range(generations):
                status_text.text(f"Generation {generation + 1}/{generations}")
                
                # Train one generation
                generation_start = time.time()
                
                # Evaluate population
                fitness_scores = []
                for i, individual in enumerate(agent.evolutionary_algorithm.population):
                    fitness = agent.evaluate_model(individual, env)
                    fitness_scores.append(fitness)
                
                # Track best individual
                best_idx = np.argmax(fitness_scores)
                best_fitness = fitness_scores[best_idx]
                
                if best_fitness > agent.best_fitness:
                    agent.best_fitness = best_fitness
                    agent.best_model = agent.evolutionary_algorithm.population[best_idx]
                
                # Calculate statistics
                mean_fitness = np.mean(fitness_scores)
                std_fitness = np.std(fitness_scores)
                
                # Update training data
                training_data['generations'].append(generation + 1)
                training_data['best_fitness'].append(best_fitness)
                training_data['mean_fitness'].append(mean_fitness)
                training_data['std_fitness'].append(std_fitness)
                
                # Update progress
                progress = (generation + 1) / generations
                progress_bar.progress(progress)
                
                # Update plot every 5 generations
                if (generation + 1) % 5 == 0 or generation == generations - 1:
                    # Create training curve plot
                    fig = make_subplots(
                        rows=1, cols=1,
                        subplot_titles=["Training Progress"]
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=training_data['generations'],
                            y=training_data['best_fitness'],
                            mode='lines+markers',
                            name='Best Fitness',
                            line=dict(color='blue', width=2)
                        )
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=training_data['generations'],
                            y=training_data['mean_fitness'],
                            mode='lines+markers',
                            name='Mean Fitness',
                            line=dict(color='green', width=2)
                        )
                    )
                    
                    # Add confidence interval
                    upper_bound = np.array(training_data['mean_fitness']) + np.array(training_data['std_fitness'])
                    lower_bound = np.array(training_data['mean_fitness']) - np.array(training_data['std_fitness'])
                    
                    fig.add_trace(
                        go.Scatter(
                            x=training_data['generations'],
                            y=upper_bound,
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False
                        )
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=training_data['generations'],
                            y=lower_bound,
                            mode='lines',
                            line=dict(width=0),
                            fill='tonexty',
                            fillcolor='rgba(0,100,80,0.2)',
                            name='±1 Std',
                            showlegend=True
                        )
                    )
                    
                    fig.update_layout(
                        title="Training Progress",
                        xaxis_title="Generation",
                        yaxis_title="Fitness",
                        height=400
                    )
                    
                    plot_placeholder.plotly_chart(fig, use_container_width=True)
                
                # Evolve population (except for last generation)
                if generation < generations - 1:
                    agent.evolutionary_algorithm.population = agent.evolutionary_algorithm.evolve(
                        agent.evolutionary_algorithm.population, fitness_scores
                    )
            
            # Training completed
            status_text.text("✅ Training completed!")
            
            # Store results in session state
            st.session_state.training_results = {
                'agent': agent,
                'training_data': training_data,
                'env': env,
                'config': {
                    'env_name': env_name,
                    'algorithm': algorithm_name,
                    'generations': generations,
                    'population_size': population_size
                }
            }
    
    with col2:
        st.header("Environment Info")
        
        if 'training_results' in st.session_state:
            env = st.session_state.training_results['env']
            
            st.subheader("Environment Details")
            st.write(f"**Name**: {env.spec.id}")
            st.write(f"**Observation Space**: {env.observation_space}")
            st.write(f"**Action Space**: {env.action_space}")
            
            st.subheader("Training Results")
            training_data = st.session_state.training_results['training_data']
            best_fitness = max(training_data['best_fitness'])
            final_mean = training_data['mean_fitness'][-1]
            
            st.metric("Best Fitness", f"{best_fitness:.2f}")
            st.metric("Final Mean Fitness", f"{final_mean:.2f}")
            st.metric("Generations", len(training_data['generations']))
        
        else:
            st.info("Start training to see environment details and results.")
    
    # Evaluation section
    if 'training_results' in st.session_state:
        st.header("Agent Evaluation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎯 Evaluate Agent"):
                agent = st.session_state.training_results['agent']
                env = st.session_state.training_results['env']
                
                # Evaluation progress
                eval_progress = st.progress(0)
                eval_status = st.empty()
                
                # Run evaluation
                episode_rewards = []
                episode_lengths = []
                
                for episode in range(50):
                    eval_status.text(f"Evaluating episode {episode + 1}/50")
                    
                    state, _ = env.reset()
                    total_reward = 0
                    steps = 0
                    
                    while steps < 1000:
                        action = agent.get_action(state, deterministic=True)
                        next_state, reward, terminated, truncated, _ = env.step(action)
                        
                        total_reward += reward
                        steps += 1
                        state = next_state
                        
                        if terminated or truncated:
                            break
                    
                    episode_rewards.append(total_reward)
                    episode_lengths.append(steps)
                    
                    eval_progress.progress((episode + 1) / 50)
                
                eval_status.text("✅ Evaluation completed!")
                
                # Display results
                st.subheader("Evaluation Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean Reward", f"{np.mean(episode_rewards):.2f}")
                with col2:
                    st.metric("Std Reward", f"{np.std(episode_rewards):.2f}")
                with col3:
                    st.metric("Success Rate", f"{np.mean([r > 0 for r in episode_rewards]):.2%}")
                
                # Reward distribution plot
                fig = px.histogram(
                    x=episode_rewards,
                    nbins=20,
                    title="Episode Reward Distribution",
                    labels={'x': 'Episode Reward', 'y': 'Frequency'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if st.button("🎮 Watch Agent Play"):
                agent = st.session_state.training_results['agent']
                env = st.session_state.training_results['env']
                
                # Create a simple visualization
                st.subheader("Agent Performance")
                
                # Run one episode
                state, _ = env.reset()
                total_reward = 0
                steps = 0
                states = [state.copy()]
                
                while steps < 1000:
                    action = agent.get_action(state, deterministic=True)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    
                    total_reward += reward
                    steps += 1
                    state = next_state
                    states.append(state.copy())
                    
                    if terminated or truncated:
                        break
                
                st.write(f"**Episode Reward**: {total_reward:.2f}")
                st.write(f"**Episode Length**: {steps}")
                
                # Simple state trajectory plot
                if len(states) > 1:
                    states_array = np.array(states)
                    
                    fig = go.Figure()
                    for i in range(states_array.shape[1]):
                        fig.add_trace(
                            go.Scatter(
                                x=list(range(len(states))),
                                y=states_array[:, i],
                                mode='lines',
                                name=f'State {i+1}'
                            )
                        )
                    
                    fig.update_layout(
                        title="State Trajectory",
                        xaxis_title="Step",
                        yaxis_title="State Value",
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### About Evolutionary Reinforcement Learning
    
    Evolutionary RL uses principles from biological evolution to train neural networks:
    - **Population**: A group of neural network policies
    - **Fitness**: Performance on the RL task (cumulative reward)
    - **Selection**: Better-performing policies are more likely to reproduce
    - **Mutation**: Random changes to network parameters
    - **Evolution**: Iterative improvement over generations
    
    This approach can be particularly effective for:
    - Complex environments with sparse rewards
    - Continuous control tasks
    - Multi-objective optimization
    - Exploration in high-dimensional spaces
    """)


if __name__ == "__main__":
    main()
