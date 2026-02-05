# Evolutionary Reinforcement Learning

Research-ready implementation of evolutionary algorithms for reinforcement learning, featuring multiple evolutionary strategies and comprehensive evaluation tools.

## ⚠️ Safety Disclaimer

**This software is for research and educational purposes only. It is NOT intended for production control of real-world systems, including but not limited to:**

- Autonomous vehicles
- Medical devices
- Financial trading systems
- Industrial control systems
- Safety-critical applications

Use at your own risk. The authors assume no responsibility for any damages or consequences arising from the use of this software.

## Overview

This project implements evolutionary reinforcement learning (ERL) using various evolutionary strategies to evolve neural network policies. The approach is inspired by biological evolution, where populations of policies evolve through mechanisms such as mutation, crossover, and selection to maximize rewards in reinforcement learning environments.

### Key Features

- **Multiple Evolutionary Algorithms**: Simple ES, CMA-ES, Differential Evolution
- **Modern Tech Stack**: PyTorch 2.x, Gymnasium, comprehensive logging
- **Comprehensive Evaluation**: Statistical analysis, benchmarking, visualization
- **Interactive Demo**: Streamlit-based visualization and experimentation
- **Production-Ready Structure**: Clean code, type hints, documentation, tests
- **Device Support**: CUDA, MPS (Apple Silicon), CPU fallback

## Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)
- MPS (optional, for Apple Silicon)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Evolutionary-Reinforcement-Learning.git
cd Evolutionary-Reinforcement-Learning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify installation:
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import gymnasium; print('Gymnasium installed successfully')"
```

## Quick Start

### Basic Training

Train an evolutionary RL agent on CartPole:

```bash
python -m src.train.train --env CartPole-v1 --generations 50 --population-size 20
```

### Using Configuration Files

```bash
python -m src.train.train --config configs/cartpole.yaml
```

### Interactive Demo

Launch the Streamlit demo:

```bash
streamlit run demo/app.py
```

## Usage

### Command Line Interface

The training script supports extensive configuration options:

```bash
python -m src.train.train \
    --env CartPole-v1 \
    --algorithm simple_es \
    --generations 100 \
    --population-size 30 \
    --mutation-rate 0.1 \
    --mutation-strength 0.1 \
    --hidden-sizes 64 64 \
    --log-dir logs/experiment_1
```

### Programmatic Usage

```python
import gymnasium as gym
from src.models.networks import PolicyNetwork
from src.algorithms.evolutionary import SimpleEvolutionStrategy
from src.policies.evolutionary_agent import EvolutionaryRLAgent

# Create environment
env = gym.make("CartPole-v1")

# Create model
model = PolicyNetwork(
    input_size=env.observation_space.shape[0],
    output_size=env.action_space.n,
    hidden_sizes=(64, 64)
)

# Create evolutionary algorithm
evolutionary_algorithm = SimpleEvolutionStrategy(
    population_size=20,
    mutation_rate=0.1,
    mutation_strength=0.1
)

# Create agent
agent = EvolutionaryRLAgent(
    model=model,
    evolutionary_algorithm=evolutionary_algorithm,
    eval_episodes=10
)

# Train agent
results = agent.train(env, num_generations=100)
```

## Algorithms

### Simple Evolution Strategy (ES)

A basic evolutionary strategy that uses mutation and selection:

- **Mutation**: Add Gaussian noise to network parameters
- **Selection**: Keep top-performing individuals
- **Reproduction**: Create offspring through mutation

### CMA-ES (Covariance Matrix Adaptation Evolution Strategy)

An advanced ES variant that adapts the mutation distribution:

- **Adaptive Covariance**: Learns the optimal mutation distribution
- **Step Size Control**: Automatically adjusts mutation strength
- **Rank-based Selection**: Uses fitness ranking for selection

### Differential Evolution (DE)

Uses difference vectors for mutation and crossover:

- **Mutation**: Create trial vectors using population differences
- **Crossover**: Combine parent and trial vectors
- **Selection**: Keep better individuals

## Environments

The framework supports various Gymnasium environments:

### Discrete Action Spaces
- **CartPole-v1**: Balance a pole on a cart
- **Acrobot-v1**: Swing up an inverted pendulum
- **MountainCar-v0**: Drive a car up a hill

### Continuous Action Spaces
- **MountainCarContinuous-v0**: Continuous version of MountainCar
- **Pendulum-v1**: Swing up a pendulum
- **BipedalWalker-v3**: Control a bipedal walker

## Evaluation

### Metrics

The evaluation system provides comprehensive metrics:

- **Performance**: Mean, median, standard deviation of returns
- **Statistical Analysis**: Confidence intervals, significance tests
- **Success Rate**: Percentage of successful episodes
- **Sample Efficiency**: Steps to reach performance threshold
- **Robustness**: Sensitivity analysis across seeds

### Benchmarking

Compare multiple agents:

```python
from src.eval.evaluator import Evaluator

evaluator = Evaluator()
results = evaluator.compare_agents(
    agents={"agent1": agent1, "agent2": agent2},
    env=env,
    num_episodes=100
)
```

## Project Structure

```
evolutionary-reinforcement-learning/
├── src/                          # Source code
│   ├── algorithms/               # Evolutionary algorithms
│   │   ├── __init__.py
│   │   └── evolutionary.py
│   ├── models/                  # Neural network models
│   │   ├── __init__.py
│   │   └── networks.py
│   ├── policies/                # RL agents
│   │   ├── __init__.py
│   │   └── evolutionary_agent.py
│   ├── eval/                    # Evaluation tools
│   │   ├── __init__.py
│   │   └── evaluator.py
│   ├── train/                   # Training scripts
│   │   ├── __init__.py
│   │   └── train.py
│   └── utils/                   # Utility functions
│       ├── __init__.py
│       └── utils.py
├── configs/                     # Configuration files
│   ├── default.yaml
│   └── cartpole.yaml
├── scripts/                     # Shell scripts
│   └── train.sh
├── demo/                        # Interactive demo
│   └── app.py
├── tests/                       # Unit tests
├── assets/                      # Generated assets
│   ├── logs/
│   ├── models/
│   ├── plots/
│   └── videos/
├── requirements.txt
├── .gitignore
└── README.md
```

## Configuration

Configuration files use YAML format and support hierarchical settings:

```yaml
# configs/custom.yaml
env:
  name: "CartPole-v1"
  seed: 42

training:
  generations: 100
  population_size: 30

algorithm:
  name: "cmaes"
  mutation_rate: 0.1

model:
  hidden_sizes: [64, 64]
  dropout: 0.1
```

## Logging and Monitoring

### Training Logs

- **Console Output**: Real-time training progress
- **File Logs**: Detailed logs saved to `logs/` directory
- **TensorBoard**: Optional integration for visualization
- **Checkpoints**: Model checkpoints saved periodically

### Evaluation Results

- **JSON Reports**: Structured evaluation results
- **Plots**: Training curves, comparison charts
- **Statistics**: Comprehensive statistical analysis

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_evolutionary.py
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run tests: `pytest`
5. Format code: `black src/ tests/`
6. Lint code: `ruff src/ tests/`
7. Commit changes: `git commit -m "Add feature"`
8. Push to branch: `git push origin feature-name`
9. Submit a pull request

## Performance Expectations

### CartPole-v1
- **Target Performance**: 195+ average reward
- **Typical Convergence**: 20-50 generations
- **Population Size**: 20-50 individuals

### MountainCar-v0
- **Target Performance**: -110 average reward
- **Typical Convergence**: 50-100 generations
- **Population Size**: 30-100 individuals

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce population size or use CPU
2. **Slow Training**: Enable GPU acceleration or reduce evaluation episodes
3. **Poor Performance**: Adjust mutation parameters or increase population size
4. **Import Errors**: Ensure all dependencies are installed

### Debug Mode

Enable debug logging:

```bash
python -m src.train.train --log-level DEBUG
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{evolutionary_rl,
  title={Evolutionary Reinforcement Learning},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Evolutionary-Reinforcement-Learning}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI Gym/Gymnasium for environment interfaces
- PyTorch team for the deep learning framework
- CMA-ES authors for the algorithm implementation
- The reinforcement learning community for inspiration and feedback
# Evolutionary-Reinforcement-Learning
