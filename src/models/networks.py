"""Neural network models for evolutionary reinforcement learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union
import numpy as np


class PolicyNetwork(nn.Module):
    """Policy network for discrete action spaces using softmax output.
    
    This network outputs action probabilities for discrete action spaces,
    commonly used in evolutionary strategies for RL.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        activation: str = "relu",
        dropout: float = 0.0
    ):
        """Initialize policy network.
        
        Args:
            input_size: Size of input state
            output_size: Number of actions
            hidden_sizes: Hidden layer sizes
            activation: Activation function name
            dropout: Dropout probability
        """
        super(PolicyNetwork, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout
        
        # Build network layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
            
        layers.append(nn.Linear(prev_size, output_size))
        self.layers = nn.ModuleList(layers)
        
        # Activation function
        self.activation = getattr(F, activation)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input state tensor
            
        Returns:
            Action probabilities
        """
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)
            if self.dropout > 0 and self.training:
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer with softmax
        x = self.layers[-1](x)
        return F.softmax(x, dim=-1)
    
    def get_action(self, state: Union[np.ndarray, torch.Tensor], deterministic: bool = False) -> int:
        """Sample action from policy.
        
        Args:
            state: Current state
            deterministic: If True, return argmax action
            
        Returns:
            Selected action
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        
        with torch.no_grad():
            action_probs = self.forward(state)
            
            if deterministic:
                return action_probs.argmax().item()
            else:
                return torch.multinomial(action_probs, 1).item()


class ContinuousPolicyNetwork(nn.Module):
    """Policy network for continuous action spaces using Gaussian output.
    
    This network outputs mean and log_std for continuous actions,
    enabling sampling from a Gaussian distribution.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        activation: str = "relu",
        dropout: float = 0.0,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0
    ):
        """Initialize continuous policy network.
        
        Args:
            input_size: Size of input state
            output_size: Number of action dimensions
            hidden_sizes: Hidden layer sizes
            activation: Activation function name
            dropout: Dropout probability
            log_std_min: Minimum log standard deviation
            log_std_max: Maximum log standard deviation
        """
        super(ContinuousPolicyNetwork, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Build network layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
            
        self.layers = nn.ModuleList(layers)
        
        # Separate heads for mean and log_std
        self.mean_head = nn.Linear(prev_size, output_size)
        self.log_std_head = nn.Linear(prev_size, output_size)
        
        # Activation function
        self.activation = getattr(F, activation)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.
        
        Args:
            x: Input state tensor
            
        Returns:
            Tuple of (mean, log_std)
        """
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
            if self.dropout > 0 and self.training:
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def get_action(
        self, 
        state: Union[np.ndarray, torch.Tensor], 
        deterministic: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Sample action from policy.
        
        Args:
            state: Current state
            deterministic: If True, return mean action
            
        Returns:
            If deterministic: action tensor
            If not deterministic: tuple of (action, log_prob)
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        if deterministic:
            return mean
        else:
            normal = torch.distributions.Normal(mean, std)
            action = normal.sample()
            log_prob = normal.log_prob(action).sum(dim=-1, keepdim=True)
            return action, log_prob


class ValueNetwork(nn.Module):
    """Value network for estimating state values.
    
    Used in actor-critic methods or for value-based evolutionary strategies.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        activation: str = "relu",
        dropout: float = 0.0
    ):
        """Initialize value network.
        
        Args:
            input_size: Size of input state
            hidden_sizes: Hidden layer sizes
            activation: Activation function name
            dropout: Dropout probability
        """
        super(ValueNetwork, self).__init__()
        
        self.input_size = input_size
        self.dropout = dropout
        
        # Build network layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
            
        layers.append(nn.Linear(prev_size, 1))
        self.layers = nn.ModuleList(layers)
        
        # Activation function
        self.activation = getattr(F, activation)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input state tensor
            
        Returns:
            State value estimate
        """
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)
            if self.dropout > 0 and self.training:
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer (no activation for value)
        x = self.layers[-1](x)
        return x.squeeze(-1)
