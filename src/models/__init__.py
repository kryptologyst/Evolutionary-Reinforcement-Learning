"""Model initialization and factory functions."""

from .networks import PolicyNetwork, ContinuousPolicyNetwork, ValueNetwork
from typing import Union, Tuple, Optional
import torch


def create_policy_network(
    input_size: int,
    output_size: int,
    action_space_type: str = "discrete",
    hidden_sizes: Tuple[int, ...] = (64, 64),
    **kwargs
) -> Union[PolicyNetwork, ContinuousPolicyNetwork]:
    """Create appropriate policy network based on action space type.
    
    Args:
        input_size: Size of input state
        output_size: Number of actions/dimensions
        action_space_type: Type of action space ("discrete" or "continuous")
        hidden_sizes: Hidden layer sizes
        **kwargs: Additional arguments for network initialization
        
    Returns:
        Appropriate policy network
    """
    if action_space_type.lower() == "discrete":
        return PolicyNetwork(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            **kwargs
        )
    elif action_space_type.lower() == "continuous":
        return ContinuousPolicyNetwork(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown action space type: {action_space_type}")


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def clone_model(model: torch.nn.Module) -> torch.nn.Module:
    """Create a deep copy of a model.
    
    Args:
        model: Model to clone
        
    Returns:
        Cloned model
    """
    cloned_model = type(model)(**model.__dict__)
    cloned_model.load_state_dict(model.state_dict())
    return cloned_model
