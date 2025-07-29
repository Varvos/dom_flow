import torch
import torch.nn as nn
import numpy as np
from ..base import Flow


class NeuralFlow(Flow):
    """Single layer neural network flow: z(t+1) = σ(W @ z(t) + b)"""
    
    def __init__(self, domain_size: int, activation: str = "tanh", weights: torch.Tensor = None, bias: torch.Tensor = None):
        super().__init__(domain_size)
        
        self.activation_name = activation
        self.activation = self._get_activation(activation)
        
        if weights is not None:
            self.W = weights
        else:
            self.W = torch.randn(domain_size, domain_size) * 0.1
            
        if bias is not None:
            self.b = bias
        else:
            self.b = torch.zeros(domain_size)
    
    def _get_activation(self, name: str):
        activations = {
            "tanh": torch.tanh,
            "relu": torch.relu,
            "sigmoid": torch.sigmoid,
            "linear": lambda x: x
        }
        if name not in activations:
            raise ValueError(f"Unknown activation: {name}")
        return activations[name]
    
    def step(self, z: np.ndarray) -> np.ndarray:
        z_tensor = torch.from_numpy(z).float()
        z_new = self.activation(self.W @ z_tensor + self.b)
        return z_new.detach().numpy()
    
    def get_parameters(self):
        """Return parameters for optimization."""
        return {"W": self.W, "b": self.b}
    
    def set_parameters(self, params: dict):
        """Set parameters from optimization."""
        self.W = params["W"]
        self.b = params["b"]


class ComposedFlow(Flow):
    """Composed flow: f^n = f ∘ f ∘ ... ∘ f (n times)"""
    
    def __init__(self, base_flow: NeuralFlow, n_compositions: int):
        super().__init__(base_flow.domain_size)
        self.base_flow = base_flow
        self.n_compositions = n_compositions
    
    def step(self, z: np.ndarray) -> np.ndarray:
        """Apply base flow n times."""
        result = z.copy()
        for _ in range(self.n_compositions):
            result = self.base_flow.step(result)
        return result
    
    def get_parameters(self):
        """Return parameters of base flow."""
        return self.base_flow.get_parameters()
    
    def set_parameters(self, params: dict):
        """Set parameters of base flow."""
        self.base_flow.set_parameters(params)