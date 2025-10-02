import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import circulant
from ..base import LinearFlow


class StochasticMatrixFlow(LinearFlow):
    """General stochastic matrix flow: z(t+1) = P @ z(t)"""
    
    def __init__(self, matrix: np.ndarray, d: int = 2, shared_dims: bool = True, trainable: bool = False):
        self.trainable = trainable
        
        if shared_dims:
            # Matrix should be (n_points, n_points)
            expected_shape = (matrix.shape[0], matrix.shape[0])
            if matrix.shape != expected_shape:
                raise ValueError(f"For shared_dims=True, matrix should be ({matrix.shape[0]}, {matrix.shape[0]}), got {matrix.shape}")
        else:
            # Matrix should be (d, n_points, n_points)
            n_points = matrix.shape[1] if len(matrix.shape) == 3 else matrix.shape[0]
            expected_shape = (d, n_points, n_points)
            if len(matrix.shape) == 2:
                # Convert (n_points, n_points) to (d, n_points, n_points) by replicating
                matrix = np.tile(matrix[np.newaxis, :, :], (d, 1, 1))
            elif matrix.shape != expected_shape:
                raise ValueError(f"For shared_dims=False, matrix should be {expected_shape}, got {matrix.shape}")
        
        if trainable:
            # Create trainable PyTorch parameter
            self._matrix_param = nn.Parameter(torch.from_numpy(matrix.copy()).float())
            self.P = None  # Will be computed from parameter
        else:
            self.P = matrix
            self._matrix_param = None
            
        self._validate_stochastic(matrix, shared_dims)
        n_points = matrix.shape[0] if shared_dims else matrix.shape[1]
        super().__init__(n_points, d, shared_dims)
    
    def _validate_stochastic(self, matrix: np.ndarray, shared_dims: bool):
        if shared_dims:
            # Matrix is (n_points, n_points)
            if not np.allclose(matrix.sum(axis=1), 1.0):
                raise ValueError("Matrix rows must sum to 1 (stochastic property)")
            if np.any(matrix < 0):
                raise ValueError("Matrix entries must be non-negative")
        else:
            # Matrix is (d, n_points, n_points)
            for dim in range(matrix.shape[0]):
                if not np.allclose(matrix[dim].sum(axis=1), 1.0):
                    raise ValueError(f"Matrix[{dim}] rows must sum to 1 (stochastic property)")
                if np.any(matrix[dim] < 0):
                    raise ValueError(f"Matrix[{dim}] entries must be non-negative")
    
    def get_matrix(self) -> np.ndarray:
        if self.trainable:
            # Apply softmax to ensure stochastic property
            if self.shared_dims:
                # Apply softmax along last dimension for (n_points, n_points)
                matrix = torch.softmax(self._matrix_param, dim=1)
            else:
                # Apply softmax along last dimension for each (n_points, n_points) slice
                matrix = torch.softmax(self._matrix_param, dim=2)
            return matrix.detach().numpy()
        else:
            return self.P
    
    def step(self, z: np.ndarray) -> np.ndarray:
        P = self.get_matrix()
        
        if self.shared_dims:
            # P is (n_points, n_points), broadcast over dimensions
            return P @ z
        else:
            # P is (d, n_points, n_points), apply each slice to corresponding dimension
            result = np.zeros_like(z)
            for dim in range(self.d):
                result[:, dim] = P[dim] @ z[:, dim]
            return result
    
    def parameters(self):
        """Return trainable parameters."""
        if self.trainable:
            return iter([self._matrix_param])
        return iter([])
    
    def to(self, device):
        """Move parameters to device."""
        if self.trainable:
            self._matrix_param = self._matrix_param.to(device)
        return self


class SimpleAveraging(StochasticMatrixFlow):
    """Simple averaging flow using circulant matrix [1/2, 1/2, 0, ..., 0]"""
    
    def __init__(self, n_points: int, d: int = 2, shared_dims: bool = True, trainable: bool = False):
        first_row = np.zeros(n_points)
        first_row[0] = 0.5
        first_row[1] = 0.5
        matrix = circulant(first_row)
        super().__init__(matrix, d, shared_dims, trainable)


class WeightedAveraging(StochasticMatrixFlow):
    """Weighted averaging flow using circulant matrix with custom weights"""
    
    def __init__(self, weights: np.ndarray, d: int = 2, shared_dims: bool = True, trainable: bool = False):
        if not np.allclose(weights.sum(), 1.0):
            raise ValueError("Weights must sum to 1")
        matrix = circulant(weights)
        super().__init__(matrix, d, shared_dims, trainable)