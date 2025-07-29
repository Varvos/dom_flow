import numpy as np
from scipy.linalg import circulant
from ..base import LinearFlow


class StochasticMatrixFlow(LinearFlow):
    """General stochastic matrix flow: z(t+1) = P @ z(t)"""
    
    def __init__(self, matrix: np.ndarray):
        self.P = matrix
        self._validate_stochastic()
        super().__init__(self.P.shape[0])
    
    def _validate_stochastic(self):
        if not np.allclose(self.P.sum(axis=1), 1.0):
            raise ValueError("Matrix rows must sum to 1 (stochastic property)")
        if np.any(self.P < 0):
            raise ValueError("Matrix entries must be non-negative")
    
    def step(self, z: np.ndarray) -> np.ndarray:
        return self.P @ z
    
    def get_matrix(self) -> np.ndarray:
        return self.P


class SimpleAveraging(StochasticMatrixFlow):
    """Simple averaging flow using circulant matrix [1/2, 1/2, 0, ..., 0]"""
    
    def __init__(self, domain_size: int):
        first_row = np.zeros(domain_size)
        first_row[0] = 0.5
        first_row[1] = 0.5
        matrix = circulant(first_row)
        super().__init__(matrix)


class WeightedAveraging(StochasticMatrixFlow):
    """Weighted averaging flow using circulant matrix with custom weights"""
    
    def __init__(self, weights: np.ndarray):
        if not np.allclose(weights.sum(), 1.0):
            raise ValueError("Weights must sum to 1")
        matrix = circulant(weights)
        super().__init__(matrix)