from abc import ABC, abstractmethod
import numpy as np


class Flow(ABC):
    """Base class for all flow types."""
    
    def __init__(self, domain_size: int):
        self.domain_size = domain_size
    
    @abstractmethod
    def step(self, z: np.ndarray) -> np.ndarray:
        """Single evolution step: z(t) -> z(t+1)"""
        pass
    
    def evolve(self, z_init: np.ndarray, steps: int) -> np.ndarray:
        """Evolve for multiple steps, returning trajectory."""
        if len(z_init) != self.domain_size:
            raise ValueError(f"Input size {len(z_init)} doesn't match domain size {self.domain_size}")
            
        trajectory = np.zeros((steps + 1, self.domain_size))
        trajectory[0] = z_init.copy()
        
        z = z_init.copy()
        for t in range(steps):
            z = self.step(z)
            trajectory[t + 1] = z
            
        return trajectory


class LinearFlow(Flow):
    """Base class for linear flows with additional methods."""
    
    @abstractmethod
    def get_matrix(self) -> np.ndarray:
        """Return the flow matrix P such that z(t+1) = P @ z(t)"""
        pass
    
    def steady_state(self, tol: float = 1e-10) -> np.ndarray:
        """Find steady state via eigenvalue decomposition."""
        P = self.get_matrix()
        eigenvals, eigenvecs = np.linalg.eig(P.T)
        idx = np.argmin(np.abs(eigenvals - 1.0))
        steady = np.real(eigenvecs[:, idx])
        return steady / steady.sum() if steady.sum() != 0 else steady
    
    def eigenvalues(self) -> np.ndarray:
        """Return eigenvalues of the flow matrix."""
        P = self.get_matrix()
        return np.linalg.eigvals(P)