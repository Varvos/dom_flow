from abc import ABC, abstractmethod
import numpy as np


class Flow(ABC):
    """Base class for all flow types."""
    
    def __init__(self, n_points: int, d: int = 2):
        self.n_points = n_points
        self.d = d
    
    @abstractmethod
    def evolve(self, z_init: np.ndarray, steps: int) -> np.ndarray:
        """Evolve for multiple steps, returning trajectory.
        
        Args:
            z_init: Initial point cloud of shape (n_points, d)
            steps: Number of evolution steps
            
        Returns:
            Trajectory of shape (steps+1, n_points, d)
        """
        pass

    def step(self, z: np.ndarray) -> np.ndarray:
        """Single step evolution: z(t+1) = F(z(t))
        
        Args:
            z: Point cloud of shape (n_points, d)
            
        Returns:
            Evolved point cloud of shape (n_points, d)
        """
        return self.evolve(z, steps=1)[1]
    
    def is_trainable(self) -> bool:
        """Check if flow has trainable parameters."""
        return hasattr(self, 'parameters') and callable(getattr(self, 'parameters'))
    
    def train_mode(self, mode: bool = True):
        """Set training mode for flows with PyTorch components."""
        if hasattr(self, 'network') and hasattr(self.network, 'train'):
            self.network.train(mode)
        return self
    
    def eval_mode(self):
        """Set evaluation mode for flows with PyTorch components."""
        return self.train_mode(False)

class LinearFlow(Flow):
    """Base class for linear flows with additional methods."""
    
    def __init__(self, n_points: int, d: int = 2, shared_dims: bool = True):
        super().__init__(n_points, d)
        self.shared_dims = shared_dims
    
    @abstractmethod
    def get_matrix(self) -> np.ndarray:
        """Return the flow matrix P.
        
        Returns:
            If shared_dims=True: matrix of shape (n_points, n_points)
            If shared_dims=False: matrix of shape (d, n_points, n_points)
        """
        pass
    
    def steady_state(self, tol: float = 1e-10) -> np.ndarray:
        """Find steady state via eigenvalue decomposition.
        
        Returns:
            Steady state of shape (n_points, d)
        """
        P = self.get_matrix()
        
        if self.shared_dims:
            # P is (n_points, n_points), same steady state for all dimensions
            eigenvals, eigenvecs = np.linalg.eig(P.T)
            idx = np.argmin(np.abs(eigenvals - 1.0))
            steady_1d = np.real(eigenvecs[:, idx])
            steady_1d = steady_1d / steady_1d.sum() if steady_1d.sum() != 0 else steady_1d
            # Broadcast to all dimensions
            steady = np.tile(steady_1d[:, np.newaxis], (1, self.d))
        else:
            # P is (d, n_points, n_points), separate steady state for each dimension
            steady = np.zeros((self.n_points, self.d))
            for dim in range(self.d):
                eigenvals, eigenvecs = np.linalg.eig(P[dim].T)
                idx = np.argmin(np.abs(eigenvals - 1.0))
                steady_dim = np.real(eigenvecs[:, idx])
                steady[:, dim] = steady_dim / steady_dim.sum() if steady_dim.sum() != 0 else steady_dim
        
        return steady

    def evolve(self, z_init: np.ndarray, steps: int) -> np.ndarray:
        trajectory = np.zeros((steps + 1, self.n_points, self.d))
        trajectory[0] = z_init
        P = self.get_matrix()
        
        for t in range(1, steps + 1):
            if self.shared_dims:
                # P is (n_points, n_points), broadcast over dimensions
                trajectory[t] = P @ trajectory[t - 1]
            else:
                # P is (d, n_points, n_points), apply each slice to corresponding dimension
                for dim in range(self.d):
                    trajectory[t, :, dim] = P[dim] @ trajectory[t - 1, :, dim]
        
        return trajectory
