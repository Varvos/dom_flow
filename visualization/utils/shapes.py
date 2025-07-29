import numpy as np


def circle(n_points: int, radius: float = 1.0, scale_x: float = 1.0, scale_y: float = 1.0) -> np.ndarray:
    """Generate points on a circle/ellipse."""
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = radius * np.cos(t) * scale_x
    y = radius * np.sin(t) * scale_y
    return np.column_stack([x, y])


def flower(n_points: int, petals: int = 5, radius: float = 1.0, scale_x: float = 1.0, scale_y: float = 1.0) -> np.ndarray:
    """Generate flower shape points using rose curve r = radius * cos(petals * t)."""
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    r = radius * np.abs(np.cos(petals * t))
    x = r * np.cos(t) * scale_x
    y = r * np.sin(t) * scale_y
    return np.column_stack([x, y])


def star(n_points: int, spikes: int = 5, radius: float = 1.0, scale_x: float = 1.0, scale_y: float = 1.0) -> np.ndarray:
    """Generate star shape points."""
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    # Alternate between outer and inner radius
    inner_radius = radius * 0.4
    r = np.where(np.sin(spikes * t) > 0, radius, inner_radius)
    x = r * np.cos(t) * scale_x
    y = r * np.sin(t) * scale_y
    return np.column_stack([x, y])


def rectangle(n_points: int, width: float = 2.0, height: float = 1.0, scale_x: float = 1.0, scale_y: float = 1.0) -> np.ndarray:
    """Generate points on rectangle perimeter."""
    # Distribute points along perimeter
    perimeter = 2 * (width + height)
    points_per_side = n_points // 4
    remainder = n_points % 4
    
    points = []
    
    # Bottom edge
    n_bottom = points_per_side + (1 if remainder > 0 else 0)
    x_bottom = np.linspace(-width/2, width/2, n_bottom, endpoint=False)
    y_bottom = np.full(n_bottom, -height/2)
    points.extend(zip(x_bottom, y_bottom))
    
    # Right edge
    n_right = points_per_side + (1 if remainder > 1 else 0)
    x_right = np.full(n_right, width/2)
    y_right = np.linspace(-height/2, height/2, n_right, endpoint=False)
    points.extend(zip(x_right, y_right))
    
    # Top edge
    n_top = points_per_side + (1 if remainder > 2 else 0)
    x_top = np.linspace(width/2, -width/2, n_top, endpoint=False)
    y_top = np.full(n_top, height/2)
    points.extend(zip(x_top, y_top))
    
    # Left edge
    n_left = n_points - len(points)
    x_left = np.full(n_left, -width/2)
    y_left = np.linspace(height/2, -height/2, n_left, endpoint=False)
    points.extend(zip(x_left, y_left))
    
    points_array = np.array(points[:n_points])
    points_array[:, 0] *= scale_x
    points_array[:, 1] *= scale_y
    return points_array


def spiral(n_points: int, turns: float = 3.0, radius: float = 1.0, scale_x: float = 1.0, scale_y: float = 1.0) -> np.ndarray:
    """Generate spiral points."""
    t = np.linspace(0, 2 * np.pi * turns, n_points)
    r = radius * t / (2 * np.pi * turns)
    x = r * np.cos(t) * scale_x
    y = r * np.sin(t) * scale_y
    return np.column_stack([x, y])