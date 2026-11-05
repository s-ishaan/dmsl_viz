"""
DMSL Visualization Utilities - Dataset Generators
Provides fast, reproducible 2D synthetic datasets tailored for statistical learning algorithms.
"""

from typing import Tuple, Optional
import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_circles


def get_dataset(
    dataset_type: str,
    n_samples: int = 100,
    noise: float = 0.1,
    random_state: int = 42,
    separation: float = 2.0,
    outlier_distance: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate 2D synthetic classification datasets.
    Labels are returned as {-1, +1} matching SVM formulation.
    """
    np.random.seed(random_state)
    n_samples = int(n_samples)

    if dataset_type == "Linearly Separable Blobs":
        # Two Gaussian blobs with controllable separation
        centers = [[-separation / 2, -separation / 2], [separation / 2, separation / 2]]
        cluster_std = max(0.15, noise * 2.0)
        X, y = make_blobs(
            n_samples=n_samples,
            centers=centers,
            cluster_std=cluster_std,
            random_state=random_state,
        )
        y = np.where(y == 0, -1, 1)

    elif dataset_type == "Overlapping Blobs":
        centers = [[-1.0, -1.0], [1.0, 1.0]]
        cluster_std = 0.8 + noise * 1.5
        X, y = make_blobs(
            n_samples=n_samples,
            centers=centers,
            cluster_std=cluster_std,
            random_state=random_state,
        )
        y = np.where(y == 0, -1, 1)

    elif dataset_type == "Moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
        # Center the data
        X[:, 0] = (X[:, 0] - 0.5) * 2
        X[:, 1] = (X[:, 1] - 0.25) * 2
        y = np.where(y == 0, -1, 1)

    elif dataset_type == "Concentric Circles":
        X, y = make_circles(
            n_samples=n_samples,
            factor=0.4,
            noise=noise,
            random_state=random_state,
        )
        X = X * 2.5
        y = np.where(y == 0, -1, 1)

    elif dataset_type == "XOR / Checkerboard":
        rng = np.random.RandomState(random_state)
        pts = rng.uniform(-2.5, 2.5, size=(n_samples, 2))
        # XOR pattern based on quadrant signs
        y = np.where((pts[:, 0] * pts[:, 1]) > 0, 1, -1)
        # Add slight noise to coordinates
        if noise > 0:
            pts += rng.randn(*pts.shape) * noise * 0.8
        X = pts

    elif dataset_type == "Outlier Challenge (Masking)":
        # Base separable blobs plus extreme outlier far on the "correct" side of class +1
        n_half = n_samples // 2
        rng = np.random.RandomState(random_state)
        
        # Class -1
        X_neg = rng.randn(n_half, 2) * 0.45 + np.array([-1.5, -1.0])
        y_neg = -np.ones(n_half, dtype=int)
        
        # Class +1
        X_pos = rng.randn(n_half - 1, 2) * 0.45 + np.array([1.5, 1.0])
        y_pos = np.ones(n_half - 1, dtype=int)
        
        # Outlier: far away at (+outlier_distance, -outlier_distance * 0.8) belonging to class +1
        outlier = np.array([[outlier_distance, -outlier_distance * 0.7]])
        y_outlier = np.array([1], dtype=int)
        
        X = np.vstack([X_neg, X_pos, outlier])
        y = np.concatenate([y_neg, y_pos, y_outlier])

    else:
        # Default fallback
        X, y = make_blobs(n_samples=n_samples, centers=2, random_state=random_state)
        y = np.where(y == 0, -1, 1)

    return X, y


def get_1d_regression_dataset(
    n_samples: int = 60,
    noise: float = 0.18,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate 1D non-linear regression data matching Slide 233.
    Returns: (x_dense, y_true, x_obs, y_obs) on interval [0, 1].
    """
    rng = np.random.RandomState(random_state)
    x_dense = np.linspace(0.0, 1.0, 300)
    # Smooth multi-frequency test function matching the slide
    y_true = np.sin(4.0 * x_dense) + 0.4 * np.cos(9.0 * x_dense) - 0.2 * x_dense

    x_obs = np.sort(rng.uniform(0.02, 0.98, n_samples))
    y_obs_true = np.sin(4.0 * x_obs) + 0.4 * np.cos(9.0 * x_obs) - 0.2 * x_obs
    y_obs = y_obs_true + rng.randn(n_samples) * noise

    return x_dense, y_true, x_obs, y_obs


def get_2d_regression_dataset(
    dataset_type: str = "Sinusoidal Surface",
    n_samples: int = 120,
    noise: float = 0.15,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate 2D regression datasets for regression trees and 3D surface visualizations.
    Returns: X (N, 2), y (N,)
    """
    rng = np.random.RandomState(random_state)
    x1 = rng.uniform(-2.5, 2.5, n_samples)
    x2 = rng.uniform(-2.5, 2.5, n_samples)
    X = np.column_stack([x1, x2])

    if dataset_type == "Sinusoidal Surface":
        y_clean = np.sin(1.4 * x1) + np.cos(1.4 * x2)
    elif dataset_type == "Non-linear Saddle":
        y_clean = 0.4 * (x1 ** 2 - x2 ** 2)
    elif dataset_type == "Step Pyramid (Blocks)":
        # Piecewise constant true surface matching Slide 262
        y_clean = (
            (x1 > 0).astype(float) * 1.5
            + (x2 > 0).astype(float) * 1.5
            - (x1 > 1.0).astype(float) * 0.8
        )
    else:
        y_clean = np.sin(x1) * np.cos(x2)

    y = y_clean + rng.randn(n_samples) * noise
    return X, y


