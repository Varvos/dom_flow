# Domain Evolution Flow - Progress Report

## Project Overview

This project implements discrete flow processes for domain evolution, supporting both linear (stochastic matrix) and nonlinear (neural network) flows with interactive visualization.

## ✅ Completed Components

### 1. Core Flow Architecture
- **Base Classes** (`src/flows/base.py`):
  - `Flow`: Abstract base class with common interface
  - `LinearFlow`: Extends Flow with linear-specific methods (steady_state, eigenvalues)

### 2. Linear Flow Implementations (`src/flows/linear/`)
- **StochasticMatrixFlow**: General stochastic matrix flow z(t+1) = P @ z(t)
- **SimpleAveraging**: Circulant matrix with [0.5, 0.5, 0, ..., 0]
- **WeightedAveraging**: Circulant matrix with custom weight vectors
- All inherit from StochasticMatrixFlow for code reuse

### 3. Nonlinear Flow Implementations (`src/flows/nonlinear/`)
- **NeuralFlow**: Single layer NN with z(t+1) = σ(W @ z(t) + b)
- **ComposedFlow**: Compositions f^n = f ∘ f ∘ ... ∘ f
- Support for different activations (tanh, relu, sigmoid, linear)
- Parameter management for optimization