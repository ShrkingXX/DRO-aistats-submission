import numpy as np

# # Define test functions (no changes needed)
# def ackley_function(x):
#     a, b, c = 20, 0.2, 2 * np.pi
#     # Ensure input is numpy array for calculations
#     x = np.array(x)
#     sum1 = -a * np.exp(-b * np.sqrt(np.mean(x**2)))
#     sum2 = -np.exp(np.mean(np.cos(c * x)))
#     return -(sum1 + sum2 + a + np.exp(1))  # Negative for maximization

# def rosenbrock_function(x):
#     """Rosenbrock function (minimization)"""
#     # Ensure input is numpy array for calculations
#     x = np.array(x)
#     result = 0
#     for i in range(len(x) - 1):
#         result += 100 * (x[i + 1] - x[i]**2)**2 + (x[i] - 1)**2
#     # Convert to maximization
#     return -result

# def levy_function(x):
#     """Levy function (minimization)"""
#     # Ensure input is numpy array for calculations
#     x = np.array(x)
#     w = 1 + (x - 1) / 4
#     term1 = np.sin(np.pi * w[0])**2
#     term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
#     term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
#     # Convert to maximization
#     return -(term1 + term2 + term3)


import torch
from botorch.test_functions.synthetic import SyntheticTestFunction
from typing import Optional

# Define constants used in Ackley
ACKLEY_A = 20.0
ACKLEY_B = 0.2
ACKLEY_C = 2 * torch.pi


class Ackley(SyntheticTestFunction):
    r"""Ackley synthetic test function.

    d-dimensional function usually evaluated on the hypercube
    `[-32.768, 32.768]^d` or `[-5, 5]^d`. The function has many local
    minima and one global minimum at `f(x*) = 0` located at `x* = (0, ..., 0)`.

    f(x) = -a * exp(-b * sqrt(1/d * sum_{i=1}^d x_i^2)) -
           exp(1/d * sum_{i=1}^d cos(c * x_i)) + a + exp(1)
    """

    dim = 2 # Default dimension
    _optimizers = [(0.0, 0.0)] # Global minimum at the origin
    optimal_value = 0.0 # Value at the global minimum

    def __init__(
        self,
        dim: int = 2,
        noise_std: Optional[float] = None,
        negate: bool = True, # True matches original code's maximization goal
        bounds: Optional[list[tuple[float, float]]] = None,
        shift: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Initialize Ackley function.

        Args:
            dim: Dimension of the function.
            noise_std: Standard deviation of observation noise. Default: None.
            negate: If True, negate the function for maximization. Default: True.
            bounds: Custom bounds for the function domain. If None, uses [-5, 5]^d.
        """
        self.dim = dim
        if bounds is None:
            # Using bounds from the original code context
            self._bounds = [(-5.0, 5.0)] * self.dim
        else:
            self._bounds = bounds
        # Ensure optimizer matches dimension
        self._optimizers = [(0.0,) * self.dim]
        self.shift = shift

        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the Ackley function (standard minimization form).

        Args:
            X: A (batch_shape) x d tensor of inputs.

        Returns:
            A (batch_shape) tensor of function values.
        """
        if X.ndim < 1 or X.shape[-1] != self.dim:
             raise ValueError(f"Input tensor X must have last dimension equal to {self.dim}")
        if self.shift is not None:
            # Shift the input tensor if a shift is provided
            X = X - self.shift

        # Ensure calculations happen on the correct device and dtype as X
        a = torch.tensor(ACKLEY_A, device=X.device, dtype=X.dtype)
        b = torch.tensor(ACKLEY_B, device=X.device, dtype=X.dtype)
        c = torch.tensor(ACKLEY_C, device=X.device, dtype=X.dtype)
        exp_1 = torch.exp(torch.tensor(1.0, device=X.device, dtype=X.dtype))
        inv_dim = torch.tensor(1.0 / self.dim, device=X.device, dtype=X.dtype)

        # Calculate terms - operate on the last dimension (dim=-1)
        term1 = -a * torch.exp(-b * torch.sqrt(torch.mean(X**2, dim=-1)))
        term2 = -torch.exp(torch.mean(torch.cos(c * X), dim=-1))

        # Standard minimization form
        result = term1 + term2 + a + exp_1
        return result


class Rosenbrock(SyntheticTestFunction):
    r"""Rosenbrock synthetic test function.

    d-dimensional function usually evaluated on the hypercube `[-5, 10]^d`.
    Its global minimum `f(x*) = 0` is located at `x* = (1, ..., 1)`. For d=2,
    often evaluated on `[-2, 2]^2` or similar.

    f(x) = sum_{i=1}^{d-1} [100 * (x_{i+1} - x_i^2)^2 + (x_i - 1)^2]
    """

    dim = 2 # Default dimension
    _optimizers = [(1.0, 1.0)] # Global minimum at (1, 1, ...)
    optimal_value = 0.0 # Value at the global minimum

    def __init__(
        self,
        dim: int = 2,
        noise_std: Optional[float] = None,
        negate: bool = True, # True matches original code's maximization goal
        bounds: Optional[list[tuple[float, float]]] = None,
        shift: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Initialize Rosenbrock function.

        Args:
            dim: Dimension of the function (must be >= 2).
            noise_std: Standard deviation of observation noise. Default: None.
            negate: If True, negate the function for maximization. Default: True.
            bounds: Custom bounds for the function domain. If None, uses [-2, 2]^d.
        """
        if dim < 2:
            raise ValueError("Rosenbrock function dimension must be >= 2")
        self.dim = dim
        if bounds is None:
            # Using bounds from the original code context
            self._bounds = [(-2.0, 2.0)] * self.dim
        else:
            self._bounds = bounds
        # Ensure optimizer matches dimension
        self._optimizers = [(1.0,) * self.dim]
        self.shift = shift

        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the Rosenbrock function (standard minimization form).

        Args:
            X: A (batch_shape) x d tensor of inputs.

        Returns:
            A (batch_shape) tensor of function values.
        """
        if X.ndim < 1 or X.shape[-1] != self.dim:
             raise ValueError(f"Input tensor X must have last dimension equal to {self.dim}")
        if self.shift is not None:
            # Shift the input tensor if a shift is provided
            X = X - self.shift

        # Standard calculation using tensor slicing
        # Calculate sum_{i=0}^{d-2} [100*(x_{i+1} - x_i^2)^2 + (x_i - 1)^2]
        # X[..., :-1] selects all but the last element along the dim dimension
        # X[..., 1:] selects all but the first element along the dim dimension
        term1 = 100.0 * (X[..., 1:] - X[..., :-1] ** 2) ** 2
        term2 = (X[..., :-1] - 1.0) ** 2
        result = torch.sum(term1 + term2, dim=-1) # Sum across the feature dimension

        return result


class Levy(SyntheticTestFunction):
    r"""Levy synthetic test function.

    d-dimensional function usually evaluated on the hypercube `[-10, 10]^d`.
    Its global minimum `f(x*) = 0` is located at `x* = (1, ..., 1)`.

    f(x) = sin^2(pi * w_1) +
           sum_{i=1}^{d-1} (w_i - 1)^2 * [1 + 10 * sin^2(pi * w_i + 1)] +
           (w_d - 1)^2 * [1 + sin^2(2 * pi * w_d)]
    where w_i = 1 + (x_i - 1) / 4
    """

    dim = 2 # Default dimension
    _optimizers = [(1.0, 1.0)] # Global minimum at (1, 1, ...)
    optimal_value = 0.0 # Value at the global minimum

    def __init__(
        self,
        dim: int = 2,
        noise_std: Optional[float] = None,
        negate: bool = True, # True matches original code's maximization goal
        bounds: Optional[list[tuple[float, float]]] = None,
        shift: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Initialize Levy function.

        Args:
            dim: Dimension of the function.
            noise_std: Standard deviation of observation noise. Default: None.
            negate: If True, negate the function for maximization. Default: True.
            bounds: Custom bounds for the function domain. If None, uses [-10, 10]^d.
        """
        self.dim = dim
        if bounds is None:
             # Using bounds from the original code context
            self._bounds = [(-10.0, 10.0)] * self.dim
        else:
            self._bounds = bounds
        # Ensure optimizer matches dimension
        self._optimizers = [(1.0,) * self.dim]
        self.shift = shift

        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the Levy function (standard minimization form).

        Args:
            X: A (batch_shape) x d tensor of inputs.

        Returns:
            A (batch_shape) tensor of function values.
        """
        if X.ndim < 1 or X.shape[-1] != self.dim:
             raise ValueError(f"Input tensor X must have last dimension equal to {self.dim}")
        if self.shift is not None:
            # Shift the input tensor if a shift is provided
            X = X - self.shift

        # Calculate w = 1 + (x - 1) / 4
        w = 1.0 + (X - 1.0) / 4.0

        # Calculate terms using tensor slicing
        term1 = torch.sin(torch.pi * w[..., 0]) ** 2

        w_mid = w[..., :-1] # w_1 to w_{d-1}
        term2_sum = (w_mid - 1.0) ** 2 * (1.0 + 10.0 * torch.sin(torch.pi * w_mid + 1.0) ** 2)
        term2 = torch.sum(term2_sum, dim=-1) # Sum across the feature dimension

        w_last = w[..., -1] # w_d
        term3 = (w_last - 1.0) ** 2 * (1.0 + torch.sin(2.0 * torch.pi * w_last) ** 2)

        result = term1 + term2 + term3
        return result

# Example Usage (Optional)
if __name__ == "__main__":
    # Ackley Example
    ackley2d = Ackley(dim=2)
    print(f"Ackley dim: {ackley2d.dim}, bounds: {ackley2d.bounds}")
    test_X_ackley = torch.tensor([[0.0, 0.0], [1.0, 1.0], [-1.0, 2.0]])
    # BoTorch functions return negated value by default (for maximization)
    print(f"Ackley values (negated): {ackley2d(test_X_ackley)}")
    # To get standard minimization value:
    print(f"Ackley values (min form): {-ackley2d(test_X_ackley)}")
    print("-" * 20)

    # Rosenbrock Example
    rosen2d = Rosenbrock(dim=2)
    print(f"Rosenbrock dim: {rosen2d.dim}, bounds: {rosen2d.bounds}")
    test_X_rosen = torch.tensor([[1.0, 1.0], [0.0, 0.0], [2.0, 3.0]])
    print(f"Rosenbrock values (negated): {rosen2d(test_X_rosen)}")
    print(f"Rosenbrock values (min form): {-rosen2d(test_X_rosen)}")
    print("-" * 20)

    # Levy Example
    levy2d = Levy(dim=2)
    print(f"Levy dim: {levy2d.dim}, bounds: {levy2d.bounds}")
    test_X_levy = torch.tensor([[1.0, 1.0], [0.0, 0.0], [-5.0, 5.0]])
    print(f"Levy values (negated): {levy2d(test_X_levy)}")
    print(f"Levy values (min form): {-levy2d(test_X_levy)}")
    print("-" * 20)

    # Example with higher dimension and custom bounds
    rosen4d_custom = Rosenbrock(dim=4, bounds=[(-3.0, 3.0)] * 4)
    print(f"Rosenbrock dim: {rosen4d_custom.dim}, bounds: {rosen4d_custom.bounds}")
    test_X_rosen4d = torch.tensor([[1.0, 1.0, 1.0, 1.0], [0.5, 0.5, 0.5, 0.5]])
    print(f"Rosenbrock 4D values (negated): {rosen4d_custom(test_X_rosen4d)}")