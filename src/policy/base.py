import torch
import numpy as np
import time
import os
import abc # Abstract Base Classes
from abc import abstractmethod
from omegaconf import DictConfig
import matplotlib.pyplot as plt
from botorch.utils.sampling import draw_sobol_samples # Keep for default initial sampling
import traceback

# Assuming DEFAULT_DTYPE is defined somewhere, e.g.
DEFAULT_DTYPE = torch.float64

class BaseBayesianOptimizer(abc.ABC):
    """
    Abstract Base Class for Bayesian Optimization implementations.
    """
    def __init__(self, config: DictConfig, objective_function):
        self.config = config
        self.objective_function = objective_function

        # Determine device (simplified common logic)
        # Subclasses might override if more complex logic (like checking availability) is needed
        self.device = torch.device(getattr(config, 'device', 'cpu')) # Default to cpu if not specified
        self.dtype = getattr(config, 'dtype', DEFAULT_DTYPE) # Use a default dtype if not specified

        print(f"Using device: {self.device}, dtype: {self.dtype}") # Log the device/dtype being used

        # Use a sub-config for BO specific parameters if available, else use root config
        self.bo_config = getattr(config, 'bo', config)

        # Set random seeds
        seed = getattr(self.config, 'seed', 1234) # Default seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"Set random seed: {seed}")

        # Initialize data storage
        self.data_x = torch.zeros((0, self.bo_config.input_dim), device=self.device, dtype=self.dtype)
        self.data_y = torch.zeros(0, device=self.device, dtype=self.dtype)

        # Define bounds for optimization
        if self.bo_config.domain_min is None or self.bo_config.domain_max is None:
            raise ValueError("Domain bounds must be specified in the configuration.")
        if isinstance(self.bo_config.domain_min, (int, float)):
            self.bo_config.domain_min = [self.bo_config.domain_min] * self.bo_config.input_dim
        if isinstance(self.bo_config.domain_max, (int, float)):
            self.bo_config.domain_max = [self.bo_config.domain_max] * self.bo_config.input_dim
        if len(self.bo_config.domain_min) != self.bo_config.input_dim or len(self.bo_config.domain_max) != self.bo_config.input_dim:
            raise ValueError("Domain bounds must match the input dimension.")
        # Convert bounds to tensors
        self.bounds = [None] * 2
        self.bounds[0] = torch.tensor(self.bo_config.domain_min, device=self.device, dtype=self.dtype)
        self.bounds[1] = torch.tensor(self.bo_config.domain_max, device=self.device, dtype=self.dtype)
        self.bounds = torch.stack(self.bounds, dim=0) # Shape [2, input_dim]
        self.bounds = self.bounds.reshape([2, -1])


        # Track optimization results
        self.objective_mode = getattr(self.bo_config, 'objective', 'maximize') # Default to maximize
        self.best_observed_value = -float('inf') if self.objective_mode == "maximize" else float('inf')
        self.best_x = None
        self.optimization_history = {
            'iterations': [],
            'targets': [],
            'elapsed_times': []
            # Subclasses might add more specific history items like 'acquisition_values'
        }

        # Verbosity
        self.verbose = getattr(self.config, 'verbose', True) # Default to verbose

        # Save directory (optional, depends on subclass needs)
        self.save_dir = getattr(self.config, 'save_dir', None)
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            print(f"Save directory set to: {self.save_dir}")


    def sample_initial_points(self, method='sobol'):
        """
        Sample initial points. Default uses Sobol sequence.
        Subclasses can override this for different sampling strategies (e.g., LHS).
        """
        n_initial = getattr(self.bo_config, 'initial_points', 5) # Default initial points
        if n_initial <= 0:
            print("No initial points requested.")
            return

        if self.data_x.shape[0] > 0:
             print("Data already exists, skipping initial sampling.")
             return

        seed = getattr(self.config, 'seed', 1234)

        if method.lower() == 'sobol':
            # Generate Sobol sequence samples
            initial_x = draw_sobol_samples(
                bounds=self.bounds,
                n=n_initial,
                q=1, # Sample one point at a time for sequential evaluation
                seed=seed
            ).squeeze(1).to(self.device, dtype=self.dtype)
        elif method.lower() == 'random':
             initial_x = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * torch.rand(
                n_initial, self.bo_config.input_dim, device=self.device, dtype=self.dtype
            )
        # Add more methods like 'lhs' if needed
        else:
             raise ValueError(f"Unknown initial sampling method: {method}")


        # Evaluate objective at initial points
        initial_y = torch.zeros(n_initial, device=self.device, dtype=self.dtype)
        print(f"Sampling {n_initial} initial points using '{method}' method...")
        for i in range(n_initial):
            x_np = initial_x[i]
            y_val = self.objective_function(x_np)
            if type(y_val) == torch.Tensor:
                initial_y[i] = y_val
            elif isinstance(y_val, (int, float)):
                initial_y[i] = torch.tensor(y_val, device=self.device, dtype=self.dtype)
            if self.verbose and (i + 1) % 10 == 0: print(f" Evaluated initial point {i+1}/{n_initial}")


        # Update data and best observation using the helper method
        for i in range(n_initial):
             self._update_data_and_best(initial_x[i].unsqueeze(0), initial_y[i].item()) # Add batch dim for helper

        if self.verbose:
            best_val, best_pt = self.get_best()
            print(f"Completed initial sampling. Initial best value: {best_val:.4f}")


    def _update_data_and_best(self, new_x: torch.Tensor, new_y_val: float):
        """
        Updates data_x, data_y and tracks the best observation found so far.
        Assumes new_x is a tensor shape [1, input_dim].
        """
        new_y = torch.tensor([new_y_val], device=self.device, dtype=self.dtype)

        # Update data (ensure correct dimensions)
        self.data_x = torch.cat([self.data_x, new_x.to(self.device, self.dtype)], dim=0)
        self.data_y = torch.cat([self.data_y, new_y.to(self.device, self.dtype)], dim=0)

        # Update best observation
        if self.objective_mode == "maximize":
            if new_y_val > self.best_observed_value:
                self.best_observed_value = new_y_val
                self.best_x = new_x.squeeze(0).cpu().numpy() # Store as numpy
                if self.verbose: print(f"New best found (max): {self.best_observed_value:.4f}")
        else: # Minimize
            if new_y_val < self.best_observed_value:
                self.best_observed_value = new_y_val
                self.best_x = new_x.squeeze(0).cpu().numpy() # Store as numpy
                if self.verbose: print(f"New best found (min): {self.best_observed_value:.4f}")


    def get_best(self):
        """Returns the best value and corresponding x found so far."""
        return self.best_observed_value, self.best_x

    @abstractmethod
    def _initialize_models(self):
        """Initialize surrogate models (e.g., GP). Called once before the loop."""
        pass

    @abstractmethod
    def _update_models(self):
        """Update surrogate models with current data. Called each iteration."""
        pass

    @abstractmethod
    def _propose_next_candidate(self) -> torch.Tensor:
        """
        Determine the next point to query based on models and acquisition strategy.
        Should return a tensor of shape [1, input_dim].
        """
        pass

    def run_optimization(self):
        """
        Run the full Bayesian optimization loop using the Template Method pattern.
        """
        start_time = time.time()
        max_iterations = getattr(self.bo_config, 'max_iterations', 10) # Default iterations

        # --- 1. Initial Sampling ---
        n_initial = getattr(self.bo_config, 'initial_points', 0)
        if self.data_x.shape[0] < n_initial and n_initial > 0:
            # Use default Sobol or allow config to specify method
            initial_sampling_method = getattr(self.bo_config, 'initial_sampling_method', 'sobol')
            self.sample_initial_points(method=initial_sampling_method)
        elif self.data_x.shape[0] == 0:
             print("Warning: No initial points sampled or provided.")
             # Decide if we should raise an error or proceed if max_iterations > 0

        # --- 2. Initialize Models ---
        if self.data_x.shape[0] > 0:
            self._initialize_models()
        else:
             # Handle case with no data - perhaps sample one random point?
             print("Cannot initialize models without data. Sampling one random point.")
             self.sample_initial_points(method='random') # Ensure at least one point
             if self.data_x.shape[0] > 0:
                 self._initialize_models()
             else:
                 raise RuntimeError("Failed to obtain any data points to start optimization.")


        # --- 3. Main Optimization Loop ---
        for iteration in range(max_iterations):
            iter_start_time = time.time()
            current_total_samples = self.data_x.shape[0] # Includes initial points

            if self.verbose:
                print(f"\n--- Iteration {iteration + 1}/{max_iterations} (Sample {current_total_samples + 1}) ---")

            # --- 3a. Update Models ---
            # Subclass implements the specifics (e.g., refitting GP)
            try:
                self._update_models()
            except Exception as e:
                print(f"Error updating models: {e}. Skipping iteration.")
                traceback.print_exc()
                continue

            # --- 3b. Propose Next Candidate ---
            # Subclass implements the specifics (e.g., optimize acquisition fn)
            # commenting out the try-except block for debugging
            try:
                new_x_candidate = self._propose_next_candidate() # Expected shape [1, input_dim]
            except Exception as e:
                 print(f"Error proposing next candidate: {e}. Skipping iteration.")
                 traceback.print_exc()
                 continue

            # --- 3c. Evaluate Objective Function ---
            x_np = new_x_candidate.squeeze(0)
            try:
                 new_y_value = self.objective_function(x_np)
            except Exception as e:
                 print(f"Error evaluating objective function at {x_np}: {e}. Skipping iteration.")
                 traceback.print_exc()
                 continue

            # --- 3d. Update Data and Best Value ---
            self._update_data_and_best(new_x_candidate, new_y_value)

            # --- 3e. Record History ---
            iter_time = time.time() - iter_start_time
            self.optimization_history['iterations'].append(current_total_samples + 1) # Total samples count
            self.optimization_history['targets'].append(new_y_value)
            self.optimization_history['elapsed_times'].append(iter_time)
            # Subclasses can append specific items like acquisition value here if needed

            if self.verbose:
                print(f"Queried point: {x_np}")
                print(f"Objective value: {new_y_value:.4f}")
                best_val, _ = self.get_best()
                print(f"Best value so far: {best_val:.4f}")
                print(f"Iteration time: {iter_time:.2f} seconds")

        # --- 4. Finalization ---
        total_time = time.time() - start_time
        if self.verbose:
            print(f"\nOptimization completed in {total_time:.2f} seconds")
            best_val, best_pt = self.get_best()
            print(f"Final best value: {best_val:.4f} found at X = {best_pt}")

        # --- 5. Return Results ---
        return self._prepare_results_dict(total_time)


    def _prepare_results_dict(self, total_time: float) -> dict:
        """Constructs the dictionary of optimization results."""
        best_val, best_pt = self.get_best()
        return {
            'best_x': best_pt,
            'best_y': best_val,
            'all_x': self.data_x.cpu().numpy(),
            'all_y': self.data_y.cpu().numpy(),
            'history': self.optimization_history,
            'total_time': total_time
        }

    # --- Plotting Methods (Optional in Base, could be moved entirely to Standard BO) ---
    # Keeping them here for potential reuse, but making them check for save_dir

    def plot_optimization_results(self, result=None):
        """Plot optimization progress (Objective vs Iteration, Best vs Iteration)."""
        if not self.save_dir:
            print("No save directory configured. Skipping progress plot.")
            return

        if result is None:
             result = self._prepare_results_dict(0) # Get current data

        if len(result['all_y']) == 0:
             print("No data to plot.")
             return

        plt.figure(figsize=(12, 6))
        iterations = np.arange(1, len(result['all_y']) + 1)

        # Plot objective values at each iteration
        plt.subplot(1, 2, 1)
        plt.plot(iterations, result['all_y'], 'bo-', linewidth=2, markersize=5, label='Observed Value')
        if not np.isnan(result['best_y']):
             plt.axhline(y=result['best_y'], color='r', linestyle='--', label=f'Best value: {result["best_y"]:.4f}')
        plt.xlabel('Sample Number (Initial + Iterations)')
        plt.ylabel('Objective Value')
        plt.title('Objective Values per Sample')
        plt.grid(True)
        plt.legend()

        # Plot best value found so far
        plt.subplot(1, 2, 2)
        if self.objective_mode == "maximize":
            best_so_far = np.maximum.accumulate(result['all_y'])
        else:
            best_so_far = np.minimum.accumulate(result['all_y'])

        plt.plot(iterations, best_so_far, 'go-', linewidth=2, markersize=5)
        plt.xlabel('Sample Number (Initial + Iterations)')
        plt.ylabel(f'Best Objective Value ({self.objective_mode})')
        plt.title('Best Value Found vs. Sample Number')
        plt.grid(True)

        plt.tight_layout()
        plot_path = os.path.join(self.save_dir, f"{self.__class__.__name__}_progress.png")
        plt.savefig(plot_path)
        print(f"Progress plot saved to {plot_path}")
        plt.close()

    def plot_2d_contour(self, result=None):
        """Create a 2D contour plot if input_dim == 2."""
        if self.bo_config.input_dim != 2:
            return # Only plot for 2D

        if not self.save_dir:
            print("No save directory configured. Skipping contour plot.")
            return

        if result is None:
             result = self._prepare_results_dict(0) # Get current data

        if len(result['all_x']) == 0:
             print("No data for 2D contour plot.")
             return

        print("Generating 2D contour plot...")
        plt.figure(figsize=(10, 8))

        try:
            # Create grid
            x_range = np.linspace(self.bo_config.domain_min, self.bo_config.domain_max, 50)
            y_range = np.linspace(self.bo_config.domain_min, self.bo_config.domain_max, 50)
            X_grid, Y_grid = np.meshgrid(x_range, y_range)
            grid_points = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T

            # Evaluate objective on grid (can be slow)
            Z = np.array([self.objective_function(p) for p in grid_points]).reshape(X_grid.shape)

            # Plot contour
            contour = plt.contourf(X_grid, Y_grid, Z, 50, cmap='viridis')
            plt.colorbar(contour, label='Objective Value')

        except Exception as e:
            print(f"Could not generate contour background: {e}")
            traceback.print_exc()
            # Continue to plot points even if contour fails

        # Plot sampled points
        all_x_np = result['all_x']
        n_initial = getattr(self.bo_config, 'initial_points', 0)
        initial_points = all_x_np[:n_initial]
        bo_points = all_x_np[n_initial:]

        if initial_points.shape[0] > 0:
            plt.scatter(initial_points[:, 0], initial_points[:, 1], c='white', s=80, marker='o', edgecolors='black', label=f'Initial Points ({len(initial_points)})')
        if bo_points.shape[0] > 0:
            plt.scatter(bo_points[:, 0], bo_points[:, 1], c='red', s=80, marker='x', label=f'BO Samples ({len(bo_points)})')

        # Plot best point
        if result['best_x'] is not None and not np.isnan(result['best_x']).any():
            plt.scatter([result['best_x'][0]], [result['best_x'][1]], c='gold', s=150, marker='*', edgecolors='black', label=f'Best Point ({result["best_y"]:.4f})', zorder=3)

        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Objective Function Contour with BO Sampling Points')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.5)
        plt.xlim(self.bo_config.domain_min, self.bo_config.domain_max)
        plt.ylim(self.bo_config.domain_min, self.bo_config.domain_max)

        plot_path = os.path.join(self.save_dir, f"{self.__class__.__name__}_contour.png")
        plt.savefig(plot_path)
        print(f"Contour plot saved to {plot_path}")
        plt.close()