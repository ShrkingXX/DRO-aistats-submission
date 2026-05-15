import torch
import numpy as np
import hydra
import math
import time
from dataclasses import dataclass # Import dataclass
from omegaconf import DictConfig, OmegaConf
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import logging
import traceback # Added for error printing in run_single_bo_trial
from typing import Tuple, List, Optional, Any, Dict # Added for type hinting
import warnings

import gpytorch
from botorch.models import SingleTaskGP
from botorch.exceptions import InputDataWarning
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll # Changed from fit_gpytorch_model for clarity
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound, ProbabilityOfImprovement
from botorch.optim import optimize_acqf
from botorch.utils.transforms import standardize, normalize, unnormalize # Need standardize
from omegaconf import DictConfig


from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from botorch.acquisition import qExpectedImprovement
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.generation import MaxPosteriorSampling # For Thompson Sampling
from torch.quasirandom import SobolEngine # For initial points and TS perturbations

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.objectives import Ackley, Rosenbrock, Levy # Objective functions
from src.policy.base import BaseBayesianOptimizer # Base class for Bayesian Optimization
import pfns4bo 
from pfns4bo.scripts.acquisition_functions import TransformerBOMethod
from pfns4bo.scripts.tune_input_warping import fit_input_warping # Optional
# ==================================

# Set default dtype (consistent with DRO experiment)
DEFAULT_DTYPE = torch.float64
torch.set_default_dtype(DEFAULT_DTYPE)

# Configure logging (optional but good practice with Hydra)
log = logging.getLogger(__name__)


class PFNS4BayesianOptimization(BaseBayesianOptimizer):
    """
    Bayesian Optimization using Prior-data Fitted Networks (PFNs),
    adapted from the pfns4bo README to fit the BaseBayesianOptimizer structure.
    Includes fix for device handling in TransformerBOMethod.
    """
    def __init__(self, config: DictConfig, objective_function):
        super().__init__(config, objective_function)

        # --- Specific Initializations for PFNs4BO ---
        self.config = config # Store full config if needed
        self.bo_config = config.bo
        self.gp_config = config.get('gp')
        self.acq_config = config.acquisition
        self.pfn_config = config.get('pfn')
        if self.pfn_config is None:
            raise ValueError("PFN configuration (`method.pfn`) is missing.")
        self.input_dim = self.bo_config.input_dim
        self.dtype = torch.float32 # Default dtype for PFNs4BO
        torch.set_default_dtype(self.dtype)
        model_name = pfns4bo.hebo_plus_model

        # PFN-specific parameters
        self.model_path = self.pfn_config.get('model_path')
        if not self.model_path or not os.path.exists(self.model_path):
             default_model = getattr(pfns4bo, 'hebo_plus_model', None)
             if default_model and os.path.exists(default_model):
                 log.warning(f"PFN model_path '{self.model_path}' not found or specified. Using default: {default_model}")
                 self.model_path = default_model
             else:
                 raise ValueError(f"PFN model_path '{self.model_path}' not found and no default available/found.")

        self.fit_input_encoder = self.pfn_config.get('fit_input_warping', False)
        self.num_pending_samples = self.pfn_config.get('pending_samples', 5000)
        self.pfn_acqf = self.acq_config.get('function', 'ei').lower()

        # Load the pre-trained PFN model
        # log.info(f"Loading PFN model from: {self.model_path}")
        try:
            # Load model to the target device specified in the base class (self.device)
            self.pfn_model = torch.load(self.model_path, map_location=self.device)
            # self.pfn_model = TransformerBOMethod(torch.load(model_name), device=str(self.device))
            self.pfn_model.eval()
            self.pfn_model.to(self.device)
        except FileNotFoundError:
            log.error(f"PFN model file not found at {self.model_path}")
            raise
        except Exception as e:
            log.error(f"Error loading PFN model: {e}", exc_info=True)
            raise

        # Instantiate the PFNs4BO interface
        fit_encoder_func = fit_input_warping if self.fit_input_encoder else None

        # # --- FIX: Pass device as a string ---
        # Convert the torch.device object (self.device) to its string representation
        device_str = str(self.device)
        log.info(f"Instantiating TransformerBOMethod (fit_input_warping={self.fit_input_encoder}, device='{device_str}')")

        try:
            self.pfn_interface = TransformerBOMethod(
                self.pfn_model,
                fit_encoder=fit_encoder_func,
                device=device_str, # Pass the string representation
                # acq_function=self.pfn_acqf
            )
        except Exception as e:
            log.error(f"Error instantiating TransformerBOMethod: {e}", exc_info=True)
            raise

        # PFNs typically work sequentially
        self.batch_size = 1
        log.info(f"PFNs4BO configured: acqf={self.pfn_acqf}, pending_samples={self.num_pending_samples}")
        log.warning("PFNs4BO interface typically suggests one point at a time (batch_size=1).")


    # --- Implementations for Abstract Methods from Base Class ---
    # (These remain the same - no-ops for PFN)
    def _initialize_models(self):
        log.debug("PFN model already loaded in __init__. No initialization step needed.")
        pass

    def _update_models(self):
        log.debug("PFN model is not updated during BO loop.")
        pass

    # --- Candidate Proposal ---
    # (This method remains largely the same, using the initialized pfn_interface)
    def _propose_next_candidate(self) -> torch.Tensor:
        if self.data_x is None or self.data_y is None:
            raise RuntimeError("Observation data is missing for PFN proposal.")

        log.debug("Generating PFN candidate...")

        # 1. Prepare observed data
        X_obs_normalized = normalize(self.data_x, bounds=self.bounds)
        y_obs = self.data_y.squeeze(-1) if self.data_y.ndim > 1 else self.data_y
        # y_obs_np = y_obs.cpu().numpy()
        # X_obs_normalized_np = X_obs_normalized.cpu().numpy()

        # 2. Generate pending points
        sobol = SobolEngine(self.input_dim, scramble=True, seed=torch.randint(0, 10000, (1,)).item())
        X_pen_normalized = sobol.draw(n=self.num_pending_samples).to(dtype=self.dtype, device=self.device)
        # X_pen_normalized_np = X_pen_normalized.cpu().numpy()

        # log.debug(f"Calling PFN observe_and_suggest with {X_obs_normalized_np.shape[0]} observations and {X_pen_normalized_np.shape[0]} pending points.")

        # 3. Use the PFN interface
        try:
            # Pass data as numpy arrays
            suggested_index = self.pfn_interface.observe_and_suggest(
                X_obs=X_obs_normalized,
                y_obs=y_obs,
                X_pen=X_pen_normalized,
            )

        except Exception as e:
             log.error(f"Error calling PFN observe_and_suggest: {e}", exc_info=True)
             log.warning("PFN suggestion failed. Proposing a random point as fallback.")
             fallback_cand_normalized = torch.rand(1, self.input_dim, device=self.device, dtype=self.dtype)
             return fallback_cand_normalized

        # 4. Get the selected candidate (normalized)
        best_pending_point_normalized = X_pen_normalized[suggested_index]
        if best_pending_point_normalized.ndim == 1:
            best_pending_point_normalized = best_pending_point_normalized.unsqueeze(0)

        log.debug(f"PFN proposed candidate (normalized): {best_pending_point_normalized.cpu().numpy()}")
        return best_pending_point_normalized

    # --- Main Optimization Execution ---
    # (This method remains the same)
    def run_optimization(self) -> dict:
        start_time = time.time()
        self.optimization_history = {'all_x': [], 'all_y': [], 'timestamps': []}

        # 1. Sample Initial Points
        self.sample_initial_points()
        if self.data_x is None:
             log.error("Initial sampling failed.")
             return {'error': 'Initial sampling failed'}
        log.info(f"Collected {self.data_x.shape[0]} initial points.")
        self.optimization_history['all_x'].append(self.data_x.clone())
        self.optimization_history['all_y'].append(self.data_y.clone())
        ts_shape = self.data_y.squeeze(-1).shape if self.data_y.ndim > 1 else self.data_y.shape
        self.optimization_history['timestamps'].append(torch.full(ts_shape, time.time() - start_time).unsqueeze(-1))

        # 2. Initialize Models (No-op for PFN)
        self._initialize_models()

        # 3. Main Optimization Loop
        num_iterations_to_run = self.bo_config.max_iterations
        # Use self.bo_config.verbose which should be set from experiment_params
        iterator = tqdm(range(num_iterations_to_run), desc="PFN BO Iterations", unit="iter", disable=not self.config.verbose)
        current_total_evals = self.bo_config.initial_points

        for i in iterator:
            iter_start_time = time.time()
            if current_total_evals >= self.bo_config.initial_points + self.bo_config.max_iterations:
                 log.info("Reached maximum number of evaluations.")
                 break

            # Propose next candidate (normalized)
            try:
                X_next_normalized = self._propose_next_candidate()
            except (RuntimeError, ValueError) as e:
                 log.error(f"Iteration {i}: Failed to propose candidates: {e}")
                 break

            # Unnormalize candidate
            X_next = unnormalize(X_next_normalized, bounds=self.bounds)

            # Evaluate objective using the objective function passed during init
            # Ensure objective_function handles batch_size=1 correctly
            Y_next = self.objective_function(X_next) # Use base class helper

            # Append new data (original scale)
            self.data_x = torch.cat((self.data_x, X_next), dim=0)
            self.data_y = torch.cat((self.data_y, Y_next), dim=0)
            current_total_evals += self.batch_size # batch_size is 1 for PFN

            # Update history
            iter_end_time = time.time()
            self.optimization_history['all_x'].append(X_next.clone())
            self.optimization_history['all_y'].append(Y_next.clone())
            ts_shape_next = Y_next.squeeze(-1).shape if Y_next.ndim > 1 else Y_next.shape
            self.optimization_history['timestamps'].append(torch.full(ts_shape_next, iter_end_time - start_time).unsqueeze(-1))

            # Log progress
            if self.config.verbose: # Check bo_config for verbosity
                 current_best_y = self.data_y.max().item()
                 iterator.set_postfix(best_y=f"{current_best_y:.4f}")


        # 4. Consolidate Results
        total_time = time.time() - start_time
        log.info(f"PFN optimization finished after {total_time:.2f} seconds and {current_total_evals} evaluations.")

        all_x_tensor = torch.cat(self.optimization_history['all_x'], dim=0)
        all_y_tensor = torch.cat(self.optimization_history['all_y'], dim=0)

        best_y_overall, best_idx = torch.max(all_y_tensor, dim=0)
        best_x_overall = all_x_tensor[best_idx]

        if best_x_overall.ndim == 1: best_x_overall = best_x_overall.unsqueeze(0)
        if best_y_overall.ndim > 0: best_y_overall = best_y_overall.squeeze()

        # Return numpy arrays as per the original PFN run_optimization example
        results = {
            'best_x': best_x_overall.cpu().numpy(), 'best_y': best_y_overall.cpu().numpy(),
            'all_x': all_x_tensor.cpu().numpy(), 'all_y': all_y_tensor.cpu().numpy(),
            'total_time': total_time, 'optimization_history': self.optimization_history
        }
        return results


@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10  # Note: The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        # Initialize failure tolerance based on dim and batch_size
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )

class TrustRegionBayesianOptimization(BaseBayesianOptimizer):
    """
    Trust Region Bayesian Optimization (TuRBO) implementation, adapted from
    turbo_1.ipynb to fit the StandardBayesianOptimization structure
    and satisfy the abstract methods of BaseBayesianOptimizer.
    """
    def __init__(self, config: DictConfig, objective_function):
        super().__init__(config, objective_function)

        # --- Specific Initializations for TuRBO ---
        self.config = config
        self.bo_config = config.bo
        self.gp_config = config.gp
        self.acq_config = config.acquisition
        self.turbo_config = config.turbo

        # TuRBO parameters
        self.batch_size = self.turbo_config.get('batch_size', 4)
        self.length_init = self.turbo_config.get('length_init', 0.8)
        self.length_min_tr = self.turbo_config.get('length_min', 0.5**7)
        self.length_max_tr = self.turbo_config.get('length_max', 1.6)
        self.success_tolerance = self.turbo_config.get('success_tolerance', 10)

        # Acquisition choice & params
        self.acqf_choice = self.acq_config.get("function", "ts").lower()
        if self.acqf_choice not in ["ts", "ei"]:
            log.warning(f"Unsupported acqf '{self.acqf_choice}'. Defaulting to 'ts'.")
            self.acqf_choice = "ts"
        self.num_restarts = self.acq_config.get("opt_restarts", 10)
        self.raw_samples = self.acq_config.get("opt_samples", 512)

        # Other settings
        self.max_cholesky_size = self.gp_config.get("max_cholesky_size", float("inf"))

        # State variables
        self.state: TurboState = None
        self.model: SingleTaskGP = None
        self.mll: ExactMarginalLogLikelihood = None
        self.input_dim = config.bo.input_dim

        # Transformed data
        self.train_x_normalized = None
        self.train_y_standardized = None

        log.info(f"TuRBO (Notebook Refactor) configured: acqf={self.acqf_choice}, batch_size={self.batch_size}")

    # --- Implementations for Abstract Methods from Base Class ---

    def _initialize_models(self):
        """
        Initializes the TuRBO state. In TuRBO, there isn't an initial global
        model fit like in standard BO. Instead, we initialize the state tracker.
        This method satisfies the abstract requirement from BaseBayesianOptimizer.
        """
        if self.data_y is None or len(self.data_y) == 0:
            # This should ideally not happen if run_optimization calls sample_initial_points first
            log.warning("Cannot initialize TuRBO state without initial data. Ensure initial points are sampled.")
            return # Or raise error

        initial_best_y = self.data_y.max().item()
        self.state = TurboState(dim=self.input_dim, batch_size=self.batch_size, best_value=initial_best_y,
                               length=self.length_init, length_min=self.length_min_tr,
                               length_max=self.length_max_tr, success_tolerance=self.success_tolerance)
        log.info(f"TuRBO state initialized: {self.state}")

    def _update_models(self):
        """
        Fits the SingleTaskGP model to the current *transformed* data.
        This is called in each iteration of the TuRBO loop.
        Satisfies the abstract requirement from BaseBayesianOptimizer.
        """
        # Prepare transformed data first
        self._prepare_data_transforms()

        if self.train_x_normalized is None or self.train_y_standardized is None:
            log.error("Normalized/standardized data not available for model fitting.")
            raise ValueError("Transformed data required for model fitting.")

        log.debug("Fitting GP model for TuRBO iteration...")
        likelihood = GaussianLikelihood(noise_constraint=Interval(
            self.gp_config.get("noise_constraint_low", 1e-8),
            self.gp_config.get("noise_constraint_high", 1e-3)
        ))
        covar_module = ScaleKernel(
            MaternKernel(
                nu=2.5, ard_num_dims=self.input_dim,
                lengthscale_constraint=Interval(
                     self.gp_config.get("lengthscale_constraint_low", 0.005),
                     self.gp_config.get("lengthscale_constraint_high", 4.0)
                )
            )
        )
        self.model = SingleTaskGP(
            self.train_x_normalized, self.train_y_standardized,
            covar_module=covar_module, likelihood=likelihood
        ).to(device=self.device, dtype=self.dtype)
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)

        try:
             with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
                 fit_gpytorch_mll(self.mll,
                                  max_retries=self.gp_config.get("fit_max_retries", 5),
                                  options={"maxiter": self.gp_config.get("train_iter", 100)})
        except Exception as fit_e:
             log.error(f"GP fitting failed: {fit_e}", exc_info=True)
             raise RuntimeError("GP model fitting failed.") from fit_e

        log.debug("GP model fitting complete.")

    def _propose_next_candidate(self) -> torch.Tensor:
        """
        Generates the next batch of candidate points using the notebook's logic
        (TS or EI within the trust region). Operates in the normalized space.
        Satisfies the abstract requirement from BaseBayesianOptimizer.
        Returns candidates in the *normalized* space.
        """
        if self.model is None:
            raise RuntimeError("GP Model not initialized for candidate proposal.")
        if self.state is None:
            raise RuntimeError("TuRBO state not initialized for candidate proposal.")
        if self.train_x_normalized is None or self.train_y_standardized is None:
             raise ValueError("Transformed data required for candidate proposal.")

        n_candidates = min(5000, max(2000, 200 * self.input_dim)) # For TS

        # Determine TR center (normalized) and bounds (normalized)
        x_center_normalized = self.train_x_normalized[self.data_y.argmax(), :].clone()

        # Get anisotropic weights from fitted model's lengthscales
        if hasattr(self.model, 'covar_module') and \
           hasattr(self.model.covar_module, 'base_kernel') and \
           hasattr(self.model.covar_module.base_kernel, 'lengthscale'):
             weights = self.model.covar_module.base_kernel.lengthscale.squeeze().detach()
             weights = weights / weights.mean()
             weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
        else:
             log.warning("Could not get lengthscales from model, using isotropic TR.")
             weights = torch.ones(self.input_dim, device=self.device, dtype=self.dtype)

        tr_lb = torch.clamp(x_center_normalized - weights * self.state.length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center_normalized + weights * self.state.length / 2.0, 0.0, 1.0)
        bounds_normalized = torch.stack([tr_lb, tr_ub])

        X_next_normalized = None

        try:
            if self.acqf_choice == "ts":
                log.debug("Using Thompson Sampling (TS) for candidate generation.")
                dim = self.input_dim
                sobol = SobolEngine(dim, scramble=True, seed=torch.randint(0, 10000, (1,)).item())
                pert = sobol.draw(n_candidates).to(dtype=self.dtype, device=self.device)
                pert = tr_lb + (tr_ub - tr_lb) * pert

                prob_perturb = min(20.0 / dim, 1.0)
                mask = torch.rand(n_candidates, dim, dtype=self.dtype, device=self.device) <= prob_perturb
                ind = torch.where(mask.sum(dim=1) == 0)[0]
                mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=self.device)] = 1

                X_cand_normalized = x_center_normalized.expand(n_candidates, dim).clone()
                X_cand_normalized[mask] = pert[mask]

                thompson_sampling = MaxPosteriorSampling(model=self.model, replacement=False)
                with torch.no_grad():
                    X_next_normalized = thompson_sampling(X_cand_normalized, num_samples=self.batch_size)

            elif self.acqf_choice == "ei":
                log.debug("Using Expected Improvement (EI) for candidate generation.")
                best_f_standardized = self.train_y_standardized.max()
                ei = LogExpectedImprovement(self.model, best_f=best_f_standardized)

                X_next_normalized, acq_value = optimize_acqf(
                    acq_function=ei, bounds=bounds_normalized, q=self.batch_size,
                    num_restarts=self.num_restarts, raw_samples=self.raw_samples,
                    options=self.acq_config.get("optimizer_options", {"batch_limit": 5, "maxiter": 200}),
                )
            else:
                raise ValueError(f"Unknown acquisition function: {self.acqf_choice}")

        except Exception as e:
            log.error(f"Error during acquisition function optimization: {e}", exc_info=True)
            log.warning("Acquisition optimization failed. Proposing random points within TR as fallback.")
            rand_pts = torch.rand(self.batch_size, self.input_dim, device=self.device, dtype=self.dtype)
            X_next_normalized = tr_lb + (tr_ub - tr_lb) * rand_pts

        return X_next_normalized.detach()

    # --- Helper Methods Specific to this Implementation ---

    def _update_state(self, Y_next: torch.Tensor):
        """
        Helper to update the TuRBO state based on the notebook logic.
        Modifies self.state in-place.
        """
        if self.state is None:
            log.error("TuRBO state is not initialized. Cannot update.")
            return

        Y_next = Y_next.to(self.device, self.dtype)
        current_max = Y_next.max().item()

        # Logic from update_state function in notebook
        if current_max > self.state.best_value + 1e-3 * math.fabs(self.state.best_value):
            self.state.success_counter += 1
            self.state.failure_counter = 0
        else:
            self.state.success_counter = 0
            self.state.failure_counter += 1

        if self.state.success_counter == self.state.success_tolerance:
            self.state.length = min(2.0 * self.state.length, self.state.length_max)
            self.state.success_counter = 0
            log.debug(f"TR Expanded to {self.state.length:.3f}")
        elif self.state.failure_counter == self.state.failure_tolerance:
            self.state.length /= 2.0
            self.state.failure_counter = 0
            log.debug(f"TR Shrunk to {self.state.length:.3f}")

        self.state.best_value = max(self.state.best_value, current_max)
        if self.state.length < self.state.length_min:
            self.state.restart_triggered = True
            log.info("TuRBO restart triggered: TR length below minimum.")

    def _prepare_data_transforms(self):
        """Normalize X and Standardize Y for GP fitting."""
        if self.data_x is None or self.data_y is None:
            raise ValueError("Data not available for normalization/standardization.")

        self.train_x_normalized = normalize(self.data_x, bounds=self.bounds)
        y_temp = self.data_y if self.data_y.ndim == 2 else self.data_y.unsqueeze(-1)
        self.train_y_standardized = (y_temp - y_temp.mean()) / y_temp.std().clamp(min=1e-6)

    # --- Main Optimization Execution ---

    def run_optimization(self) -> dict:
        """
        Runs the TuRBO optimization loop, structured similarly to StandardBO.
        """
        start_time = time.time()
        self.optimization_history = {'all_x': [], 'all_y': [], 'timestamps': []}

        # 1. Sample Initial Points (from Base Class)
        self.sample_initial_points()
        if self.data_x is None: # Check if sampling failed or returned None
             log.error("Initial sampling did not produce data. Aborting.")
             return {'error': 'Initial sampling failed'}
        log.info(f"Collected {self.data_x.shape[0]} initial points.")
        self.optimization_history['all_x'].append(self.data_x.clone())
        self.optimization_history['all_y'].append(self.data_y.clone())
        ts_shape = self.data_y.squeeze(-1).shape if self.data_y.ndim > 1 else self.data_y.shape
        self.optimization_history['timestamps'].append(torch.full(ts_shape, time.time() - start_time).unsqueeze(-1))


        # 2. Initialize TuRBO State (Satisfies _initialize_models concept)
        try:
            self._initialize_models() # Calls the state initialization logic
        except RuntimeError as e:
             log.error(f"Failed to initialize TuRBO state: {e}")
             return {'error': 'State initialization failed'}
        if self.state is None: # Double check if state init failed silently
             log.error("TuRBO state object is None after initialization call.")
             return {'error': 'State object None after init'}


        # 3. Main Optimization Loop
        num_turbo_steps = self.bo_config.max_iterations // self.batch_size
        iterator = tqdm(range(num_turbo_steps), desc=f"TuRBO Steps (Batch Size {self.batch_size})", unit="step", disable=not self.config.verbose)
        current_total_evals = self.bo_config.initial_points

        for i in iterator:
            iter_start_time = time.time()
            # Check termination conditions
            if current_total_evals >= self.bo_config.initial_points + self.bo_config.max_iterations:
                 log.info("Reached maximum number of evaluations.")
                 break
            if self.state.restart_triggered:
                 log.info("TuRBO restart triggered. Stopping run.")
                 break

            # Update/fit the GP model for this iteration
            try:
                # _update_models includes data prep and fitting
                self._update_models()
            except (RuntimeError, ValueError) as e:
                 log.error(f"Iteration {i}: Failed to update GP model: {e}")
                 # Optionally try to continue with old model or break
                 break # Safest to break if model fitting fails

            # Propose next candidates (normalized)
            try:
                X_next_normalized = self._propose_next_candidate()
            except (RuntimeError, ValueError) as e:
                 log.error(f"Iteration {i}: Failed to propose candidates: {e}")
                 # Optionally try random or break
                 break # Safest to break

            # Unnormalize candidates
            X_next = unnormalize(X_next_normalized, bounds=self.bounds)

            # Evaluate objective
            Y_next = self.objective_function(X_next)

            # Update TuRBO state (counters, length, best_value)
            self._update_state(Y_next)

            # Append new data (original scale)
            self.data_x = torch.cat((self.data_x, X_next), dim=0)
            self.data_y = torch.cat((self.data_y, Y_next), dim=0)
            current_total_evals += self.batch_size

            # Update history
            iter_end_time = time.time()
            self.optimization_history['all_x'].append(X_next.clone())
            self.optimization_history['all_y'].append(Y_next.clone())
            ts_shape_next = Y_next.squeeze(-1).shape if Y_next.ndim > 1 else Y_next.shape
            self.optimization_history['timestamps'].append(torch.full(ts_shape_next, iter_end_time - start_time).unsqueeze(-1))

            # Log progress
            if self.config.verbose:
                 iterator.set_postfix(
                     best_val=f"{self.state.best_value:.3e}",
                     tr_len=f"{self.state.length:.2e}",
                     fail_succ=f"{self.state.failure_counter}/{self.state.success_counter}"
                 )

        # 4. Consolidate Results
        total_time = time.time() - start_time
        log.info(f"TuRBO optimization finished after {total_time:.2f} seconds and {current_total_evals} evaluations.")

        all_x_tensor = torch.cat(self.optimization_history['all_x'], dim=0)
        all_y_tensor = torch.cat(self.optimization_history['all_y'], dim=0)

        best_y_overall, best_idx = torch.max(all_y_tensor, dim=0)
        best_x_overall = all_x_tensor[best_idx]

        if best_x_overall.ndim == 1: best_x_overall = best_x_overall.unsqueeze(0)
        if best_y_overall.ndim > 0: best_y_overall = best_y_overall.squeeze()

        results = {
            'best_x': best_x_overall.cpu().numpy(), 'best_y': best_y_overall.cpu().numpy(),
            'all_x': all_x_tensor.cpu().numpy(), 'all_y': all_y_tensor.cpu().numpy(),
            'total_time': total_time, 'optimization_history': self.optimization_history
        }
        return results
class StandardBayesianOptimization(BaseBayesianOptimizer):
    """
    Standard Bayesian Optimization implementation using BoTorch/GPyTorch,
    inheriting from BaseBayesianOptimizer. Adapted for consistency with DRO experiments.
    """
    def __init__(self, config: DictConfig, objective_function):
        # Call parent constructor for common setup (device, dtype, bounds, data storage etc.)
        super().__init__(config, objective_function)

        # --- Specific Initializations for Standard BO ---

        # Store relevant sub-configs (similar to DRO)
        # We expect the main script to populate these within the passed 'config'
        self.bo_config = config.bo # General BO settings (dim, iterations, etc.)
        self.gp_config = config.gp # GP model settings (kernel, training)
        self.acquisition_config = config.acquisition # Acquisition function settings

        # GP Model and Marginal Log Likelihood (initialized later)
        self.model = None
        self.mll = None
        self.train_y_standardized = None # To store standardized targets

        # Add acquisition value tracking to history (if not already in base class)
        if 'acquisition_values' not in self.optimization_history:
            self.optimization_history['acquisition_values'] = []

        # Override initial sampling method if specified in config (consistent with DRO)
        self.initial_sampling_method = getattr(self.bo_config, 'initial_sampling_method', 'sobol') # Default to Sobol for standard BO, but allow override

    def _initialize_models(self):
        """
        Initializes the SingleTaskGP model. In standard BO, this involves fitting
        the model to the initial data.
        """
        if self.data_x is None or self.data_x.shape[0] == 0:
             log.warning("Attempting to initialize model with no data. Ensure initial points are sampled first.")
             # Rely on the base class run_optimization loop to call sample_initial_points first
             return

        log.info("Initializing GP model...")
        self._update_models() # Fit model to initial data

    def _update_models(self):
        """
        Update (fit) the SingleTaskGP model to the current data (self.data_x, self.data_y)
        using BoTorch's fit_gpytorch_mll.
        """
        train_x = self.data_x
        train_y = self.data_y

        if train_x is None or train_x.shape[0] == 0:
            log.error("Cannot update model with no data.")
            raise ValueError("Cannot update model with no data.")

        # Convert to a maximization problem if minimizing (for internal GP modeling)
        # train_y_objective = -train_y if self.objective_mode == "minimize" else train_y

        # Standardize targets for numerical stability
        # Ensure train_y_objective is 1D before unsqueezing for standardize
        
        self.train_y_standardized = standardize(train_y.squeeze(-1) if train_y.ndim > 1 else train_y) # Shape [n]

        # Reshape standardized targets to [n, 1] for GP model input
        train_y_gp = self.train_y_standardized.unsqueeze(-1) # Shape [n, 1]
        # train_y_gp = train_y.reshape([1, -1]) # Use original targets for fitting, not standardized

        # --- Define GP Model ---
        # Create a new model instance each time or update?
        # For simplicity and consistency with many BoTorch examples, let's recreate it.
        # If stateful updates are needed (e.g., warm-starting hyperparameters),
        # the logic would need adjustment.
        # if self.model is None: # Original logic
        if True: # Recreate model on each update
            log.debug("Creating new SingleTaskGP model for fitting.")
            # Use SingleTaskGP with standardized outputs
            with warnings.catch_warnings():
                # This line specifically targets the InputDataWarning
                warnings.simplefilter("ignore", category=InputDataWarning)
                # This ignores other potential runtime warnings during init/fit
                # warnings.simplefilter("ignore", category=RuntimeWarning)
                model = SingleTaskGP(
                    train_X=train_x,
                    train_Y=train_y_gp, # Use [n, 1] shape
                    # outcome_transform=Standardize(m=1), # Optional: Standardize outputs
                    # input_transform=Normalize(d=input_dim), # Ensure this remains commented out or removed
                )
                self.model = model.to(device=self.device, dtype=self.dtype)

            # Add noise constraint if configured
            noise_constraint = getattr(self.gp_config, 'noise_constraint', None)
            if noise_constraint is not None and noise_constraint > 0:
                if hasattr(self.model, 'likelihood') and hasattr(self.model.likelihood, 'noise_covar'):
                    self.model.likelihood.noise_covar.register_constraint(
                        "raw_noise", gpytorch.constraints.GreaterThan(noise_constraint)
                    )
                    log.debug(f"Applied noise constraint > {noise_constraint}")
                else:
                    log.warning("Could not apply noise constraint: Model structure unexpected.")
        # else: # Logic for updating existing model (less common with fit_gpytorch_mll)
        #     log.debug("Updating existing GP model with new data.")
        #     self.model.set_train_data(train_x, train_y_gp, strict=False) # strict=False allows changing data size

        # Define the Marginal Log Likelihood
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model).to(self.device, self.dtype)

        # --- Train Model using fit_gpytorch_mll ---
        log.debug("Fitting GP model using fit_gpytorch_mll...")
        # fit_gpytorch_mll handles the optimization loop internally
        # It uses L-BFGS-B by default
        # Options can be passed via `optimizer_kwargs`, `max_retries`, etc. if needed
        # Example: fit_gpytorch_mll(self.mll, optimizer_kwargs={'options': {'maxiter': 100}})
        try:
             fit_gpytorch_mll(self.mll)
        except Exception as e:
             log.error(f"Error during fit_gpytorch_mll: {e}", exc_info=True)
             # Consider adding fallback or error handling logic here
             raise RuntimeError("GP model fitting failed.") from e


        log.debug("GP model fitting complete.")

        # Model is already in eval mode after fit_gpytorch_mll finishes
        # self.model.eval() # Not strictly necessary but doesn't hurt
        # self.mll.eval()

    def _get_acquisition_function(self):
        """Create acquisition function based on configuration and current model state."""
        if self.model is None or self.train_y_standardized is None:
            log.error("Model must be initialized/updated before creating acquisition function.")
            raise RuntimeError("Model not available for acquisition function.")

        # Best observed value in the *standardized* space (since GP models standardized data)
        try:
             # Ensure train_y_standardized is on the correct device
             best_f_standardized = self.train_y_standardized.max().to(self.device, self.dtype)
        except IndexError: # Handle empty tensor case if it somehow occurs
            log.error("train_y_standardized is empty. Cannot determine best_f.")
            raise RuntimeError("Cannot determine best standardized value.")


        # Access acquisition config safely using getattr
        acq_name = getattr(self.acquisition_config, 'function', 'ei').lower() # Default to ei
        xi = getattr(self.acquisition_config, 'xi', 0.01) # Default xi for EI/PI
        kappa = getattr(self.acquisition_config, 'kappa', 2.5) # Default kappa for UCB

        log.debug(f"Creating acquisition function: {acq_name.upper()} (xi={xi}, kappa={kappa})")

        if acq_name == "ei":
            acq_func = ExpectedImprovement(
                model=self.model,
                best_f=best_f_standardized,
                maximize=True, # Always maximize in the standardized space (objective converted earlier)
                xi=xi
            )
        elif acq_name == "ucb":
            acq_func = UpperConfidenceBound(
                model=self.model,
                beta=kappa, # BoTorch uses 'beta' for the trade-off parameter
                maximize=True # Always maximize
            )
        elif acq_name == "pi":
            acq_func = ProbabilityOfImprovement(
                model=self.model,
                best_f=best_f_standardized,
                maximize=True, # Always maximize
                xi=xi
            )
        else:
            log.error(f"Unknown acquisition function specified: {acq_name}")
            raise ValueError(f"Unknown acquisition function: {acq_name}")

        return acq_func

    def _propose_next_candidate(self) -> torch.Tensor:
        """
        Optimize the acquisition function to determine the next point to query.
        Returns a tensor of shape [1, input_dim].
        """
        if self.model is None:
             # This can happen if initialization failed or wasn't called
             log.error("Cannot propose candidate: GP Model not initialized.")
             raise RuntimeError("GP Model not initialized.")

        acq_function = self._get_acquisition_function()

        # Get optimization parameters from config
        # These parameters mirror DRO's simulation._optimize_acquisition somewhat
        num_restarts = getattr(self.acquisition_config, 'opt_restarts', 10) # Equivalent to 'num_restarts'
        raw_samples = getattr(self.acquisition_config, 'opt_samples', 512) # Equivalent to 'num_samples' for initial seeding

        # BoTorch optimize_acqf options (can be nested in config if needed)
        # Example: config.acquisition.optimizer_options: {"batch_limit": 5, "maxiter": 200}
        opt_options = getattr(self.acquisition_config, 'optimizer_options', {"batch_limit": 5, "maxiter": 200})

        log.debug(f"Optimizing acquisition function using {num_restarts} restarts and {raw_samples} raw samples...")

        try:
            candidates, acq_value = optimize_acqf(
                acq_function=acq_function,
                bounds=self.bounds, # Use bounds from base class (should be shape [2, dim])
                q=1, # Query one point at a time for standard sequential BO
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options=opt_options,
                return_best_only=True,
                 # Default is True, explicitly set for clarity
                # Note: Some BoTorch versions might handle constraints differently here if needed
            )
        except Exception as e:
            log.error(f"Error during acquisition function optimization: {e}", exc_info=True)
             # Fallback? Suggest random point within bounds?
            log.warning("Acquisition optimization failed. Proposing a random point as fallback.")
            candidates = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * torch.rand(1, self.bo_config.input_dim, device=self.device, dtype=self.dtype)
            acq_value = torch.tensor([float('nan')], device=self.device, dtype=self.dtype) # Indicate failure

        # candidates shape should be [1, input_dim] because q=1
        new_x = candidates.detach()

        # Store acquisition value in history (specific to this BO type)
        self.optimization_history['acquisition_values'].append(acq_value.item())
        log.debug(f"Proposed candidate: {new_x.cpu().numpy()}, Acq value: {acq_value.item():.4f}")

        return new_x



def setup_experiment(cfg: DictConfig) -> Tuple[Any, torch.Tensor, int, str]:
    """Sets up the objective function, bounds, dimension, and save directory."""
    log.info("--- Experiment Configuration Setup ---")
    input_dim = cfg.test_function.input_dim
    domain_min_raw = cfg.test_function.domain_min
    domain_max_raw = cfg.test_function.domain_max

    # Ensure bounds are lists of floats
    if isinstance(domain_min_raw, (int, float)):
        domain_min = [float(domain_min_raw)] * input_dim
        domain_max = [float(domain_max_raw)] * input_dim
    else:
        # Assumes domain_min/max_raw are iterable (like lists from YAML)
        domain_min = [float(b) for b in domain_min_raw]
        domain_max = [float(b) for b in domain_max_raw]

    # Convert bounds to [2, dim] tensor for BoTorch compatibility
    bounds_tensor = torch.tensor([domain_min, domain_max], dtype=DEFAULT_DTYPE)
    log.info(f"Input Dimension: {input_dim}")
    log.info(f"Bounds: Min={domain_min}, Max={domain_max}")

    objective_name = cfg.test_function.name
    negate = bool(cfg.test_function.get('negate', True)) # Default to negate for maximization
    noise_std = cfg.test_function.get('noise_std', 0.0)
    log.info(f"Objective Function: {objective_name} (Negate={negate}, Noise Std={noise_std})")

    # Instantiate objective function (ensure it handles noise and negation)
    if objective_name == "Ackley":
        objective_function = Ackley(dim=input_dim, bounds=bounds_tensor, negate=negate, noise_std=noise_std)
    elif objective_name == "Rosenbrock":
        objective_function = Rosenbrock(dim=input_dim, bounds=bounds_tensor, negate=negate, noise_std=noise_std)
    elif objective_name == "Levy":
        objective_function = Levy(dim=input_dim, bounds=bounds_tensor, negate=negate, noise_std=noise_std)
    else:
        log.error(f"Unsupported objective function specified: {objective_name}")
        raise ValueError(f"Unsupported objective function: {objective_name}")

    # Determine save directory (Hydra's output dir by default)
    save_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    if cfg.get('save_dir'): # Allow overriding save_dir in config
        save_dir = cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)
    log.info(f"Output directory: {save_dir}")
    log.info("------------------------------------")

    return objective_function, bounds_tensor, input_dim, save_dir


def configure_bo_trial_settings(
    base_cfg: DictConfig, # The main config loaded by Hydra
    domain_min: List[float],
    domain_max: List[float],
    input_dim: int,
    trial_idx: int
) -> DictConfig:
    """
    Constructs the specific configuration object needed by StandardBayesianOptimization
    for a trial, creating the nested structure (bo, gp, acquisition).
    """
    log.info(f"--- Configuring Standard BO Settings for Trial {trial_idx + 1} ---")

    bo_trial_cfg = OmegaConf.create()

    # --- 1. BO Sub-config (bo.*) ---
    bo_trial_cfg.bo = OmegaConf.create()
    bo_trial_cfg.bo.input_dim = input_dim
    bo_trial_cfg.bo.domain_min = domain_min # List of floats
    bo_trial_cfg.bo.domain_max = domain_max # List of floats
    bo_trial_cfg.bo.initial_points = base_cfg.experiment_params.initial_points
    bo_trial_cfg.bo.max_iterations = base_cfg.experiment_params.max_iterations
    bo_trial_cfg.bo.seed = base_cfg.experiment_params.seed_start + trial_idx
    # Assuming maximization because objective function has negate=True
    bo_trial_cfg.bo.objective_mode = "maximize"
    # Device and Dtype
    bo_trial_cfg.bo.device = base_cfg.experiment_params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    bo_trial_cfg.bo.dtype = "float64" # Match DEFAULT_DTYPE or make configurable
    bo_trial_cfg.bo.verbose = base_cfg.experiment_params.get('verbose', True)

    # Check for BO specific overrides in base_cfg (e.g., under standard_bo_params.bo)
    std_bo_params = base_cfg.get('standard_bo_params', {})
    if 'bo' in std_bo_params:
        log.info("Merging parameters from base_cfg.standard_bo_params.bo")
        bo_trial_cfg.bo = OmegaConf.merge(bo_trial_cfg.bo, std_bo_params.bo)

    # --- 2. GP Sub-config (gp.*) ---
    bo_trial_cfg.gp = OmegaConf.create()
    default_gp_params = {
        'kernel': 'matern',
        'lr': 0.1,
        'train_iter': 50,
        'noise_constraint': 1e-6,
        'optimizer': 'adam' # or 'lbfgs'
    }
    # Set defaults first
    for key, value in default_gp_params.items():
        setattr(bo_trial_cfg.gp, key, value)
    # Override with specific settings if available
    if 'gp' in std_bo_params:
        log.info("Merging parameters from base_cfg.standard_bo_params.gp")
        bo_trial_cfg.gp = OmegaConf.merge(bo_trial_cfg.gp, std_bo_params.gp)

    # --- 3. Acquisition Sub-config (acquisition.*) ---
    bo_trial_cfg.acquisition = OmegaConf.create()
    default_acq_params = {
        'function': 'ei', # ei, ucb, pi
        'xi': 0.01,      # for ei, pi
        'kappa': 2.5,     # for ucb
        'opt_restarts': 10,
        'opt_samples': 512,
        'optimizer_options': {"batch_limit": 5, "maxiter": 200} # For optimize_acqf
    }
    # Set defaults first
    for key, value in default_acq_params.items():
         setattr(bo_trial_cfg.acquisition, key, value)
    # Override with specific settings if available
    if 'acquisition' in std_bo_params:
        log.info("Merging parameters from base_cfg.standard_bo_params.acquisition")
        bo_trial_cfg.acquisition = OmegaConf.merge(bo_trial_cfg.acquisition, std_bo_params.acquisition)

    log.info(f"  Trial Seed: {bo_trial_cfg.bo.seed}")
    log.info(f"  Device: {bo_trial_cfg.bo.device}, Dtype: {bo_trial_cfg.bo.dtype}")
    log.info(f"  Initial Points: {bo_trial_cfg.bo.initial_points}, Max Iterations: {bo_trial_cfg.bo.max_iterations}")
    log.info(f"  GP Kernel: {bo_trial_cfg.gp.kernel}, Train Iter: {bo_trial_cfg.gp.train_iter}")
    log.info(f"  Acquisition Func: {bo_trial_cfg.acquisition.function.upper()}")
    log.info("----------------------------------------------------------")

    return bo_trial_cfg


def run_single_bo_trial(bo_cfg_trial: DictConfig, objective_function: Any, trial_idx: int) -> Optional[Dict]:
    """Initializes and runs Standard BO for one trial."""
    log.info(f"--- Starting Standard BO Trial {trial_idx + 1}/{bo_cfg_trial.bo.initial_points + bo_cfg_trial.bo.max_iterations} Iterations ---")
    result = None
    try:
        # Instantiate optimizer with the specifically constructed config for this trial
        optimizer = StandardBayesianOptimization(bo_cfg_trial, objective_function)
        result = optimizer.run_optimization() # Assumes run_optimization is implemented in base or here
        log.info(f"Trial {trial_idx + 1} Standard BO complete.")
    except Exception as e:
        log.error(f"An error occurred during optimization in Trial {trial_idx + 1}: {e}", exc_info=True)
        # Optionally print traceback: traceback.print_exc()
        return None # Return None to indicate failure
    return result # Should be a dict like {'best_x': ..., 'best_y': ..., 'all_x': ..., 'all_y': ..., 'total_time': ...}


def plot_and_save_trial_results(result: Optional[Dict], trial_idx: int, bo_cfg_trial: DictConfig, objective_function: Any, bounds_tensor: torch.Tensor, save_dir: str):
    """Plots optimization progress and optionally contour plot for a single trial."""
    if not result or 'all_y' not in result or result['all_y'] is None or 'all_x' not in result or result['all_x'] is None:
        log.warning(f"Skipping plotting/saving for Trial {trial_idx + 1} due to missing or invalid results.")
        return

    log.info(f"Processing results for Trial {trial_idx + 1}...")
    trial_save_path = os.path.join(save_dir, f"trial_{trial_idx}")
    os.makedirs(trial_save_path, exist_ok=True)

    all_y_np = result['all_y'].cpu().numpy() if isinstance(result['all_y'], torch.Tensor) else np.array(result['all_y'])
    all_x_np = result['all_x'].cpu().numpy() if isinstance(result['all_x'], torch.Tensor) else np.array(result['all_x'])
    best_y_val = result['best_y'].item() if isinstance(result['best_y'], torch.Tensor) else result['best_y']
    best_x_val = result['best_x'].cpu().numpy() if isinstance(result['best_x'], torch.Tensor) else result['best_x']


    # --- Plot optimization progress (Cumulative Best) ---
    try:
        plt.figure(figsize=(10, 6))
        # Since we are maximizing (negated objectives), higher is better.
        cumulative_best_y = np.maximum.accumulate(all_y_np)
        iterations = np.arange(len(all_y_np))
        plt.plot(iterations, cumulative_best_y, marker='.', linestyle='-', markersize=8, label=f'Best Found (Trial {trial_idx+1})')
        plt.xlabel('Iteration (incl. initial points)')
        plt.ylabel('Best Objective Value Found')
        plt.title(f'Standard BO Progress (Trial {trial_idx + 1}, Seed {bo_cfg_trial.bo.seed})')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(trial_save_path, f"optimization_progress_trial_{trial_idx}.png")
        plt.savefig(plot_path)
        log.info(f"Saved progress plot: {plot_path}")
        plt.close()
    except Exception as e:
        log.error(f"Failed to plot optimization progress for trial {trial_idx+1}: {e}")


    # --- Plot contour (2D only) ---
    if bo_cfg_trial.bo.input_dim == 2:
        try:
            plt.figure(figsize=(10, 8))
            b_min, b_max = bounds_tensor[0].tolist(), bounds_tensor[1].tolist() # Use tensor bounds
            x1_lin = np.linspace(b_min[0], b_max[0], 100)
            x2_lin = np.linspace(b_min[1], b_max[1], 100)
            X1, X2 = np.meshgrid(x1_lin, x2_lin)

            # Evaluate objective on the grid
            grid_points = torch.tensor(np.stack([X1.ravel(), X2.ravel()], axis=-1), dtype=DEFAULT_DTYPE, device=bo_cfg_trial.bo.device)
            with torch.no_grad():
                # Use the raw objective function without noise for visualization
                raw_objective = type(objective_function)(dim=bo_cfg_trial.bo.input_dim, bounds=bounds_tensor, negate=True, noise_std=0.0) # Assuming constructor signature
                Z_raw = raw_objective(grid_points)
                Z = Z_raw.cpu().reshape(X1.shape).numpy()

            plt.contourf(X1, X2, Z, 50, cmap='viridis')
            plt.colorbar(label='Objective Value')

            # Plot sampled points
            if all_x_np.shape[0] > 0:
                 plt.scatter(all_x_np[:, 0], all_x_np[:, 1], c='red', marker='x', s=50, label='Sampled Points')

            # Plot best point found
            if best_x_val is not None:
                plt.scatter(best_x_val[0], best_x_val[1], c='yellow', marker='*', s=200, label='Best Point Found', edgecolors='black')

            plt.xlabel('x1')
            plt.ylabel('x2')
            plt.xlim(b_min[0], b_max[0])
            plt.ylim(b_min[1], b_max[1])
            plt.title(f'Objective Contour & Samples (Trial {trial_idx + 1})')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            contour_path = os.path.join(trial_save_path, f"contour_plot_trial_{trial_idx}.png")
            plt.savefig(contour_path)
            log.info(f"Saved contour plot: {contour_path}")
            plt.close()
        except Exception as e:
             log.error(f"Failed to plot contour for trial {trial_idx+1}: {e}")


    # --- Save data ---
    try:
        data_path = os.path.join(trial_save_path, f"results_trial_{trial_idx}.npz")
        np.savez(data_path,
                 best_x=best_x_val,
                 best_y=best_y_val,
                 all_x=all_x_np,
                 all_y=all_y_np,
                 total_time=result.get('total_time'),
                 seed=bo_cfg_trial.bo.seed,
                 initial_points=bo_cfg_trial.bo.initial_points,
                 max_iterations=bo_cfg_trial.bo.max_iterations)
        log.info(f"Saved numerical data: {data_path}")
    except Exception as e:
        log.error(f"Error saving .npz data for trial {trial_idx + 1}: {e}")


def aggregate_and_plot_results(all_results: List[Optional[Dict]], cfg: DictConfig, save_dir: str) -> Optional[Dict]:
    """Aggregates results across trials and plots the mean cumulative best objective."""
    log.info("\n--- Aggregating Results Across All Trials ---")
    valid_results = [res for res in all_results if res and 'all_y' in res and res['all_y'] is not None and len(res['all_y']) > 0]
    n_trials = len(all_results)
    n_valid_trials = len(valid_results)

    if n_valid_trials == 0:
        log.warning("No valid trial results available for aggregation.")
        return None

    num_expected_iterations = cfg.experiment_params.initial_points + cfg.experiment_params.max_iterations
    all_trials_all_y = [res['all_y'].cpu().numpy() if isinstance(res['all_y'], torch.Tensor) else np.array(res['all_y']) for res in valid_results]

    # Pad/truncate sequences to ensure they have the same length
    all_trials_all_y_processed = []
    for i, y_list in enumerate(all_trials_all_y):
        current_len = len(y_list)
        if current_len == 0: continue # Skip empty arrays

        if current_len > num_expected_iterations:
            all_trials_all_y_processed.append(y_list[:num_expected_iterations])
        elif current_len < num_expected_iterations:
            padding_val = y_list[-1] # Pad with the last observed value
            padded_seq = np.pad(y_list, (0, num_expected_iterations - current_len), 'constant', constant_values=padding_val)
            all_trials_all_y_processed.append(padded_seq)
        else:
            all_trials_all_y_processed.append(y_list)

    if not all_trials_all_y_processed:
        log.warning("No processable 'all_y' data after padding/truncation.")
        return None

    try:
        # Calculate cumulative best for each trial (assuming maximization)
        all_y_array = np.array(all_trials_all_y_processed)
        cumulative_best_y_array = np.maximum.accumulate(all_y_array, axis=1)

        # Calculate mean and standard error across trials, ignoring NaNs
        mean_cumulative_best_y = np.nanmean(cumulative_best_y_array, axis=0)
        std_cumulative_best_y = np.nanstd(cumulative_best_y_array, axis=0)
        # Calculate valid counts per iteration step carefully
        valid_counts_per_step = np.sum(~np.isnan(cumulative_best_y_array), axis=0)
        valid_counts_per_step[valid_counts_per_step == 0] = 1 # Avoid division by zero
        stderr_cumulative_best_y = std_cumulative_best_y / np.sqrt(valid_counts_per_step)
        iterations = np.arange(len(mean_cumulative_best_y))

        # --- Plot Aggregate Results ---
        plt.figure(figsize=(12, 7))
        plt.plot(iterations, mean_cumulative_best_y, label=f'Mean Best Objective ({n_valid_trials} trials)', color='blue', marker='.', markersize=5)
        plt.fill_between(iterations, mean_cumulative_best_y - stderr_cumulative_best_y, mean_cumulative_best_y + stderr_cumulative_best_y, color='blue', alpha=0.2, label='Mean +/- 1 Std Error')
        plt.xlabel('Iteration (incl. initial points)')
        plt.ylabel('Mean Best Objective Value Found')
        objective_name = cfg.test_function.name
        plt.title(f'Aggregate Standard BO Progress ({objective_name}, {n_valid_trials}/{n_trials} Valid Trials)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        aggregate_plot_path = os.path.join(save_dir, "aggregate_bo_progress.png")
        plt.savefig(aggregate_plot_path)
        log.info(f"Saved aggregate plot: {aggregate_plot_path}")
        plt.close()

        # --- Save Aggregate Data ---
        # Extract final best 'y' from each valid trial result dictionary
        all_trials_final_best_y = []
        for res in valid_results:
            best_y = res.get('best_y')
            if isinstance(best_y, torch.Tensor):
                all_trials_final_best_y.append(best_y.item())
            elif best_y is not None:
                 all_trials_final_best_y.append(float(best_y))
            else:
                 all_trials_final_best_y.append(np.nan) # Use NaN if missing

        aggregate_save_path = os.path.join(save_dir, "aggregate_bo_results.npz")
        np.savez(aggregate_save_path,
                 mean_cumulative_best_y=mean_cumulative_best_y,
                 std_cumulative_best_y=std_cumulative_best_y,
                 stderr_cumulative_best_y=stderr_cumulative_best_y,
                 all_trials_final_best_y=np.array(all_trials_final_best_y), # Final best y from each trial
                 iterations=iterations,
                 num_valid_trials=n_valid_trials,
                 num_total_trials=n_trials)
        log.info(f"Saved aggregate results data: {aggregate_save_path}")

        return {
            "mean_cumulative_best_y": mean_cumulative_best_y,
            "stderr_cumulative_best_y": stderr_cumulative_best_y,
            "all_trials_final_best_y": all_trials_final_best_y,
            "iterations": iterations
        }
    except Exception as e:
        log.error(f"Error during aggregation or plotting: {e}", exc_info=True)
        return None


def summarize_trial_results(result: Optional[Dict], trial_idx: int):
    """Prints a brief summary of a single trial's outcome."""
    log.info(f"\n--- Trial {trial_idx + 1} Summary ---")
    best_x_str = "N/A"
    best_y_str = "N/A"
    time_str = ""

    if result and result.get('best_y') is not None:
        best_y_val = result['best_y'].item() if isinstance(result['best_y'], torch.Tensor) else result['best_y']
        best_y_str = f"{best_y_val:.6f}"
        if result.get('best_x') is not None:
            best_x_val = result['best_x'].cpu().numpy() if isinstance(result['best_x'], torch.Tensor) else result['best_x']
            best_x_str = np.array2string(np.array(best_x_val), precision=4, suppress_small=True)
        if 'total_time' in result and result['total_time'] is not None:
            time_str = f" Trial optimization time: {result['total_time']:.2f} seconds"
    else:
        log.warning("No valid result obtained for this trial.")

    log.info(f"Best solution found (X): {best_x_str}")
    log.info(f"Best objective value (Y): {best_y_str}")
    log.info(time_str)
    log.info("-" * (23 + len(str(trial_idx+1))))


def summarize_overall_results(all_results: List[Optional[Dict]], save_dir: str):
    """Prints a summary of the best result found across all successful trials."""
    log.info("\n--- Overall Experiment Summary ---")
    valid_results = [res for res in all_results if res and res.get('best_y') is not None]

    if not valid_results:
        log.warning("No valid results found across any trials.")
        log.info(f"Results and logs saved in: {save_dir}")
        return

    best_trial_idx = -1
    overall_best_y = -np.inf # Initialize for maximization
    overall_best_x = None
    # Keep track of original trial index for reporting
    original_indices = [i for i, res in enumerate(all_results) if res and res.get('best_y') is not None]

    for i, res in enumerate(valid_results):
        current_best_y = res['best_y'].item() if isinstance(res['best_y'], torch.Tensor) else res['best_y']
        if current_best_y > overall_best_y:
            overall_best_y = current_best_y
            overall_best_x = res['best_x'].cpu().numpy() if isinstance(res['best_x'], torch.Tensor) else res['best_x']
            best_trial_idx = original_indices[i] # Get the original index

    log.info(f"Total trials attempted: {len(all_results)}")
    log.info(f"Total successful trials: {len(valid_results)}")

    if best_trial_idx != -1 and overall_best_x is not None:
        best_x_str = np.array2string(np.array(overall_best_x), precision=4, suppress_small=True)
        log.info(f"Overall best result found in Trial: {best_trial_idx + 1}")
        log.info(f"Overall best solution (X): {best_x_str}")
        log.info(f"Overall best objective value (Y): {overall_best_y:.6f}")
    else:
        log.warning("Could not determine overall best result.")

    # Final summary of aggregated results (mean/std of final best values)
    final_best_y_values = []
    for res in valid_results:
        best_y = res.get('best_y')
        final_best_y_values.append(best_y.item() if isinstance(best_y, torch.Tensor) else float(best_y))

    if final_best_y_values:
         mean_final_best = np.nanmean(final_best_y_values)
         std_final_best = np.nanstd(final_best_y_values)
         log.info(f"Mean Final Best Objective Value across {len(final_best_y_values)} trials: {mean_final_best:.6f}")
         log.info(f"Std Dev Final Best Objective Value across {len(final_best_y_values)} trials: {std_final_best:.6f}")


    log.info(f"\nAll results, plots, and logs saved in: {save_dir}")
    log.info("----------------------------------")


# ==================================
# Main Function
# ==================================

@hydra.main(config_path='../../config', config_name="experiment", version_base=None)
def main(cfg: DictConfig):
    # --- Configuration Setup (Common for all trials) ---
    input_dim = cfg.test_function.input_dim
    domain_min = cfg.test_function.domain_min
    domain_max = cfg.test_function.domain_max
    if isinstance(domain_min, (int, float)):
        domain_min = [float(domain_min)] * input_dim
        domain_max = [float(domain_max)] * input_dim
    else:
        domain_min = [float(b) for b in domain_min]
        domain_max = [float(b) for b in domain_max]
    bounds_list = list(zip(domain_min, domain_max))

    if cfg.test_function.name == "Ackley":
        objective_function = Ackley(dim=input_dim, bounds=bounds_list, negate=True)
    elif cfg.test_function.name == "Rosenbrock":
        objective_function = Rosenbrock(dim=input_dim, bounds=bounds_list, negate=True)
    elif cfg.test_function.name == "Levy":
        objective_function = Levy(dim=input_dim, bounds=bounds_list, negate=True)
    else:
        raise ValueError(f"Unsupported objective function specified: {cfg.experiment.objective}") # Corrected cfg path

    # Override the cfg.method.bo parameters
    cfg.method.bo.domain_min = domain_min # Use processed list
    cfg.method.bo.domain_max = domain_max # Use processed list
    cfg.method.bo.input_dim = input_dim
    cfg.method.bo.initial_points = cfg.experiment_params.initial_points
    cfg.method.bo.objective = cfg.experiment_params.objective 
    cfg.method.bo.max_iterations = cfg.experiment_params.max_iterations


    print("--- Experiment Configuration ---")
    print(OmegaConf.to_yaml(cfg))
    print("-----------------------------")

    # make sure the method name is consistent
    if cfg.method.name == "standard_bo":
        Optimizer = StandardBayesianOptimization
    elif cfg.method.name == "turbo":
        Optimizer = TrustRegionBayesianOptimization
    elif cfg.method.name == "pfns4bo":
        Optimizer = PFNS4BayesianOptimization
    else:
        raise ValueError(f"Unsupported method specified: {cfg.method.name}")
    save_prefix = f"{cfg.method.name}_{cfg.test_function.name}"

    n_trials = cfg.experiment_params.num_trials
    all_trials_results = []
    all_trials_all_y = []
    all_trials_best_y = []

    if n_trials <= 0:
        print("Warning: cfg.experiment_params.n_trials is <= 0. No trials will be run.")
        return None # Or raise error

    # --- Run Multiple Trials ---
    iterator = tqdm(range(n_trials), desc="Trials", unit="trial")
    for trial_idx in iterator:
        # Set up the optimization problem for this trial
        # Re-instantiate to ensure independent runs if the optimizer has state
        cfg.method.seed = cfg.experiment_params.seed_start + trial_idx
        opt = Optimizer(cfg.method, objective_function)

        # Run optimization
        result = opt.run_optimization()
        all_trials_results.append(result)
        all_trials_all_y.append(result['all_y'])
        all_trials_best_y.append(result['best_y'])

        print(f"\nTrial {trial_idx + 1} Complete!")
        print(f"Best solution found: {result['best_x']}")
        print(f"Best objective value: {result['best_y']}")

        # --- Save Individual Trial Results ---
        trial_save_dir = cfg.save_dir if hasattr(cfg, 'save_dir') else "."
        os.makedirs(trial_save_dir, exist_ok=True) # Ensure directory exists

        # Save individual optimization progress plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(result['all_y'])), result['all_y'], marker='o', linestyle='-') # Use plot for clarity
        plt.xlabel('Iteration (incl. initial points)')
        plt.ylabel('Objective Value')
        plt.title(f'Optimization Progress (Trial {trial_idx + 1})')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(trial_save_dir, f"{save_prefix}_optimization_progress_trial_{trial_idx}.png"))
        plt.close() # Close plot to free memory

        # Save individual contour plot (if applicable)
        if cfg.method.bo.input_dim == 2:
            plt.figure(figsize=(10, 8))
            # Create grid for contour
            x1_lin = np.linspace(bounds_list[0][0], bounds_list[0][1], 100)
            x2_lin = np.linspace(bounds_list[1][0], bounds_list[1][1], 100)
            X1, X2 = np.meshgrid(x1_lin, x2_lin)
            Z = np.zeros_like(X1)

            # Evaluate objective on grid
            grid_points = torch.tensor(np.stack([X1.ravel(), X2.ravel()], axis=-1), dtype=torch.float32)
            with torch.no_grad(): # Ensure no gradient tracking for plotting
                 Z = objective_function(grid_points).reshape(X1.shape).numpy()


            plt.contourf(X1, X2, Z, 50, cmap='viridis')
            plt.colorbar(label='Objective Value')

            # Plot sampled points for this trial
            all_x_trial = result['all_x']
            plt.scatter(
                all_x_trial[:, 0],
                all_x_trial[:, 1],
                c='red',
                marker='x',
                s=50,
                label='Sampled Points'
            )

            # Highlight best point for this trial
            best_x_trial = result['best_x'].reshape([2, -1])
            plt.scatter(
                best_x_trial[0],
                best_x_trial[1],
                c='yellow',
                marker='*',
                s=200,
                label='Best Point Found',
                edgecolors='black'
            )

            plt.xlabel('x1')
            plt.ylabel('x2')
            plt.xlim(bounds_list[0])
            plt.ylim(bounds_list[1])
            plt.title(f'Objective Contour with Sampled Points (Trial {trial_idx + 1})')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(trial_save_dir, f"{save_prefix}_contour_plot_trial_{trial_idx}.png"))
            plt.close() # Close plot

        # Save individual trial data
        np.savez(
            os.path.join(trial_save_dir, f"{save_prefix}_results_trial_{trial_idx}.npz"),
            best_x=result['best_x'],
            best_y=result['best_y'],
            all_x=result['all_x'],
            all_y=result['all_y']
        )

    # --- Aggregate and Plot Results ---
    print("\n--- Aggregating Results Across Trials ---")

    # Ensure all 'all_y' arrays have the same length (they should if max_iterations is fixed)
    # If lengths can vary, padding or truncation might be needed before stacking
    max_len = max(len(y) for y in all_trials_all_y)
    # Pad shorter sequences if necessary (e.g., with last value or NaN)
    # For simplicity here, assume fixed length based on initial_points + max_iterations
    num_iterations = cfg.method.bo.initial_points + cfg.method.bo.max_iterations
    if any(len(y) != num_iterations for y in all_trials_all_y):
         print(f"Warning: Not all trials completed {num_iterations} evaluations. Aggregation might be inaccurate.")
         # Simple approach: use the minimum length across trials for aggregation
         min_len = min(len(y) for y in all_trials_all_y)
         all_trials_all_y_padded = np.array([y[:min_len] for y in all_trials_all_y])
    else:
         all_trials_all_y_padded = np.array(all_trials_all_y)


    # Calculate mean and standard deviation (or standard error)
    mean_all_y = np.mean(all_trials_all_y_padded, axis=0)
    std_all_y = np.std(all_trials_all_y_padded, axis=0)
    # Optional: Calculate standard error: std_all_y / np.sqrt(n_trials)
    stderr_all_y = std_all_y / np.sqrt(n_trials)

    iterations = np.arange(len(mean_all_y)) # X-axis for the plot

    # Plot Aggregate Results (Mean +/- Std Dev or Std Error)
    plt.figure(figsize=(12, 7))
    plt.plot(iterations, mean_all_y, label='Mean Objective Value', color='blue')
    plt.fill_between(
        iterations,
        mean_all_y - std_all_y, # Or use stderr_all_y
        mean_all_y + std_all_y, # Or use stderr_all_y
        color='blue',
        alpha=0.2,
        label='Mean +/- 1 Std Dev' # Adjust label if using stderr
    )
    plt.xlabel('Iteration (incl. initial points)')
    plt.ylabel('Objective Value')
    plt.title(f'Aggregate Optimization Progress ({n_trials} Trials)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    aggregate_plot_path = os.path.join(trial_save_dir, f"{save_prefix}_aggregate_optimization_progress.png")
    plt.savefig(aggregate_plot_path)
    print(f"Saved aggregate plot to: {aggregate_plot_path}")
    plt.close()

    # --- Save Aggregate Results ---
    aggregate_save_path = os.path.join(trial_save_dir, f"{save_prefix}_aggregate_results.npz")
    np.savez(
        aggregate_save_path,
        mean_all_y=mean_all_y,
        std_all_y=std_all_y,
        stderr_all_y = stderr_all_y,
        all_trials_best_y=np.array(all_trials_best_y), # Store best y from each trial
        iterations = iterations
        # Optionally save all_trials_all_y_padded if needed, but it can be large
    )
    print(f"Saved aggregate results to: {aggregate_save_path}")

    # Find the overall best result across all trials
    overall_best_trial_idx = np.argmax(all_trials_best_y)
    overall_best_y = all_trials_best_y[overall_best_trial_idx]
    overall_best_x = all_trials_results[overall_best_trial_idx]['best_x']

    print("\n--- Overall Best Result Across All Trials ---")
    print(f"Found in Trial: {overall_best_trial_idx + 1}")
    print(f"Best solution (X): {overall_best_x}")
    print(f"Best objective value (Y): {overall_best_y}")
    print("---------------------------------------------")


    return all_trials_results # Return list of dictionaries containing results for each trial

if __name__ == "__main__":
    main()