import torch
import torch.optim as optim
import numpy as np
import gpytorch
import hydra
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from tqdm import tqdm

import warnings
import sys
import os
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.model import DecisionTransformer, ExactGPModel
from src.model.gp_ensemble import build_gp_ensemble
from src.model.deep_kernel_gp import DeepKernelGP as _DeepKernelGP
from src.objectives import Ackley, Rosenbrock, Levy
from src.policy.base import BaseBayesianOptimizer

from botorch.models import SingleTaskGP # BoTorch standard GP model
from botorch.exceptions import InputDataWarning
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel # Standard GPyTorch kernels
from gpytorch.constraints import GreaterThan
from gpytorch.mlls import ExactMarginalLogLikelihood # Often used for training
from botorch.acquisition import (
    ExpectedImprovement,
    UpperConfidenceBound,
    ProbabilityOfImprovement,
    qMaxValueEntropy,
)
from botorch.acquisition.analytic import LogExpectedImprovement

DEFAULT_DTYPE = torch.float64

class DirectRegretOptimization(BaseBayesianOptimizer):
    """
    Direct Regret Optimization using GP ensembles and Decision Transformer,
    inheriting from BaseBayesianOptimizer.
    """
    def __init__(self, config: DictConfig, objective_function):
        # Call parent constructor first
        super().__init__(config, objective_function)

        # --- Specific Initializations for Direct Regret ---

        # Configs specific to this method (assuming nested structure in main config)
        self.gp_config = config.gp # Expects a 'gp' sub-config
        self.transformer_config = config.transformer # Expects a 'transformer' sub-config
        self.simulation_config = config.simulation # Expects a 'simulation' sub-config
        self.acquisition_config = config.acquisition # Expects an 'acquisition' sub-config

        # Initialize GP Ensemble (models will be created but empty)
        self.gp_ensemble = [] # List of dicts: {'model': ..., 'likelihood': ...}
        # Note: _initialize_models will populate this based on data later

        # Initialize Decision Transformer
        # State dim needs careful calculation based on actual GP params + extras
        state_dim = self._get_state_dim()
        action_dim = self.bo_config.input_dim
        self.decision_transformer = DecisionTransformer(
            self.transformer_config, state_dim, action_dim
        ).to(self.device, self.dtype)

        # Initialize Optimizer for the Transformer
        self.optimizer = optim.Adam(
            self.decision_transformer.parameters(),
            lr=self.transformer_config.lr,
            weight_decay=getattr(self.transformer_config, 'weight_decay', 0.0) # Optional weight decay
        )

        # Override initial sampling method if specified in config
        self.initial_sampling_method = getattr(self.bo_config, 'initial_sampling_method', 'lhs') # Default to LHS for this class

        # --- Fidelity Audit (RQ2) ---
        # trajectory_logs      — Setup A: GP-chosen query points, true f(x) called as counterfactual.
        #                        Valid for per-step metrics (rtg_rmse, rtg_bias, per_step_*).
        # true_trajectory_logs — Setup B: independent true BO; true_observed_best drives the
        #                        acquisition, so query points diverge from the simulated path.
        #                        Valid for Wasserstein distance and DT-GT training.
        self.trajectory_logs: list       = []
        self.true_trajectory_logs: list  = []
        self._bo_iteration_count: int    = 0
        self.reward_diagnostics: list    = []  # per-BO-iter reward stats for both modes
        self._running_max_rtg: float     = 0.0  # historical max for floored_dynamic RTG mode
        self.dkl_diagnostics: list       = []  # per-BO-iter deep-kernel diagnostics


    def sample_initial_points(self, method=None):
        """ Overrides base sampler to use the method preferred by this class (e.g., LHS)"""
        if method is None:
             method = self.initial_sampling_method

        if method.lower() == 'lhs':
             self._sample_initial_points_lhs()
        else:
             # Fallback to parent implementation for 'sobol', 'random', etc.
             super().sample_initial_points(method=method)


    def _sample_initial_points_lhs(self):
        """ Sample initial points using Latin Hypercube Sampling (basic implementation) """
        n_initial = getattr(self.bo_config, 'initial_points', 5)
        dim = self.bo_config.input_dim

        if n_initial <= 0: return
        if self.data_x.shape[0] > 0: return # Already have data

        print(f"Sampling {n_initial} initial points using Latin Hypercube Sampling...")

        # Basic LHS: grid + shuffle permutation per dimension
        points = torch.zeros((n_initial, dim), device=self.device, dtype=self.dtype)
        for i in range(dim):
            # Create evenly spaced points in [0, 1] range first
            grid_coords = torch.linspace(0.0, 1.0, n_initial + 1)[:-1] + 0.5 / n_initial
            perm = torch.randperm(n_initial, device=self.device)
            points[:, i] = grid_coords[perm] # Apply permutation

        # Scale points to the actual domain
        domain_min = self.bounds[0]
        domain_max = self.bounds[1]
        initial_x = domain_min + (domain_max - domain_min) * points

        # Evaluate points
        initial_y = torch.zeros(n_initial, device=self.device, dtype=self.dtype)
        for i in range(n_initial):
            y_val = self.objective_function(initial_x[i])
            if isinstance(y_val, torch.Tensor):
                initial_y[i] = y_val
            else:
                initial_y[i] = torch.tensor(y_val, device=self.device, dtype=self.dtype)

        # Update data and best observation
        for i in range(n_initial):
             self._update_data_and_best(initial_x[i].unsqueeze(0), initial_y[i].item())

        if self.verbose:
            best_val, _ = self.get_best()
            print(f"Completed LHS initial sampling. Initial best value: {best_val:.4f}")


    def _get_state_dim(self):
        """Calculate the dimension of the state vector for the Decision Transformer.

        When simulation.state_include_gp_params is False the GP hyperparameter
        block is omitted and the state is purely data-driven:
            [best_value (1), step_norm (1), best_position (input_dim)]
        This removes the training/deployment inconsistency that arises in DT-GT
        because rollout-refitted GP hyperparameters differ from the deployment GP.
        """
        include_gp = getattr(self.simulation_config, 'state_include_gp_params', True)

        if not include_gp:
            state_dim = 2 + self.bo_config.input_dim  # best_value + step_norm + best_position
            if getattr(self.gp_config, 'verbose', False):
                print(f"State dim (no GP params): {state_dim}")
            return state_dim

        gp_param_dim = 0
        try:
             dummy_likelihood = gpytorch.likelihoods.GaussianLikelihood()
             dummy_model = ExactGPModel(torch.zeros(1, self.bo_config.input_dim), torch.zeros(1), dummy_likelihood)
             gp_param_dim = sum(p.numel() for p in dummy_model.hyperparameters())
             if getattr(self.gp_config, 'verbose', False):
                 print(f"Detected {gp_param_dim} hyperparameters per GP model.")
        except Exception:
             print("Warning: Could not automatically determine GP hyperparameter count. Assuming 2.")
             gp_param_dim = 2

        num_models = self.gp_config.num_models
        state_dim = num_models * gp_param_dim + 2 + self.bo_config.input_dim
        if getattr(self.gp_config, 'verbose', False):
            print(f"Calculated state dimension: {state_dim}")
        return state_dim


    # ── Deep-kernel diagnostics helper ────────────────────────────────────────

    def _store_dkl_diagnostics(self, train_x, train_y, anchor_loss, warm_start_used: bool):
        """Collect per-iteration diagnostics from the deep-kernel ensemble and append
        to self.dkl_diagnostics.

        Logged fields
        -------------
        dkl_train_loss    : final MLL loss of the anchor / model-0 fit.
        dkl_latent_norm   : mean L2 norm of projected training points (~1.0 by construction).
        dkl_lengthscales  : list of M lists — one ARD lengthscale vector per model.
        dkl_warm_start_used : True for every call from _update_models (False for init).
        """
        if not self.gp_ensemble:
            return
        model0 = self.gp_ensemble[0]["model"]

        # Latent norm
        with torch.no_grad():
            model0.eval()
            projected = model0.feature_extractor(train_x)
            latent_norm = projected.norm(dim=-1).mean().item()

        # Per-model lengthscales (ARD vector → list)
        lengthscales = []
        for gp in self.gp_ensemble:
            with torch.no_grad():
                ls = gp["model"].covar_module.base_kernel.lengthscale \
                         .detach().cpu().squeeze().tolist()
            lengthscales.append(ls if isinstance(ls, list) else [ls])

        self.dkl_diagnostics.append({
            "bo_iteration":       len(self.dkl_diagnostics),
            "dkl_train_loss":     float(anchor_loss) if anchor_loss is not None else None,
            "dkl_latent_norm":    latent_norm,
            "dkl_lengthscales":   lengthscales,
            "dkl_warm_start_used": warm_start_used,
        })

    def _initialize_models(self):
        """Initialize the ensemble of GP models using BoTorch SingleTaskGP."""
        print("Initializing BoTorch GP ensemble...")
        self.gp_ensemble = [] # Clear any previous models
        if self.data_x is None or self.data_y is None:
            # BoTorch models require training data at initialization
            print(
                "WARN: Training data (self.data_x, self.data_y) is None. "
                "Cannot initialize models. Ensure data is loaded or passed."
            )
            return
        # --- 1. Prepare Data ---
        train_x = self.data_x.to(device=self.device, dtype=self.dtype)
        # SingleTaskGP expects train_Y to be shape [n] or [n, 1].
        # Squeeze last dim if it's [n, 1] to get [n] which is often preferred.
        train_y = self.data_y.to(device=self.device, dtype=self.dtype).squeeze(-1)
        if train_y.ndim != 1:
             raise ValueError(f"train_y must be 1-dimensional, but got shape {train_y.shape}")
        if train_x.ndim != 2:
             raise ValueError(f"train_x must be 2-dimensional (n x d), but got shape {train_x.shape}")
        if train_x.shape[0] != train_y.shape[0]:
             raise ValueError(f"Number of samples mismatch: X={train_x.shape[0]}, Y={train_y.shape[0]}")

        # --- 2. Get Configuration ---
        min_scale = self.gp_config.lengthscale_min
        max_scale = self.gp_config.lengthscale_max
        num_models = self.gp_config.num_models
        # Generate the initial lengthscales to try for each model
        initial_lengthscales = np.linspace(min_scale, max_scale, num_models)
        kernel_type = getattr(self.gp_config, 'kernel', 'rbf').lower()
        noise_constraint_val = getattr(self.gp_config, 'noise_constraint', 1e-6)
        use_ard = getattr(self.gp_config, 'ard', False) # Check for Automatic Relevance Determination
        input_dim = train_x.shape[-1]

        # --- 2b. Deep-kernel shortcut (build_gp_ensemble handles full training) ---
        ensemble_kernel_type = getattr(self.gp_config, 'kernel_type', 'rbf_isotropic')
        if ensemble_kernel_type == 'deep_kernel':
            _dkl_result = build_gp_ensemble(
                train_x, train_y, M=num_models, kernel_type='deep_kernel',
                device=self.device, dtype=self.dtype,
                dkl_hidden_dim=getattr(self.gp_config, 'dkl_hidden_dim', 32),
                dkl_latent_dim=getattr(self.gp_config, 'dkl_latent_dim', None),
                dkl_n_iter=getattr(self.gp_config, 'dkl_n_iter', 100),
                dkl_lr=getattr(self.gp_config, 'dkl_lr', 0.01),
                dkl_weight_decay=getattr(self.gp_config, 'dkl_weight_decay', 1e-2),
                collect_diagnostics=True,
            )
            self.gp_ensemble, _dkl_diag = _dkl_result
            self._store_dkl_diagnostics(train_x, train_y,
                                        _dkl_diag.get("anchor_final_loss"),
                                        warm_start_used=False)
            if self.verbose:
                print(f"Initialized deep-kernel ensemble ({num_models} models).")
            return  # ensemble already trained; skip SingleTaskGP loop + _update_models

        # --- 3. Loop to Create Models ---
        for i, initial_ls in enumerate(initial_lengthscales):
            if self.verbose:print(f"Initializing model {i} with initial lengthscale: {initial_ls:.4f}")
            # a) Create Likelihood
            likelihood = GaussianLikelihood(
                noise_constraint=GreaterThan(noise_constraint_val)
            )
            # b) Create Kernel
            ard_num_dims = input_dim if use_ard else None
            if kernel_type == 'rbf':
                base_kernel = RBFKernel(ard_num_dims=ard_num_dims)
            elif kernel_type == 'matern':
                # Default to nu=2.5 for Matern, make configurable if needed
                matern_nu = getattr(self.gp_config, 'matern_nu', 2.5)
                base_kernel = MaternKernel(nu=matern_nu, ard_num_dims=ard_num_dims)
            # Add elif clauses for other kernels ('cosine', 'linear', etc.)
            else:
                raise ValueError(f"Unsupported kernel type: {kernel_type}")
            covar_module = ScaleKernel(base_kernel)
            # c) Set Initial Lengthscale on the Kernel Parameter
            covar_module.base_kernel.initialize(lengthscale=float(initial_ls))
            # d) Instantiate BoTorch SingleTaskGP Model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=InputDataWarning)
                # Note: SingleTaskGP expects train_Y to be 2D (n x 1) for GPyTorch
                model = SingleTaskGP(
                    train_X=train_x,
                    train_Y=train_y.reshape(-1, 1), # Ensure correct shape for SingleTaskGP
                    likelihood=likelihood,
                    covar_module=covar_module,
                )
                model = model.to(device=self.device, dtype=self.dtype)
            # e) Store Model and Likelihood
            self.gp_ensemble.append({
                'model': model,
                'likelihood': model.likelihood, # Access likelihood directly from model
                'id': i
            })
        if self.verbose: print(f"Successfully initialized {len(self.gp_ensemble)} BoTorch SingleTaskGP models.")
        # --- 4. Perform Initial Training/Fitting ---
        if self.verbose: print("Proceeding to initial model fitting (_update_models)...")
        self._update_models()


    def _update_models(self):
        """Update all GP models in the ensemble with current data and retrain."""
        # ── Deep-kernel warm-start rebuild ────────────────────────────────────
        ensemble_kernel_type = getattr(self.gp_config, 'kernel_type', 'rbf_isotropic')
        if ensemble_kernel_type == 'deep_kernel':
            if not self.gp_ensemble:
                # Defensive: should not happen, but fall back to fresh build
                self._initialize_models()
                return
            train_x = self.data_x.to(device=self.device, dtype=self.dtype)
            train_y = self.data_y.to(device=self.device, dtype=self.dtype).squeeze(-1)
            warm_start = [(gp['model'], gp['likelihood']) for gp in self.gp_ensemble]
            _dkl_result = build_gp_ensemble(
                train_x, train_y, M=self.gp_config.num_models,
                kernel_type='deep_kernel',
                device=self.device, dtype=self.dtype,
                dkl_hidden_dim=getattr(self.gp_config, 'dkl_hidden_dim', 32),
                dkl_latent_dim=getattr(self.gp_config, 'dkl_latent_dim', None),
                dkl_n_iter=getattr(self.gp_config, 'dkl_n_iter', 100),
                dkl_lr=getattr(self.gp_config, 'dkl_lr', 0.01),
                dkl_weight_decay=getattr(self.gp_config, 'dkl_weight_decay', 1e-2),
                dkl_warm_start_models=warm_start,
                collect_diagnostics=True,
            )
            self.gp_ensemble, _dkl_diag = _dkl_result
            self._store_dkl_diagnostics(train_x, train_y,
                                        _dkl_diag.get("anchor_final_loss"),
                                        warm_start_used=True)
            if self.verbose:
                print(f"Updated deep-kernel ensemble with warm-start ({len(warm_start)} models.")
            return

        retrain= getattr(self.gp_config, 'retrain', False) # Default to True
        if retrain:
            #  Update all GP models with current data
            for gp in self.gp_ensemble:
                # Update model with current data
                gp['model'].set_train_data(self.data_x, self.data_y, strict=False)
                
                # Train the model
                gp['model'].train()
                gp['likelihood'].train()
                
                # Use exact marginal log likelihood for optimization
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp['likelihood'], gp['model'])
                
                # Use L-BFGS for optimization
                optimizer = torch.optim.LBFGS(
                    gp['model'].parameters(),
                    lr=0.1,
                    max_iter=20,
                    line_search_fn="strong_wolfe"
                )
                
                def closure():
                    optimizer.zero_grad()
                    output = gp['model'](self.data_x)
                    loss = -mll(output, self.data_y)
                    loss.backward()
                    return loss
                
                optimizer.step(closure)
                
                # Set to eval mode
                gp['model'].eval()
                gp['likelihood'].eval()
        else:
            if not self.gp_ensemble:
                print("Warning: GP ensemble not initialized. Cannot update.")
                # Try to initialize if data exists? Or rely on _initialize_models being called first.
                if self.data_x.shape[0] > 0:
                    print("Attempting to initialize GP ensemble now...")
                    self._initialize_models() # This might recursive call _update_models, be careful
                    if not self.gp_ensemble: # If still failed
                        raise RuntimeError("Failed to initialize GP models.")
                else:
                    return # Cannot update without data

        if self.verbose: print(f"Updating {len(self.gp_ensemble)} GP models...")

        for gp_dict in self.gp_ensemble:
            model = gp_dict['model']
            likelihood = gp_dict['likelihood']

            # Update model with current data
            model.set_train_data(self.data_x, self.data_y, strict=False)

            # Train the model
            model.train()
            likelihood.train()

            # Use Exact MLL for optimization
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            # Use Adam or LBFGS for optimization (Adam often more robust)
            gp_optimizer = torch.optim.Adam(model.parameters(), lr=getattr(self.gp_config,'lr', 0.1)) # Configurable LR
            max_iter = getattr(self.gp_config, 'train_iter', 50) # Configurable iterations

            for i in range(max_iter):
                gp_optimizer.zero_grad()
                output = model(self.data_x)
                loss = -mll(output, self.data_y)
                loss.backward()
                gp_optimizer.step()

            # Set to eval mode
            model.eval()
            likelihood.eval()
        if self.verbose: print("GP models updated and trained.")


    # --- Acquisition Function Logic (Internal Helper) ---
    def _acquisition_function_value(self, x: torch.Tensor, gp_idx: int, observed_best: float) -> torch.Tensor:
        """Compute acquisition function value for a given point x (single point tensor)"""
        if not self.gp_ensemble:
            raise RuntimeError("GP Ensemble not initialized.")
        if gp_idx >= len(self.gp_ensemble):
             raise IndexError(f"gp_idx {gp_idx} out of bounds for ensemble size {len(self.gp_ensemble)}")

        x = x.to(self.device, self.dtype) # Ensure correct device/dtype
        if x.ndim == 1: x = x.unsqueeze(0) # Ensure batch dimension [1, D]

        gp = self.gp_ensemble[gp_idx]
        model = gp['model']
        likelihood = gp['likelihood']

        model.eval()
        likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = likelihood(model(x)) # Get predictive distribution (includes noise)
            mean = posterior.mean
            stddev = posterior.variance.sqrt()

            # Clamp stddev to avoid numerical issues with near-zero variance
            stddev = stddev.clamp_min(1e-9)

            acq_name = getattr(self.acquisition_config, 'function', 'ei')
            xi = getattr(self.acquisition_config, 'xi', 0.01)
            kappa = getattr(self.acquisition_config, 'kappa', 2.5)

            if acq_name == "ei":
                imp = mean - observed_best - xi
                Z = imp / stddev
                normal = torch.distributions.Normal(0.0, 1.0)
                ei = imp * normal.cdf(Z) + stddev * normal.log_prob(Z).exp()
                return ei
            elif acq_name == "ucb":
                return mean + kappa * stddev
            elif acq_name == "pi":
                Z = (mean - observed_best - xi) / stddev
                return torch.distributions.Normal(0.0, 1.0).cdf(Z)
            else:
                raise ValueError(f"Unknown acquisition function: {acq_name}")

    def _acquisition_function_value_botorch(
            self, x: torch.Tensor, gp_idx: int, observed_best: float, acq_name: str = None
        ) -> torch.Tensor:
        """
        Compute acquisition function value using BoTorch for a given point x.

        Args:
            x: A single point tensor [D] or [1, D] representing the candidate point.
            gp_idx: The index of the Gaussian Process model in the ensemble to use.
            observed_best: The best objective function value observed so far.

        Returns:
            A tensor containing the acquisition function value for the point x.
        """
        if not hasattr(self, 'gp_ensemble') or not self.gp_ensemble:
            raise RuntimeError("GP Ensemble (`self.gp_ensemble`) not initialized.")
        if gp_idx >= len(self.gp_ensemble):
            raise IndexError(
                f"gp_idx {gp_idx} out of bounds for ensemble size {len(self.gp_ensemble)}"
            )
        if not hasattr(self, 'acquisition_config'):
             raise RuntimeError("Acquisition config (`self.acquisition_config`) not found.")

        # --- 1. Prepare Input and Model ---
        x = x.to(device=self.device, dtype=self.dtype) # Ensure correct device/dtype
        if x.ndim == 1:
            x = x.unsqueeze(0) # Shape: [1, D]

        # BoTorch analytic acquisition functions expect input shape (b x q x d) or (q x d).
        # We have a single point (q=1), so reshape to [1, 1, D]
        # where the first dimension is the batch dimension ('b') for the acquisition function.
        if x.shape[0] == 1:
            X = x.unsqueeze(0) # Shape: [1, 1, D]
        else:
            X = x.reshape(x.shape[0], 1, x.shape[1]) # Shape: [b, 1, D]

        # Get the specific GP model from the ensemble
        model = self.gp_ensemble[gp_idx]['model']

        # Ensure the model is in evaluation mode
        model.eval()
        # No need to explicitly call likelihood.eval() here, BoTorch handles it via the model.

        # --- 2. Get Acquisition Function Configuration ---
        # Default to 'ei' if not specified
        if acq_name is None:
            acq_name = getattr(self.acquisition_config, 'function', 'ei').lower()
        # Assuming maximization objective, change if minimizing
        maximize_objective = getattr(self.acquisition_config, 'maximize', True)

        # --- 3. Instantiate BoTorch Acquisition Function ---
        # qMaxValueEntropy (MES) relies on internal BoTorch model APIs not implemented
        # by DeepKernelGP; fall back to EI transparently.
        if acq_name == "mes" and isinstance(model, _DeepKernelGP):
            acq_name = "ei"

        if acq_name == "ei":
            # Note: BoTorch EI's 'best_f' corresponds to 'observed_best'.
            # The 'xi' parameter from the original code often acts as a trade-off.
            # In BoTorch, this can sometimes be incorporated by adjusting best_f.
            xi = getattr(self.acquisition_config, 'xi', 0.01)
            effective_best_f = observed_best + xi if maximize_objective else observed_best - xi
            acq_func = LogExpectedImprovement(
                model=model,
                best_f=effective_best_f,
                maximize=maximize_objective,
            )
        elif acq_name == "ucb":
            kappa = getattr(self.acquisition_config, 'kappa', 2.5)
            # BoTorch UCB uses 'beta' which often corresponds to 'kappa' or kappa^2
            # depending on the literature. Check BoTorch docs for the exact definition used.
            # As of recent versions, beta is typically the direct trade-off parameter (like kappa).
            acq_func = UpperConfidenceBound(
                model=model,
                beta=kappa, 
                maximize=maximize_objective,
                # objective=None
            )
        elif acq_name == "pi":
            xi = getattr(self.acquisition_config, 'xi', 0.01)
            effective_best_f = observed_best + xi if maximize_objective else observed_best - xi
            acq_func = ProbabilityOfImprovement(
                model=model,
                best_f=effective_best_f,
                maximize=maximize_objective,
                # objective=None
            )
        # --- Max-value Entropy Search (MES) ---
        elif acq_name == "mes":
            if self.verbose==2: print("INFO: Using the input evaluation points `x` as the `candidate_set` for MES.")

            # Get necessary MES parameters from config
            num_mv_samples = getattr(self.acquisition_config, 'mes_num_mv_samples', 10)
            # Note: mes_num_candidates config is ignored when x is used directly.

            # Ensure x is on the correct device/dtype (already done in Step 1)
            # The input x (shape [q, D]) IS the candidate set for MES.
            candidate_set_for_mes = x

            # Instantiate qMaxValueEntropy
            acq_func = qMaxValueEntropy(
                model=model,
                candidate_set=candidate_set_for_mes, # Pass x directly here
                num_mv_samples=num_mv_samples,
                maximize=maximize_objective,
                # use_posterior_mean=False, # Optional optimization
            )

        # --- GIBBON Placeholder ---
        elif acq_name == "gibbon":
            raise NotImplementedError(
                "GIBBON acquisition function requires custom implementation based on "
                "the specific algorithm/paper and is not a standard BoTorch class."
            )
        elif acq_name == "rotate_acq":
            # randomly rotate the acquisition function
            acq_names = ["ei", "ucb", "pi", 'mes']
            acq_name = np.random.choice(acq_names)
            return self._acquisition_function_value_botorch(
                x, gp_idx, observed_best, acq_name=acq_name,
            )
        else:
            raise ValueError(f"Unknown acquisition function: {acq_name}")

        # --- 4. Compute Acquisition Value ---
        acq_value = acq_func(X) # Input shape [1, 1, D] -> Output shape [1]

        # --- 5. Return Result ---
        # Return the tensor containing the value (original function returned a tensor)
        return acq_value # Shape [1]

    # --- Helper Function ---
    def _get_posterior_mean_stddev(self, x: torch.Tensor, gp_idx: int):
        """
        Helper function to get posterior mean and standard deviation for input x.

        Args:
            x (torch.Tensor): Input points (shape [b, D] or [D]).
            gp_idx (int): Index of the GP model in the ensemble.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Posterior mean and standard deviation.
        """
        if not self.gp_ensemble:
            raise RuntimeError("GP Ensemble not initialized.")
        if gp_idx >= len(self.gp_ensemble):
             raise IndexError(f"gp_idx {gp_idx} out of bounds for ensemble size {len(self.gp_ensemble)}")

        x = x.to(self.device, self.dtype) # Ensure correct device/dtype
        if x.ndim == 1: x = x.unsqueeze(0) # Ensure batch dimension [1, D]

        gp = self.gp_ensemble[gp_idx]
        model = gp['model']
        likelihood = gp['likelihood'] # Included for completeness, though BoTorch model handles it

        model.eval()
        likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Use model.posterior(x) for BoTorch models - gives GPyTorch posterior object
            posterior = model.posterior(x)
            mean = posterior.mean.squeeze(-1) # Squeeze trailing dimension [b, 1] -> [b]
            variance = posterior.variance.squeeze(-1) # Squeeze trailing dimension [b, 1] -> [b]
            # Clamp variance for numerical stability before sqrt
            stddev = variance.clamp_min(1e-9).sqrt()

        return mean, stddev

    # --- Optimize Acquisition Function (Internal Helper for Simulation) ---
    def _optimize_acquisition(self, gp_idx: int, observed_best: float) -> torch.Tensor:
        """
        Optimize acquisition function for a specific GP using random search + refinement,
        with an option to constrain the search space based on UCB >= max(LCB).
        """
        # --- Configurable Parameters ---
        num_samples = getattr(self.acquisition_config, 'opt_samples', 1000)
        num_restarts = getattr(self.acquisition_config, 'opt_restarts', 10)
        refinement_samples = getattr(self.acquisition_config, 'refinement_samples', 100)
        refinement_noise = getattr(self.acquisition_config, 'refinement_noise', 0.05)
        # --- Constraint Parameters ---
        constrain_ucb_lcb = getattr(self.acquisition_config, 'constrain_ucb_lcb', True)
        # Use kappa from UCB acquisition if available, otherwise default (or add specific config)
        kappa_constraint = getattr(self.acquisition_config, 'ucb_lcb_kappa',
                                   getattr(self.acquisition_config, 'kappa', 2.5))


        dim = self.bo_config.input_dim
        domain_min, domain_max = self.bounds[0], self.bounds[1]
        best_x_overall = None # Keep track of the best valid point found

        # Ensure observed_best is a scalar float or tensor for BoTorch acq functions
        if isinstance(observed_best, torch.Tensor) and observed_best.numel() > 1:
             raise ValueError("observed_best must be a scalar or scalar tensor for acquisition optimization.")

        # --- 1. Broad Random Search ---
        with torch.no_grad():
            # Generate initial random samples
            if not constrain_ucb_lcb:
                x_samples = domain_min + (domain_max - domain_min) * torch.rand(
                    num_samples, dim, device=self.device, dtype=self.dtype
                )
            else:
                # Generate samples around best observed point
                if self.data_y.shape[0] > 1:
                    sorted_idx = self.data_y.argsort(descending=True) if self.objective_mode == "maximize" else self.data_y.argsort(descending=False)
                    best_x = self.data_x[sorted_idx[0]]
                    second_best_x = self.data_x[sorted_idx[1]]
                    diameter = torch.norm(best_x - second_best_x)
                    # Generate samples around the best point
                    x_samples = best_x + diameter * torch.randn(
                        num_samples, dim, device=self.device, dtype=self.dtype
                    )
                    # clip to domain
                    x_samples = torch.min(torch.max(x_samples, domain_min), domain_max)

                else:
                    x_samples = domain_min + (domain_max - domain_min) * torch.rand(
                        num_samples, dim, device=self.device, dtype=self.dtype
                    )

            max_lcb = -float('inf') # Initialize max LCB

            # --- Apply UCB >= max(LCB) constraint IF enabled ---
            if constrain_ucb_lcb:
                # Calculate LCB and UCB for all random samples
                mean_samples, stddev_samples = self._get_posterior_mean_stddev(x_samples, gp_idx)
                lcb_samples = mean_samples - kappa_constraint * stddev_samples
                ucb_samples = mean_samples + kappa_constraint * stddev_samples

                # Find the maximum LCB value among the samples
                max_lcb = torch.max(lcb_samples).item()

                # Create a mask for valid points (UCB >= max_LCB)
                valid_mask = (ucb_samples >= max_lcb)

                if self.verbose and valid_mask.sum() == 0:
                    print(f"Warning: No points satisfied UCB >= max(LCB) constraint (max LCB: {max_lcb:.4f}). Optimization might be difficult.")
                elif self.verbose == 2:
                    #  pass
                     print(f"Constraint Info GP ({gp_idx}): max(LCB)={max_lcb:.4f}. Found {valid_mask.sum()}/{num_samples} valid points in broad search.")


            # Evaluate the PRIMARY acquisition function
            acq_values = self._acquisition_function_value_botorch(x_samples, gp_idx, observed_best)

            # If constraint is active, filter out invalid points by setting acq value low
            if constrain_ucb_lcb:
                # Use a very small number instead of -inf to avoid issues with argmax if all are invalid
                acq_values[~valid_mask] = -1e20 # Or -float('inf') if argmax handles it

            # Check if any valid points were found
            if constrain_ucb_lcb and valid_mask.sum() == 0:
                 # If no points are valid, we can't really optimize.
                 # Let's pick the point with the highest (invalid) UCB.
                 best_fallback_idx = torch.argmax(ucb_samples)
                 best_x_overall = x_samples[best_fallback_idx].clone()
                 print(f"WARN: No valid points found for constraint UCB >= max(LCB). Falling back to point with highest UCB: {best_x_overall}")
                 # Skip refinement if no valid points
                 num_restarts = 0
            else:
                 # Get top candidates based on acquisition values
                 # Ensure we don't request more restarts than valid points found
                 actual_num_restarts = min(num_restarts, num_samples if not constrain_ucb_lcb else valid_mask.sum().item())
                 if actual_num_restarts <=0 and num_restarts > 0:
                      # This case should be covered above, but as a safeguard
                      print("WARN: num_restarts > 0 but no valid points found. Setting restarts to 0.")
                      actual_num_restarts = 0

                 if actual_num_restarts > 0:
                    top_indices = torch.argsort(acq_values, descending=True)[:actual_num_restarts]
                    top_points = x_samples[top_indices]
                    # Initialize overall best with the best from broad search (that satisfies constraint if active)
                    best_x_overall = top_points[0].clone()
                 else:
                      # Handle case where restarts is 0 or no valid points for restarts
                      # Find the single best point according to acq_values (might be invalid if constraint active and all fail)
                      best_idx = torch.argmax(acq_values)
                      best_x_overall = x_samples[best_idx].clone()
                      # Set top_points empty to skip refinement loop
                      top_points = torch.empty((0, dim), device=self.device, dtype=self.dtype)
                      if constrain_ucb_lcb and not valid_mask[best_idx]:
                           print(f"WARN: Best point from broad search ({best_x_overall}) doesn't satisfy constraint, but was chosen as best overall.")


        # --- 2. Local Refinement around top candidates (if any) ---
        best_acq_value_refined = -float('inf') # Track best acq value found during refinement

        # Only refine if we have valid starting points
        if best_x_overall is not None and top_points.shape[0] > 0 :
             # Get the initial best acquisition value (might be low if constrained)
             initial_best_acq_val = self._acquisition_function_value_botorch(best_x_overall, gp_idx, observed_best).item()
             best_acq_value_refined = initial_best_acq_val

             with torch.no_grad():
                 for start_point in top_points:
                     # Generate local perturbations
                     noise = refinement_noise * (domain_max - domain_min) * (
                         torch.rand(refinement_samples, dim, device=self.device, dtype=self.dtype) * 2 - 1
                     )
                     perturbed_points = start_point.unsqueeze(0) + noise
                     perturbed_points = torch.min(torch.max(perturbed_points, domain_min), domain_max) # Clip

                     # --- Apply constraint check to refinement samples IF enabled ---
                     valid_refinement_mask = torch.ones(refinement_samples, dtype=torch.bool, device=self.device) # Default to all valid
                     if constrain_ucb_lcb:
                         mean_perturbed, stddev_perturbed = self._get_posterior_mean_stddev(perturbed_points, gp_idx)
                         ucb_perturbed = mean_perturbed + kappa_constraint * stddev_perturbed
                         valid_refinement_mask = (ucb_perturbed >= max_lcb)

                     # Evaluate primary acquisition function for perturbed points
                     local_acq_values = self._acquisition_function_value_botorch(
                         perturbed_points, gp_idx, observed_best
                         ) # Batch evaluation

                     # Filter out invalid refinement points
                     if constrain_ucb_lcb:
                         local_acq_values[~valid_refinement_mask] = -1e20 # Or -float('inf')

                     # Find best in this local search (among potentially filtered points)
                     if valid_refinement_mask.sum() > 0 or not constrain_ucb_lcb:
                         best_local_idx = torch.argmax(local_acq_values)
                         local_best_acq = local_acq_values[best_local_idx].item()

                         # Update overall best if improved AND the point is valid (or constraint inactive)
                         current_best_is_valid = True # Assume valid if constraint inactive
                         if constrain_ucb_lcb:
                              current_best_is_valid = valid_refinement_mask[best_local_idx].item()

                         if current_best_is_valid and local_best_acq > best_acq_value_refined:
                             best_acq_value_refined = local_best_acq
                             best_x_overall = perturbed_points[best_local_idx].clone()

        # --- Return the best point found ---
        if best_x_overall is None:
             # This should only happen if opt_samples is 0. Fallback: return a random point.
             print("Error: Optimization failed to find any candidate point. Returning random.")
             best_x_overall = domain_min + (domain_max - domain_min) * torch.rand(
                 1, dim, device=self.device, dtype=self.dtype
             )
             return best_x_overall # Shape [1, D]
        else:
            # Ensure shape [1, D]
            return best_x_overall.unsqueeze(0) if best_x_overall.ndim == 1 else best_x_overall


    # --- State Extraction (Internal Helper) ---
    def _extract_state(self, current_data_x, current_data_y, current_step) -> torch.Tensor:
        """Extract state representation for the transformer.

        When simulation.state_include_gp_params is True (default) the state is:
            [GP_params × (num_models × 2), best_value, step_norm, best_position × input_dim]

        When False the GP hyperparameter block is omitted:
            [best_value, step_norm, best_position × input_dim]

        The no-GP-params variant removes the training/deployment state inconsistency
        that exists in DT-GT mode: rollout GP hyperparameters are refitted on
        growing true-observation datasets, while deployment uses the base-fitted GP.
        """
        include_gp = getattr(self.simulation_config, 'state_include_gp_params', True)

        # 1. GP Hyperparameters (skipped when include_gp is False)
        gp_params_flat = []
        if include_gp:
            if not self.gp_ensemble:
                num_models = self.gp_config.num_models
                gp_params_flat = [1.0] * (num_models * 2)
                print("Warning: Extracting state before GP ensemble is fully initialized. Using default params.")
            else:
                for gp_dict in self.gp_ensemble:
                    model = gp_dict['model']
                    params = []
                    for param_name, param, constraint in model.named_parameters_and_constraints():
                        if 'lengthscale' in param_name:
                            params.append(param.detach().mean().item())
                        elif 'outputscale' in param_name:
                            params.append(param.detach().mean().item())
                    if not params:
                        params = [p.detach().mean().item() for p in model.parameters()]
                    gp_param_dim_assumed = 2
                    if len(params) < gp_param_dim_assumed:
                        params.extend([1.0] * (gp_param_dim_assumed - len(params)))
                    elif len(params) > gp_param_dim_assumed:
                        params = params[:gp_param_dim_assumed]
                    gp_params_flat.extend(params)

        # 2. Best observed value
        if current_data_y.shape[0] == 0:
            best_value = 0.0
        else:
            best_value = (current_data_y.max().item() if self.objective_mode == "maximize"
                          else current_data_y.min().item())

        # 3. Normalised step counter
        step_norm = float(current_step) / self.bo_config.max_iterations

        # 4. Best observed position
        best_position = torch.zeros(self.bo_config.input_dim, device=self.device, dtype=self.dtype)
        if current_data_x.shape[0] > 0:
            best_position = current_data_x[
                current_data_y.argmax() if self.objective_mode == "maximize"
                else current_data_y.argmin()
            ]

        # 5. Assemble and validate length
        expected_state_dim = self._get_state_dim()
        state_list = gp_params_flat + [best_value, step_norm] + best_position.flatten().tolist()

        if len(state_list) < expected_state_dim:
            state_list.extend([0.0] * (expected_state_dim - len(state_list)))
        elif len(state_list) > expected_state_dim:
            state_list = state_list[:expected_state_dim]

        return torch.tensor(state_list, device=self.device, dtype=self.dtype)


    # --- Simulation Logic (Internal Helper) ---
    def _simulate_trajectory(self, gp_idx: int, initial_state: torch.Tensor, max_length: int) -> dict:
        """
        Simulate a BO trajectory using one GP model from the ensemble.

        RQ2 addition: at every step we also call the true objective f(x) at the
        same query point selected by the acquisition function.  This produces a
        parallel 'true' reward sequence that can be compared with the GP-simulated
        one without any extra BO iterations.

        The true tracker uses the same starting best (from real observed data) but
        updates independently on true f(x) values, so divergence after step 1
        reflects pure GP misspecification — not different starting conditions.
        """
        if not self.gp_ensemble: raise RuntimeError("GP Ensemble not initialized.")
        gp_dict = self.gp_ensemble[gp_idx]
        model = gp_dict['model']
        likelihood = gp_dict['likelihood']

        # Track trajectory (existing)
        states = [initial_state]
        actions = []
        rewards = []          # GP-simulated improvement rewards

        # RQ2: parallel lists for true function values and true rewards
        simulated_ys: list = []   # raw GP posterior samples per step
        true_ys: list      = []   # true f(x) at the same query point
        true_rewards: list = []   # true improvement rewards (same definition as sim)

        # Use copies of data for simulation to avoid modifying real data
        sim_data_x = self.data_x.clone()
        sim_data_y = self.data_y.clone()

        if sim_data_y.shape[0] > 0:
            observed_best      = sim_data_y.max().item() if self.objective_mode == "maximize" else sim_data_y.min().item()
            true_observed_best = observed_best   # both trackers start from the same real best
        else:
            observed_best      = -float('inf') if self.objective_mode == "maximize" else float('inf')
            true_observed_best = observed_best

        current_step = self.data_x.shape[0]  # Starting step number

        model.eval()
        likelihood.eval()

        for step in range(max_length):
            # 1. Select next point using acquisition function for *this specific GP*
            obs_best_tensor = torch.tensor(observed_best, device=self.device, dtype=self.dtype)
            next_x_tensor = self._optimize_acquisition(gp_idx, obs_best_tensor)  # [1, D]
            actions.append(next_x_tensor.squeeze(0))

            # 2. Sample simulated observation from GP posterior
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                posterior  = likelihood(model(next_x_tensor))
                sampled_y  = posterior.sample()   # shape [1]
            sampled_y_item = sampled_y.item()
            simulated_ys.append(sampled_y_item)

            # 3. RQ2 — query the TRUE objective at the same point
            with torch.no_grad():
                true_y_raw = self.objective_function(next_x_tensor.squeeze(0))
                true_y_item = true_y_raw.item() if isinstance(true_y_raw, torch.Tensor) else float(true_y_raw)
            true_ys.append(true_y_item)

            # 4. Update simulated data (used by _extract_state — unchanged)
            sim_data_x = torch.cat([sim_data_x, next_x_tensor], dim=0)
            sim_data_y = torch.cat([sim_data_y, sampled_y], dim=0)

            # 5. Simulated reward: improvement over GP-tracked best
            reward = 0.0
            if self.objective_mode == "maximize":
                if sampled_y_item > observed_best:
                    reward        = sampled_y_item - observed_best
                    observed_best = sampled_y_item
            else:
                if sampled_y_item < observed_best:
                    reward        = observed_best - sampled_y_item
                    observed_best = sampled_y_item
            rewards.append(reward)

            # 6. RQ2 — True reward: same improvement definition but using f(x)
            true_reward = 0.0
            if self.objective_mode == "maximize":
                if true_y_item > true_observed_best:
                    true_reward        = true_y_item - true_observed_best
                    true_observed_best = true_y_item
            else:
                if true_y_item < true_observed_best:
                    true_reward        = true_observed_best - true_y_item
                    true_observed_best = true_y_item
            true_rewards.append(true_reward)

            # 7. Update state for the next step (uses simulated data — unchanged)
            current_step += 1
            new_state = self._extract_state(sim_data_x, sim_data_y, current_step)
            states.append(new_state)

            # 8. Early stopping (unchanged)
            early_stop_thresh = getattr(self.simulation_config, 'early_stop_threshold', 1e-6)
            if getattr(self.simulation_config, 'early_stop', False) and reward < early_stop_thresh:
                if getattr(self.simulation_config, 'verbose', False):
                    print(f" Sim Rollout Early Stop: Reward {reward:.2e} < {early_stop_thresh:.2e}")
                break

        # Simulated final regret (existing)
        final_regret = self._compute_simulated_regret(sim_data_y)

        # RQ2 — Counterfactual final regret: best true f(x) reached along the GP-chosen path.
        # NOTE: this is NOT an independent true trajectory — query points were still selected
        # by the GP's acquisition function using sim_observed_best.  It answers the
        # counterfactual "how good were the GP's chosen points in reality?" and is valid
        # for per-step metrics (rtg_rmse, rtg_bias, per_step_*) but NOT for Wasserstein
        # distance (which requires Setup B: independent true BO with true_observed_best
        # driving the acquisition).
        counterfactual_final_regret = (
            -true_observed_best if self.objective_mode == "maximize" else true_observed_best
        )

        return {
            'states':                       torch.stack(states).to(self.device, self.dtype),
            'actions':                      torch.stack(actions).to(self.device, self.dtype),
            'rewards':                      torch.tensor(rewards,       device=self.device, dtype=self.dtype),
            'final_regret':                 final_regret,
            # RQ2 Setup A additions (same query points as GP, counterfactual true values)
            'simulated_ys':                 torch.tensor(simulated_ys,  device=self.device, dtype=self.dtype),
            'true_ys':                      torch.tensor(true_ys,       device=self.device, dtype=self.dtype),
            'true_rewards':                 torch.tensor(true_rewards,  device=self.device, dtype=self.dtype),
            'counterfactual_final_regret':  counterfactual_final_regret,
        }


    def _compute_simulated_regret(self, sim_data_y):
        """Compute simple regret based on simulated observations."""
        if sim_data_y.shape[0] == 0: return 0.0 # Or NaN?

        if self.objective_mode == "maximize":
            best_sim_value = sim_data_y.max().item()
            # Regret for maximization is often defined relative to a known optimum (if available)
            # Or simply use negative best value (lower is better regret)
            return -best_sim_value
        else:
            best_sim_value = sim_data_y.min().item()
            # Regret for minimization is often best value itself (lower is better)
            return best_sim_value


    # --- Decision Transformer Training ---
    def _train_decision_transformer(self, trajectories: list):
        """Train the Decision Transformer on simulated trajectories."""
        if not trajectories:
             print("No trajectories provided for training.")
             return

        # Prepare data for transformer (padding, masking, return-to-go)
        # Prepare batches
        all_states = []
        all_actions = []
        all_rewards = []
        all_timesteps = []
        all_masks = []
        
        max_len = 0
        for traj in trajectories:
            # Get length of this trajectory
            traj_len = min(len(traj['states']) - 1, self.config.transformer.max_seq_length)
            max_len = max(max_len, traj_len)
            
            # Add to lists
            all_states.append(traj['states'][:traj_len])
            all_actions.append(traj['actions'][:traj_len])
            
            # For return-to-go, we want the cumulative future reward at each step
            traj_rewards = traj['rewards'][:traj_len]
            rtg = torch.zeros_like(traj_rewards)
            for i in range(traj_len):
                rtg[i] = traj_rewards[i:].sum()  # Sum of future rewards
            all_rewards.append(rtg)
            
            # Create timestep tensor
            timesteps = torch.arange(traj_len, device=self.device, dtype=self.dtype)
            all_timesteps.append(timesteps)
            
            # Create mask (1 for real data, 0 for padding)
            mask = torch.ones(traj_len, device=self.device, dtype=torch.bool)
            all_masks.append(mask)

        if not all_actions: # Check if any valid trajectories remained
             print("No valid actions found in trajectories for training.")
             return

        # Pad sequences to same length
        padded_states = torch.zeros((len(trajectories), max_len, all_states[0].shape[-1]), device=self.device, dtype=self.dtype)
        padded_actions = torch.zeros((len(trajectories), max_len, self.config.bo.input_dim), device=self.device, dtype=self.dtype)
        padded_rewards = torch.zeros((len(trajectories), max_len), device=self.device, dtype=self.dtype)
        padded_timesteps = torch.zeros((len(trajectories), max_len), device=self.device, dtype=torch.long)
        padded_masks = torch.zeros((len(trajectories), max_len), device=self.device, dtype=torch.bool)
        
        for i in range(len(trajectories)):
            traj_len = len(all_states[i])
            padded_states[i, :traj_len] = all_states[i]
            padded_actions[i, :traj_len] = all_actions[i]
            padded_rewards[i, :traj_len] = all_rewards[i]
            padded_timesteps[i, :traj_len] = all_timesteps[i]
            padded_masks[i, :traj_len] = all_masks[i]
        


        # Training loop
        self.decision_transformer.train()
        
        epoch_iterator = tqdm(range(self.config.transformer.num_epochs), desc="Training Decision Transformer", disable=not self.verbose)
        for epoch in epoch_iterator:
            # Create batch indices
            batch_size = min(self.config.transformer.batch_size, len(trajectories))
            indices = torch.randperm(len(trajectories))
            
            total_loss = 0
            num_batches = 0
            
            for i in range(0, len(trajectories), batch_size):
                batch_indices = indices[i:i+batch_size]
                
                # Get batch data
                batch_states = padded_states[batch_indices]
                batch_actions = padded_actions[batch_indices]
                batch_rewards = padded_rewards[batch_indices]
                batch_timesteps = padded_timesteps[batch_indices]
                batch_masks = padded_masks[batch_indices]
                
                # Forward pass
                predicted_actions = self.decision_transformer(
                    batch_states, 
                    batch_actions,
                    batch_rewards, 
                    batch_timesteps,
                    batch_masks
                )
                
                # Compute loss (MSE on action prediction)
                loss = torch.nn.functional.mse_loss(
                    predicted_actions[batch_masks], 
                    batch_actions[batch_masks]
                )
                
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.decision_transformer.parameters(), 
                    1.0
                )
                
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)
            epoch_iterator.set_postfix(loss=avg_loss)

        return avg_loss   # final epoch loss, used by _propose_next_candidate


    # --- Main Logic for Proposing Next Candidate ---
    def _propose_next_candidate(self) -> torch.Tensor:
        """
        Generate training trajectories, train the Decision Transformer, and propose
        the next query point.  Returns tensor shape [1, input_dim].

        Two trajectory modes are supported, controlled by
        simulation_config.use_true_trajectories:

          False (default) — DT-GP mode: standard DRO.
            The GP's acquisition function selects query points; observations come
            from GP posterior samples.  RQ2 Setup A and optional Setup B logs are
            collected as side-effects.

          True — DT-GT mode: oracle upper bound.
            The TRUE objective drives acquisition and observations at every rollout
            step.  The DT is trained on ground-truth return-to-go sequences.  This
            consumes (num_rollouts × max_rollout_length) additional true evaluations
            per BO iteration — report this cost explicitly when comparing with DT-GP.

        All BO mechanics outside the trajectory generation step (GP fitting, state
        extraction, DT architecture, real query evaluation) are identical between
        the two modes, so the regret gap is attributable solely to simulation bias.
        """
        use_gt       = getattr(self.simulation_config, 'use_true_trajectories', False)
        num_rollouts   = self.simulation_config.num_rollouts
        rollout_length = self.simulation_config.max_rollout_length

        # Helper: iterate over GP ensemble indices + remainder, yield (gp_idx, n_times)
        def _rollout_schedule():
            rpt = max(1, num_rollouts // len(self.gp_ensemble))
            for idx in range(len(self.gp_ensemble)):
                yield idx, rpt
            for idx in range(num_rollouts % len(self.gp_ensemble)):
                yield idx, 1

        if use_gt:
            # ── DT-GT mode ────────────────────────────────────────────────────────
            # Generate independent true-function rollouts.  The acquisition function
            # is anchored to true_observed_best at every step, so query points
            # diverge from the GP-simulated path — this is what makes the
            # trajectories a genuine oracle training set for the DT.
            if self.verbose:
                print("Generating true trajectories for DT-GT training...")
            training_trajectories = []
            for gp_idx, n in tqdm(_rollout_schedule(), total=num_rollouts,
                                   desc="True Rollouts (DT-GT)", disable=not self.verbose):
                for _ in range(n):
                    s = self._extract_state(self.data_x, self.data_y, self.data_x.shape[0])
                    training_trajectories.append(
                        self._generate_true_trajectory(gp_idx, s, rollout_length)
                    )

            # Store for optional later analysis / save_true_trajectory_logs()
            self.true_trajectory_logs.append(
                self._collect_true_iteration_log(training_trajectories)
            )

        else:
            # ── DT-GP mode (standard DRO) ─────────────────────────────────────────
            if self.verbose:
                print("Simulating GP trajectories for DT-GP training...")
            training_trajectories = []
            for gp_idx, n in tqdm(_rollout_schedule(), total=num_rollouts,
                                   desc="GP Rollouts (DT-GP)", disable=not self.verbose):
                for _ in range(n):
                    s = self._extract_state(self.data_x, self.data_y, self.data_x.shape[0])
                    training_trajectories.append(
                        self._simulate_trajectory(gp_idx, s, rollout_length)
                    )

            # RQ2 Setup A — per-step counterfactual fidelity log
            self.trajectory_logs.append(
                self._collect_iteration_log(training_trajectories)
            )

            # RQ2 Setup B — optional independent true rollouts for Wasserstein metric
            if getattr(self.simulation_config, 'log_true_trajectories', False):
                true_trajectories = []
                for gp_idx, n in _rollout_schedule():
                    for _ in range(n):
                        s = self._extract_state(self.data_x, self.data_y, self.data_x.shape[0])
                        true_trajectories.append(
                            self._generate_true_trajectory(gp_idx, s, rollout_length)
                        )
                self.true_trajectory_logs.append(
                    self._collect_true_iteration_log(true_trajectories)
                )

        self._bo_iteration_count += 1

        # ── Log reward/RTG diagnostics before training ────────────────────────────
        mode_label = "DT-GT" if use_gt else "DT-GP"
        diag = self._log_trajectory_rewards(training_trajectories, mode_label)
        if diag:
            self.reward_diagnostics.append(diag)

        # ── Train the Decision Transformer on whichever trajectories were generated ──
        if self.verbose:
            print(f"Training Decision Transformer ({mode_label})...")
        dt_final_loss = self._train_decision_transformer(training_trajectories)

        # ── Propose next candidate via trained DT ─────────────────────────────────
        if self.verbose:
            print("Proposing next candidate via Decision Transformer...")
        self.decision_transformer.eval()

        # Target RTG determination — three modes:
        #   fixed          : constant value from config.simulation.target_rtg
        #   dynamic        : max_rtg from current training batch (0 if all-zero)
        #   floored_dynamic: max(max_rtg_batch, alpha * running_max_rtg)
        #                    prevents zero-collapse while staying proportional to
        #                    the function's actual reward scale.
        fixed_rtg       = getattr(self.simulation_config, 'target_rtg', None)
        rtg_mode        = getattr(self.simulation_config, 'rtg_mode', 'dynamic')
        max_rtg_batch   = float(diag.get('max_rtg', 0.0)) if diag else 0.0

        floor_value  = 0.0
        floor_active = 0

        if fixed_rtg is not None:
            # Legacy fixed mode (backward-compatible: any non-null target_rtg → fixed)
            target_rtg_val = float(fixed_rtg)
        elif rtg_mode == 'floored_dynamic':
            alpha          = float(self.simulation_config.rtg_alpha)
            floor_value    = alpha * self._running_max_rtg
            target_rtg_val = float(max(max_rtg_batch, floor_value))
            floor_active   = int(floor_value > max_rtg_batch)
            # Update running max AFTER computing target so iteration t uses history 1…t-1
            self._running_max_rtg = max(self._running_max_rtg, max_rtg_batch)
        else:
            # Standard dynamic mode
            target_rtg_val = float(max_rtg_batch) if max_rtg_batch > 0.0 else 0.0

        with torch.no_grad():
            current_real_state = self._extract_state(
                self.data_x, self.data_y, self.data_x.shape[0]
            )
            state_seq     = current_real_state.unsqueeze(0).unsqueeze(0)          # [1, 1, S]
            dummy_action  = torch.zeros((1, 1, self.bo_config.input_dim),
                                        device=self.device, dtype=self.dtype)     # [1, 1, A]
            target_rtg    = torch.tensor([[target_rtg_val]], device=self.device,
                                          dtype=self.dtype)                        # [1, 1]
            dummy_ts      = torch.zeros((1, 1), device=self.device,
                                         dtype=torch.long)                         # [1, 1]

            predicted_seq = self.decision_transformer(
                state_seq, dummy_action, target_rtg, dummy_ts, attention_mask=None
            )                                                                      # [1, 1, A]
            next_action   = predicted_seq.squeeze(0).squeeze(0)                   # [A]

            domain_min, domain_max = self.bounds[0], self.bounds[1]
            next_action = torch.min(torch.max(next_action, domain_min), domain_max)

        # ── Back-fill diagnostics with post-training metrics ──────────────────────
        if diag:
            diag['dt_final_loss']      = round(float(dt_final_loss), 6) if dt_final_loss is not None else None
            diag['target_rtg_used']    = round(target_rtg_val, 6)
            diag['max_rtg_from_batch'] = round(max_rtg_batch, 6)
            # Floored-dynamic fields (zero/None for other modes so JSON is always uniform)
            diag['running_max_rtg']    = round(self._running_max_rtg, 6)
            diag['floor_value']        = round(floor_value, 6)
            diag['floor_active']       = floor_active
            print(
                f"[{mode_label}] iter={self._bo_iteration_count:3d} | "
                f"dt_loss={diag['dt_final_loss']}  target_RTG={diag['target_rtg_used']:.4f}"
                + (f"  floor={floor_value:.4f}(active={floor_active})" if rtg_mode == 'floored_dynamic' else "")
            )

        return next_action.unsqueeze(0)   # [1, input_dim]


    # -----------------------------------------------------------------------
    # RQ2 — Fidelity Logging Helpers
    # -----------------------------------------------------------------------

    def _collect_iteration_log(self, trajectories: list) -> dict:
        """
        Pack fidelity data from all rollouts of one BO iteration into a dict
        of fixed-shape numpy arrays.

        Short rollouts (early-stopped) are right-padded with NaN so every
        array has shape [num_rollouts, max_rollout_length].  The metrics code
        uses np.nanmean / np.nanstd to skip padded positions automatically.

        Returns
        -------
        dict with keys:
            bo_iteration        int
            simulated_ys        [R, T]  raw GP posterior samples
            true_ys             [R, T]  true f(x) at same query points
            sim_rewards         [R, T]  GP-improvement rewards
            true_rewards        [R, T]  true-improvement rewards
            sim_final_regrets   [R]     final simulated regret per rollout
            true_final_regrets  [R]     final true regret per rollout
        """
        T = self.simulation_config.max_rollout_length
        R = len(trajectories)

        simulated_ys                = np.full((R, T), np.nan)
        true_ys                     = np.full((R, T), np.nan)
        sim_rewards                 = np.full((R, T), np.nan)
        true_rewards                = np.full((R, T), np.nan)
        sim_final_regrets           = np.zeros(R)
        counterfactual_final_regrets = np.zeros(R)   # best true f at GP-chosen points

        for i, traj in enumerate(trajectories):
            t = traj['rewards'].shape[0]   # actual steps taken (may be < T)
            simulated_ys[i, :t]  = traj['simulated_ys'].cpu().numpy()
            true_ys[i, :t]       = traj['true_ys'].cpu().numpy()
            sim_rewards[i, :t]   = traj['rewards'].cpu().numpy()
            true_rewards[i, :t]  = traj['true_rewards'].cpu().numpy()
            sim_final_regrets[i]            = float(traj['final_regret'])
            counterfactual_final_regrets[i] = float(traj['counterfactual_final_regret'])

        return {
            'bo_iteration':               self._bo_iteration_count,
            'simulated_ys':               simulated_ys,
            'true_ys':                    true_ys,
            'sim_rewards':                sim_rewards,
            'true_rewards':               true_rewards,
            'sim_final_regrets':          sim_final_regrets,
            'counterfactual_final_regrets': counterfactual_final_regrets,
        }

    def save_fidelity_logs(self, save_dir: str, prefix: str = "dro_fidelity") -> str:
        """
        Save all accumulated trajectory logs to disk as two files:

          <prefix>_log.npz   — numerical arrays of shape [I, R, T]
                               (I = BO iterations, R = rollouts, T = steps)
          <prefix>_meta.json — human-readable experiment metadata

        Call this once after run_optimization() completes.

        Returns
        -------
        Path to the .npz file (empty string if no logs were collected).
        """
        if not self.trajectory_logs:
            print("Warning: No trajectory logs to save (trajectory_logs is empty).")
            return ""

        os.makedirs(save_dir, exist_ok=True)

        def _stack(key: str) -> np.ndarray:
            """Stack per-iteration arrays along a new axis-0."""
            return np.array([log[key] for log in self.trajectory_logs])

        npz_path = os.path.join(save_dir, f"{prefix}_log.npz")
        np.savez(
            npz_path,
            simulated_ys                 = _stack('simulated_ys'),                # [I, R, T]
            true_ys                      = _stack('true_ys'),                     # [I, R, T]
            sim_rewards                  = _stack('sim_rewards'),                  # [I, R, T]
            true_rewards                 = _stack('true_rewards'),                 # [I, R, T]
            sim_final_regrets            = _stack('sim_final_regrets'),            # [I, R]
            counterfactual_final_regrets = _stack('counterfactual_final_regrets'), # [I, R]
            bo_iterations                = np.array(
                [log['bo_iteration'] for log in self.trajectory_logs]
            ),                                                                     # [I]
        )

        meta = {
            'num_bo_iterations':  len(self.trajectory_logs),
            'num_rollouts':       int(self.trajectory_logs[0]['simulated_ys'].shape[0]),
            'max_rollout_length': int(self.simulation_config.max_rollout_length),
            'num_gp_models':      int(self.gp_config.num_models),
            'input_dim':          int(self.bo_config.input_dim),
            'objective_mode':     self.objective_mode,
            'acquisition_fn':     str(getattr(self.acquisition_config, 'function', 'unknown')),
            'seed':               int(getattr(self.config, 'seed', -1)),
        }
        json_path = os.path.join(save_dir, f"{prefix}_meta.json")
        with open(json_path, 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"Fidelity log  saved → {npz_path}")
        print(f"Fidelity meta saved → {json_path}")
        return npz_path

    # -----------------------------------------------------------------------
    # RQ2 Setup B — Independent True Trajectory Generation
    # -----------------------------------------------------------------------

    def _generate_true_trajectory(self, gp_idx: int, initial_state: torch.Tensor, max_length: int) -> dict:
        """
        Setup B: run an independent BO rollout where the TRUE function drives
        everything — both reward and the acquisition function's best_f anchor.

        Key differences from _simulate_trajectory():
          - 'true_observed_best' (updated by f(x)) anchors the acquisition, so
            query points diverge from the simulated trajectory after step 1.
          - There is NO GP posterior sampling — observations are always true f(x).
          - After each true observation the GP posterior is updated so subsequent
            acquisition steps see coherent, non-stale beliefs about the landscape.

        How the posterior update works
        ------------------------------
        After observing true f(x_t) we call model.set_train_data() to append
        (x_t, y_t) to the model's training set, then run a short Adam refit
        (controlled by simulation_config.refit_steps_in_true_trajectory, default 20)
        to update kernel hyperparameters for the expanded dataset.  This is
        critical: without refitting, the GP uses hyperparameters calibrated on only
        the initial n points even as the rollout dataset grows — the acquisition
        function then points to regions the GP misidentifies as promising, producing
        zero-reward training sequences that cause DT collapse.

        The full model state (hyperparameters + training data) is saved before the
        rollout and restored afterwards so nothing outside this rollout is affected.
        Within the rollout, hyperparameters warm-start from the previous step's fit,
        so each refit needs only a few steps to track the growing dataset.

        Why NOT get_fantasy_model?
        --------------------------
        BoTorch's get_fantasy_model returns a model with an extra batch dimension
        representing "fantasy index".  This breaks qMaxValueEntropy (MES does
        not support batch GP models) and causes indexing errors in the acquisition
        optimisation loop.  set_train_data avoids the batch dimension entirely
        while producing an identical posterior update.

        Returns the same keys as _simulate_trajectory() so _train_decision_transformer
        works unchanged on either trajectory type.
        """
        if not self.gp_ensemble:
            raise RuntimeError("GP Ensemble not initialized.")
        gp_dict = self.gp_ensemble[gp_idx]
        model   = gp_dict['model']          # the actual object in the ensemble
        likelihood = gp_dict['likelihood']

        states  = [initial_state]
        actions = []
        rewards = []   # true improvement rewards — DT return-to-go is built from these
        true_ys = []   # raw true f(x) values at each step

        # True data bookkeeping — starts from the same real observations
        true_data_x = self.data_x.clone()
        true_data_y = self.data_y.clone()

        if true_data_y.shape[0] > 0:
            true_observed_best = (
                true_data_y.max().item() if self.objective_mode == "maximize"
                else true_data_y.min().item()
            )
        else:
            true_observed_best = -float('inf') if self.objective_mode == "maximize" else float('inf')

        # Save the model's original training data so we can restore it each iteration.
        # train_inputs is a tuple; index 0 is the X tensor.
        # train_targets holds Y in the 1-D form that set_train_data expects.
        original_train_X = model.train_inputs[0].clone()
        original_train_Y = model.train_targets.clone()

        # Save the FULL model state (hyperparameters + data) so we can restore
        # everything at the end of the rollout.  Within the rollout, we intentionally
        # let the hyperparameters accumulate updates (warm-start across steps).
        original_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

        # Running dataset for this rollout: starts as real data, grows with each step.
        rollout_X = original_train_X.clone()
        rollout_Y = original_train_Y.clone()

        # How many Adam steps to use when refitting GP hyperparameters each rollout step.
        # A value of 0 disables refitting (fast but stale — produces zero-reward trajectories).
        refit_steps = int(getattr(self.simulation_config, 'refit_steps_in_true_trajectory', 20))

        current_step = self.data_x.shape[0]
        model.eval()
        likelihood.eval()

        for step in range(max_length):
            # 1. Expand the model's training data to include all true observations
            #    collected so far in this rollout, then quickly refit hyperparameters
            #    so the acquisition function uses calibrated, non-stale beliefs.
            #    The finally block restores the ORIGINAL data (but not hyperparameters —
            #    those are warm-started across steps and restored only after the loop).
            obs_best_tensor = torch.tensor(true_observed_best, device=self.device, dtype=self.dtype)
            try:
                model.set_train_data(rollout_X, rollout_Y, strict=False)

                if refit_steps > 0:
                    # Fast hyperparameter refit on the expanded rollout dataset.
                    # Warm-starts from the current hyperparameters (previous step's fit),
                    # so convergence is fast even with few steps.
                    model.train()
                    likelihood.train()
                    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
                    refit_optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        for _ in range(refit_steps):
                            refit_optimizer.zero_grad()
                            output = model(rollout_X)
                            loss = -mll(output, rollout_Y)
                            loss.backward()
                            refit_optimizer.step()

                model.eval()
                likelihood.eval()
                next_x_tensor = self._optimize_acquisition(gp_idx, obs_best_tensor)  # [1, D]
            finally:
                # Restore original training DATA only — hyperparameters carry forward
                # to warm-start the next step's refit (more efficient than cold-start).
                model.set_train_data(original_train_X, original_train_Y, strict=False)
                model.eval()

            actions.append(next_x_tensor.squeeze(0))

            # 2. Evaluate TRUE function — no GP posterior sampling
            with torch.no_grad():
                true_y_raw  = self.objective_function(next_x_tensor.squeeze(0))
                true_y_item = true_y_raw.item() if isinstance(true_y_raw, torch.Tensor) else float(true_y_raw)
            true_ys.append(true_y_item)

            # 3. Grow the rollout dataset so the next step sees this observation.
            new_y_scalar = torch.tensor([true_y_item], device=self.device, dtype=self.dtype)
            rollout_X = torch.cat([rollout_X, next_x_tensor], dim=0)   # [n+step+1, D]
            rollout_Y = torch.cat([rollout_Y, new_y_scalar],  dim=0)   # [n+step+1]

            # 4. Update true data bookkeeping (used for state extraction below)
            true_data_x = torch.cat([true_data_x, next_x_tensor], dim=0)
            true_data_y = torch.cat([true_data_y, new_y_scalar],   dim=0)

            # 5. True reward (same improvement definition as simulated)
            true_reward = 0.0
            if self.objective_mode == "maximize":
                if true_y_item > true_observed_best:
                    true_reward        = true_y_item - true_observed_best
                    true_observed_best = true_y_item
            else:
                if true_y_item < true_observed_best:
                    true_reward        = true_observed_best - true_y_item
                    true_observed_best = true_y_item
            rewards.append(true_reward)

            # 6. State uses true data so the DT sees a self-consistent trajectory
            current_step += 1
            new_state = self._extract_state(true_data_x, true_data_y, current_step)
            states.append(new_state)

            # 7. Early stopping on true reward
            early_stop_thresh = getattr(self.simulation_config, 'early_stop_threshold', 1e-6)
            if getattr(self.simulation_config, 'early_stop', False) and true_reward < early_stop_thresh:
                if getattr(self.simulation_config, 'verbose', False):
                    print(f" True Rollout Early Stop: Reward {true_reward:.2e} < {early_stop_thresh:.2e}")
                break

        # Restore the model's full state (hyperparameters + data) to exactly what it
        # was before the rollout began.  This ensures the real BO loop is unaffected
        # by the hyperparameter updates accumulated during trajectory generation.
        model.load_state_dict(original_state_dict)
        model.set_train_data(original_train_X, original_train_Y, strict=False)
        model.eval()
        likelihood.eval()

        # Final regret based on best true value reached in this independent rollout
        true_final_regret = (
            -true_observed_best if self.objective_mode == "maximize" else true_observed_best
        )

        return {
            # Same keys as _simulate_trajectory() so _train_decision_transformer() is unchanged
            'states':       torch.stack(states).to(self.device, self.dtype),
            'actions':      torch.stack(actions).to(self.device, self.dtype),
            'rewards':      torch.tensor(rewards,  device=self.device, dtype=self.dtype),
            'final_regret': true_final_regret,
            # Extra field for logging (not used by DT)
            'true_ys':      torch.tensor(true_ys,  device=self.device, dtype=self.dtype),
        }

    def _collect_true_iteration_log(self, true_trajectories: list) -> dict:
        """
        Pack Setup B true trajectory data into fixed-shape numpy arrays.
        Shape: [num_rollouts, max_rollout_length], NaN-padded for early stops.
        """
        T = self.simulation_config.max_rollout_length
        R = len(true_trajectories)

        true_ys           = np.full((R, T), np.nan)
        true_rewards      = np.full((R, T), np.nan)
        true_final_regrets = np.zeros(R)

        for i, traj in enumerate(true_trajectories):
            t = traj['rewards'].shape[0]
            true_ys[i, :t]      = traj['true_ys'].cpu().numpy()
            true_rewards[i, :t] = traj['rewards'].cpu().numpy()
            true_final_regrets[i] = float(traj['final_regret'])

        return {
            'bo_iteration':       self._bo_iteration_count,
            'true_ys':            true_ys,
            'true_rewards':       true_rewards,
            'true_final_regrets': true_final_regrets,
        }

    def save_true_trajectory_logs(self, save_dir: str, prefix: str = "dro_true_traj") -> str:
        """
        Save Setup B (independent true trajectory) logs.

        Produces:
          <prefix>_log.npz   — true_ys [I,R,T], true_rewards [I,R,T],
                               true_final_regrets [I,R], bo_iterations [I]
          <prefix>_meta.json — experiment metadata

        The true_final_regrets here ARE comparable with sim_final_regrets from
        save_fidelity_logs() for computing Wasserstein distance, because both
        represent the best value reached in rollouts starting from the same initial
        data — the difference is only in how query points were chosen.
        """
        if not self.true_trajectory_logs:
            print("Warning: No true trajectory logs to save (true_trajectory_logs is empty).")
            return ""

        os.makedirs(save_dir, exist_ok=True)

        def _stack(key: str) -> np.ndarray:
            return np.array([log[key] for log in self.true_trajectory_logs])

        npz_path = os.path.join(save_dir, f"{prefix}_log.npz")
        np.savez(
            npz_path,
            true_ys             = _stack('true_ys'),             # [I, R, T]
            true_rewards        = _stack('true_rewards'),        # [I, R, T]
            true_final_regrets  = _stack('true_final_regrets'),  # [I, R]
            bo_iterations       = np.array(
                [log['bo_iteration'] for log in self.true_trajectory_logs]
            ),
        )

        meta = {
            'setup':              'B_independent_true_trajectory',
            'num_bo_iterations':  len(self.true_trajectory_logs),
            'num_rollouts':       int(self.true_trajectory_logs[0]['true_ys'].shape[0]),
            'max_rollout_length': int(self.simulation_config.max_rollout_length),
            'num_gp_models':      int(self.gp_config.num_models),
            'input_dim':          int(self.bo_config.input_dim),
            'objective_mode':     self.objective_mode,
            'acquisition_fn':     str(getattr(self.acquisition_config, 'function', 'unknown')),
            'seed':               int(getattr(self.config, 'seed', -1)),
            'note': (
                'Query points are driven by true_observed_best. '
                'true_final_regrets are comparable with sim_final_regrets from '
                'save_fidelity_logs() for Wasserstein distance.'
            ),
        }
        json_path = os.path.join(save_dir, f"{prefix}_meta.json")
        with open(json_path, 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"True trajectory log  saved → {npz_path}")
        print(f"True trajectory meta saved → {json_path}")
        return npz_path

    # -----------------------------------------------------------------------
    # Reward / RTG diagnostics
    # -----------------------------------------------------------------------

    def _log_trajectory_rewards(self, trajectories: list, mode_label: str) -> dict:
        """
        Compute reward, RTG, rollout-length, and GP-lengthscale statistics for one
        BO iteration's worth of trajectories, called just before
        _train_decision_transformer().

        Metrics
        -------
        frac_zero_steps  : fraction of all (rollout × step) entries with reward == 0.
        mean_rtg / std_rtg : mean ± std of total trajectory return across rollouts.
        max_rtg          : maximum total return seen — used to contextualise target_rtg_used.
        mean_rollout_len : mean actual rollout length (may be < max due to early_stop).
        min_rollout_len  : shortest rollout — reveals if most rollouts collapse to 1 step.
        gp_lengthscales  : mean learned lengthscale for each GP ensemble model.

        Fields dt_final_loss and target_rtg_used are filled in by _propose_next_candidate
        after training and proposal respectively.

        Prints one summary line per BO iteration; verbose==2 adds per-rollout breakdown.
        Returns a flat dict appended to self.reward_diagnostics.
        """
        if not trajectories:
            return {}

        all_rewards  = []
        total_rtgs   = []
        rollout_lens = []
        for traj in trajectories:
            r = traj['rewards'].cpu().numpy()   # [T_actual]
            all_rewards.append(r)
            total_rtgs.append(float(r.sum()))
            rollout_lens.append(len(r))

        rewards_flat  = np.concatenate(all_rewards)
        total_rtgs_np = np.array(total_rtgs)
        lens_np       = np.array(rollout_lens)

        frac_zero_steps  = float(np.mean(rewards_flat == 0.0))
        mean_rtg         = float(np.mean(total_rtgs_np))
        std_rtg          = float(np.std(total_rtgs_np))
        max_rtg          = float(total_rtgs_np.max()) if len(total_rtgs_np) > 0 else 0.0
        max_reward       = float(rewards_flat.max()) if len(rewards_flat) > 0 else 0.0
        mean_rollout_len = float(np.mean(lens_np))
        min_rollout_len  = int(np.min(lens_np))

        # GP ensemble lengthscales (mean per model, after most recent hyperparameter update)
        gp_lengthscales = []
        for gp_dict in self.gp_ensemble:
            ls = gp_dict['model'].covar_module.base_kernel.lengthscale.detach().mean().item()
            gp_lengthscales.append(round(ls, 4))

        stats = {
            'bo_iteration':      self._bo_iteration_count,
            'mode':              mode_label,
            'n_trajectories':    len(trajectories),
            'frac_zero_steps':   frac_zero_steps,
            'mean_rtg':          mean_rtg,
            'std_rtg':           std_rtg,
            'max_rtg':           max_rtg,
            'max_reward':        max_reward,
            'mean_rollout_len':  mean_rollout_len,
            'min_rollout_len':   min_rollout_len,
            'gp_lengthscales':   gp_lengthscales,
            # filled later by _propose_next_candidate:
            'dt_final_loss':     None,
            'target_rtg_used':   None,
        }

        print(
            f"[{mode_label}] iter={self._bo_iteration_count:3d} | "
            f"zero_steps={frac_zero_steps:.1%}  "
            f"mean_RTG={mean_rtg:.4f}±{std_rtg:.4f}  max_RTG={max_rtg:.4f}  "
            f"avg_len={mean_rollout_len:.1f}  min_len={min_rollout_len}  "
            f"ls={gp_lengthscales}"
        )

        if self.verbose == 2:
            for i, (r_arr, rtg_val) in enumerate(zip(all_rewards, total_rtgs)):
                nz = int((r_arr > 0).sum())
                print(
                    f"  rollout {i:3d}: steps={len(r_arr)}  nonzero={nz}  "
                    f"RTG={rtg_val:.4f}  rewards={np.round(r_arr, 4)}"
                )

        return stats

    def save_reward_diagnostics(self, save_dir: str, prefix: str = "reward_diag") -> str:
        """
        Save per-BO-iteration reward/RTG statistics accumulated in
        self.reward_diagnostics to a JSON file.

        The file is a list of dicts, one per BO iteration, each containing:
          bo_iteration, mode, n_trajectories,
          frac_zero_steps, mean_rtg, std_rtg, max_rtg, max_reward,
          mean_rollout_len, min_rollout_len, gp_lengthscales,
          dt_final_loss, target_rtg_used

        This is the primary evidence for (or against) the sparse-reward hypothesis.
        Compare DT-GP and DT-GT entries side-by-side to quantify the reward density gap.
        """
        if not self.reward_diagnostics:
            print("Warning: No reward diagnostics to save (reward_diagnostics is empty).")
            return ""

        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{prefix}_reward_diag.json")
        with open(path, 'w') as f:
            json.dump(self.reward_diagnostics, f, indent=2)

        print(f"Reward diagnostics saved → {path}")
        return path


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

    n_trials = cfg.experiment_params.num_trials
    all_trials_results = []
    all_trials_all_y = []
    all_trials_best_y = []

    if n_trials <= 0:
        print("Warning: cfg.experiment_params.n_trials is <= 0. No trials will be run.")
        return None # Or raise error

    cfg.method.name = 'dro'
    save_prefix = f"{cfg.method.name}_{cfg.test_function.name}"
    

    # --- Run Multiple Trials ---
    iterator = tqdm(range(n_trials), desc="Trials", unit="trial")
    for trial_idx in iterator:
        # Set up the optimization problem for this trial
        # Re-instantiate to ensure independent runs if the optimizer has state
        cfg.method.seed = cfg.experiment_params.seed_start + trial_idx
        dro = DirectRegretOptimization(cfg.method, objective_function)

        # Run optimization
        result = dro.run_optimization()
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
            best_x_trial = result['best_x']
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
            plt.savefig(os.path.join(trial_save_dir, f"contour_plot_trial_{trial_idx}.png"))
            plt.close() # Close plot

        # Save individual trial data
        np.savez(
            os.path.join(trial_save_dir, f"results_trial_{trial_idx}.npz"),
            best_x=result['best_x'],
            best_y=result['best_y'],
            all_x=result['all_x'],
            all_y=result['all_y']
        )

    # --- Aggregate and Plot Results ---
    print("\n--- Aggregating Results Across Trials ---")

    max_len = max(len(y) for y in all_trials_all_y)
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