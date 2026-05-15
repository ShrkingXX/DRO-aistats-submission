import torch
import numpy as np
# import hydra # Not typically used directly within the policy file
import math
import time
from dataclasses import dataclass, field # Not used in this specific file version
from omegaconf import DictConfig, OmegaConf # For config handling
import os
# import matplotlib.pyplot as plt # Not directly used in this file usually
from tqdm import tqdm
# from datetime import datetime # Not directly used
import logging
# import traceback # Not directly used

from typing import Tuple, List, Optional, Any, Dict
import warnings

import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
# from gpytorch.distributions import MultivariateNormal # Not directly used here
from gpytorch.priors import LogNormalPrior, GammaPrior, NormalPrior # For priors

from botorch.models import SingleTaskGP
from botorch.exceptions import InputDataWarning
# from botorch.models.transforms.outcome import Standardize # Standardization handled in class
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import AcquisitionFunction # Base class for custom acqf
from botorch.optim import optimize_acqf, optimize_acqf_discrete
from botorch.utils.transforms import standardize, normalize, unnormalize
from botorch.utils.sampling import draw_sobol_samples
from botorch.generation.sampling import MaxPosteriorSampling


try:
    from .base import BaseBayesianOptimizer
except ImportError:
    # Fallback for different import structures or direct execution
    try:
        from base import BaseBayesianOptimizer
    except ImportError:
        # If base.py is in the same directory as scorebo.py when run directly
        from base import BaseBayesianOptimizer


DEFAULT_DTYPE = torch.float64
log = logging.getLogger(__name__)

# --- SCoreBO Specific Helper Functions ---
def hellinger_distance_gaussian(mean1, var1, mean2, var2, eps=1e-9):
    var1 = torch.clamp(var1, min=eps)
    var2 = torch.clamp(var2, min=eps)
    sd1 = torch.sqrt(var1)
    sd2 = torch.sqrt(var2)
    term1 = torch.sqrt((2 * sd1 * sd2) / (var1 + var2))
    term2 = torch.exp(-0.25 * (mean1 - mean2)**2 / (var1 + var2))
    return torch.sqrt(torch.clamp(1 - term1 * term2, min=0.0, max=1.0))

def wasserstein2_distance_gaussian(mean1, var1, mean2, var2, eps=1e-9):
    var1 = torch.clamp(var1, min=eps)
    var2 = torch.clamp(var2, min=eps)
    sd1 = torch.sqrt(var1)
    sd2 = torch.sqrt(var2)
    return (mean1 - mean2)**2 + (sd1 - sd2)**2

def kl_divergence_gaussian(mean1, var1, mean2, var2, eps=1e-9):
    var1 = torch.clamp(var1, min=eps)
    var2 = torch.clamp(var2, min=eps)
    sd1 = torch.sqrt(var1)
    sd2 = torch.sqrt(var2)
    if torch.any(sd1 <= eps) or torch.any(sd2 <= eps):
        log.warning(f"KL Divergence: Encountered near-zero std dev. sd1: {sd1}, sd2: {sd2}. Clamping KL to large value.")
        return torch.full_like(mean1, 1e9)
    kl_div = torch.log(sd2 / sd1) + (var1 + (mean1 - mean2)**2) / (2 * var2) - 0.5
    return torch.clamp(kl_div, min=0.0)

def moment_match_gmm(means: torch.Tensor, variances: torch.Tensor, eps=1e-9) -> Tuple[torch.Tensor, torch.Tensor]:
    if means.ndim == 0:
        return means, torch.clamp(variances, min=eps)
    if means.shape[0] == 1: # Handles case where dim 0 is 1 after stacking
         return means.squeeze(0), torch.clamp(variances.squeeze(0), min=eps)
    m_gmm = torch.mean(means, dim=0)
    s_gmm = torch.mean(variances + means**2, dim=0) - m_gmm**2
    return m_gmm, torch.clamp(s_gmm, min=eps)


class SCoreBOAcquisitionFunction(AcquisitionFunction):
    def __init__(
        self,
        model: SingleTaskGP,
        base_train_x: torch.Tensor,
        base_train_y: torch.Tensor,
        sampled_hyperparams: List[Dict[str, Any]],
        sampled_optima_per_hyperparam: List[List[Tuple[torch.Tensor, torch.Tensor]]],
        input_dim: int,
        distance_metric: str = "hellinger",
        gp_kernel_config: Optional[DictConfig] = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = DEFAULT_DTYPE,
    ):
        super().__init__(model=model)
        self.base_model = model
        self.base_train_x = base_train_x
        self.base_train_y = base_train_y
        self.sampled_hyperparams = sampled_hyperparams
        self.sampled_optima_per_hyperparam = sampled_optima_per_hyperparam
        self.input_dim = input_dim
        self.distance_metric = distance_metric
        self.gp_kernel_config = gp_kernel_config if gp_kernel_config is not None else OmegaConf.create({'type': 'matern', 'nu': 2.5})
        self.device = device
        self.dtype = dtype
        self.num_hyperparameter_samples_M = len(sampled_hyperparams)
        if self.num_hyperparameter_samples_M > 0 and sampled_optima_per_hyperparam:
            self.num_optima_samples_N = len(sampled_optima_per_hyperparam[0]) if sampled_optima_per_hyperparam[0] else 0
        else:
            self.num_optima_samples_N = 0
        if self.num_hyperparameter_samples_M == 0 or self.num_optima_samples_N == 0:
            log.warning("SCoreBOAcquisitionFunction initialized with no hyperparams or optima samples.")

    def _reconstruct_gp_model(self, train_x, train_y, hyperparams_dict):
        # Ensure prior parameters are tensors on the correct device
        prior_params_dtype_device = {'device': self.device, 'dtype': self.dtype}

        # _noise_prior = LogNormalPrior(
        #     hyperparams_dict['noise_prior_loc'].clone().detach().to(**prior_params_dtype_device),
        #     hyperparams_dict['noise_prior_scale'].clone().detach().to(**prior_params_dtype_device)
        # )
        likelihood = GaussianLikelihood(
            noise_constraint=Interval(1e-8, 1e-3,)
        ).to(self.device, self.dtype)

        _lengthscale_prior = LogNormalPrior(
            hyperparams_dict['lengthscale_prior_loc'].clone().detach().to(**prior_params_dtype_device),
            hyperparams_dict['lengthscale_prior_scale'].clone().detach().to(**prior_params_dtype_device)
        )
        _outputscale_prior = LogNormalPrior(
            hyperparams_dict['outputscale_prior_loc'].clone().detach().to(**prior_params_dtype_device),
            hyperparams_dict['outputscale_prior_scale'].clone().detach().to(**prior_params_dtype_device)
        )
        
        if not hasattr(self, 'gp_kernel_config') or self.gp_kernel_config is None:
            kernel_type = 'matern'
            self.gp_kernel_config = OmegaConf.create({'type': kernel_type, 'nu': 2.5})
        else:
            kernel_type = self.gp_kernel_config.get('type', 'matern').lower()
        if not hasattr(self, 'input_dim') or self.input_dim is None:
            self.input_dim = train_x.shape[-1] if train_x.ndim > 1 else 1
        if not hasattr(self, 'distance_metric') or self.distance_metric is None:
            self.distance_metric = 'hellinger'
        if not hasattr(self, 'dtype') or self.dtype is None:
            self.dtype = DEFAULT_DTYPE
        if not hasattr(self, 'device') or self.device is None:
            self.device = torch.device("cpu")

        if kernel_type == 'matern':
            base_kernel = MaternKernel(
                nu=self.gp_kernel_config.get('nu', 2.5),
                ard_num_dims=self.input_dim,
                lengthscale_prior=_lengthscale_prior
            )
        elif kernel_type == 'rbf':
            base_kernel = RBFKernel(
                ard_num_dims=self.input_dim,
                lengthscale_prior=_lengthscale_prior
            )
        else:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")
        covar_module = ScaleKernel(base_kernel, outputscale_prior=_outputscale_prior).to(self.device, self.dtype)

        mean_module = None
        if 'mean_module.constant' in hyperparams_dict:
            _mean_constant_prior = None
            if 'mean_constant_prior_loc' in hyperparams_dict and 'mean_constant_prior_scale' in hyperparams_dict:
                _mean_constant_prior = NormalPrior(
                    hyperparams_dict['mean_constant_prior_loc'].clone().detach().to(**prior_params_dtype_device),
                    hyperparams_dict['mean_constant_prior_scale'].clone().detach().to(**prior_params_dtype_device)
                )
            mean_module = gpytorch.means.ConstantMean(prior=_mean_constant_prior).to(self.device, self.dtype)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            temp_model = SingleTaskGP(
                train_X=train_x, train_Y=train_y,
                likelihood=likelihood, covar_module=covar_module, mean_module=mean_module
            ).to(self.device, self.dtype)

        temp_model.likelihood.noise = hyperparams_dict['likelihood.noise'].clone().detach().to(self.device, self.dtype)
        temp_model.covar_module.outputscale = hyperparams_dict['covar_module.outputscale'].clone().detach().to(self.device, self.dtype)
        temp_model.covar_module.base_kernel.lengthscale = hyperparams_dict['covar_module.base_kernel.lengthscale'].clone().detach().to(self.device, self.dtype)
        if 'mean_module.constant' in hyperparams_dict and hasattr(temp_model, "mean_module") and hasattr(temp_model.mean_module, "constant"):
             temp_model.mean_module.constant = hyperparams_dict['mean_module.constant'].clone().detach().to(self.device, self.dtype)
        temp_model.eval()
        return temp_model

    def _get_marginal_prediction(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.sampled_hyperparams:
            log.warning("No sampled hyperparameters for marginal prediction, using base model's direct prediction.")
            posterior = self.base_model.posterior(X)
            return posterior.mean.squeeze(-1), posterior.variance.squeeze(-1).clamp_min(1e-9)
        all_means, all_variances = [], []
        for hyperparams_dict in self.sampled_hyperparams:
            temp_model = self._reconstruct_gp_model(self.base_train_x, self.base_train_y, hyperparams_dict)
            posterior = temp_model.posterior(X)
            all_means.append(posterior.mean.squeeze(-1))
            all_variances.append(posterior.variance.squeeze(-1).clamp_min(1e-9))
        means_tensor = torch.stack(all_means, dim=0)
        variances_tensor = torch.stack(all_variances, dim=0)
        matched_means_list, matched_variances_list = [], []
        for i in range(X.shape[0]):
            m_mean, m_var = moment_match_gmm(means_tensor[:, i], variances_tensor[:, i])
            matched_means_list.append(m_mean)
            matched_variances_list.append(m_var)
        return torch.stack(matched_means_list), torch.stack(matched_variances_list)

    def _get_conditional_on_optimum_prediction(
        self, X_query: torch.Tensor, hyperparams_dict: Dict[str, Any],
        optimum_x_norm: torch.Tensor, optimum_f_stand: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        opt_x_formatted = optimum_x_norm.to(self.device, self.dtype).reshape(1, -1)
        opt_f_formatted = optimum_f_stand.to(self.device, self.dtype).reshape(1, 1)
        current_train_x = torch.cat([self.base_train_x, opt_x_formatted], dim=0)
        current_train_y = torch.cat([self.base_train_y, opt_f_formatted], dim=0)
        cond_model = self._reconstruct_gp_model(current_train_x, current_train_y, hyperparams_dict)
        posterior = cond_model.posterior(X_query)
        return posterior.mean.squeeze(-1), posterior.variance.squeeze(-1).clamp_min(1e-9)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.ndim == 3: X_actual = X.squeeze(1)
        elif X.ndim == 2: X_actual = X
        else:
            log.error(f"Unexpected shape for X in SCoreBOAcquisitionFunction: {X.shape}")
            return torch.zeros(X.shape[0] if X.ndim > 0 else 1, device=self.device, dtype=self.dtype)
        if self.num_hyperparameter_samples_M == 0 or self.num_optima_samples_N == 0:
            return torch.zeros(X_actual.shape[0], device=self.device, dtype=self.dtype)
        marginal_means, marginal_variances = self._get_marginal_prediction(X_actual)
        all_distances_for_avg = []
        for m_idx, hyperparams_m in enumerate(self.sampled_hyperparams):
            optima_list_for_m = self.sampled_optima_per_hyperparam[m_idx]
            for n_idx, (opt_x_mn_norm, opt_f_mn_stand) in enumerate(optima_list_for_m):
                cond_means, cond_variances = self._get_conditional_on_optimum_prediction(
                    X_actual, hyperparams_m, opt_x_mn_norm, opt_f_mn_stand)
                current_distances_batch = []
                for i in range(X_actual.shape[0]):
                    dist_val = torch.tensor(0.0, device=self.device, dtype=self.dtype)
                    if self.distance_metric == "hellinger":
                        dist_val = hellinger_distance_gaussian(marginal_means[i], marginal_variances[i], cond_means[i], cond_variances[i])
                    elif self.distance_metric == "wasserstein":
                        dist_val = torch.sqrt(torch.clamp(wasserstein2_distance_gaussian(marginal_means[i], marginal_variances[i], cond_means[i], cond_variances[i]), min=1e-9))
                    elif self.distance_metric == "kl_divergence":
                        dist_val = kl_divergence_gaussian(cond_means[i], cond_variances[i], marginal_means[i], marginal_variances[i])
                    else: raise ValueError(f"Unknown distance metric: {self.distance_metric}")
                    current_distances_batch.append(dist_val)
                all_distances_for_avg.append(torch.stack(current_distances_batch))
        if not all_distances_for_avg: return torch.zeros(X_actual.shape[0], device=self.device, dtype=self.dtype)
        distances_tensor = torch.stack(all_distances_for_avg, dim=0)
        acq_values = torch.mean(distances_tensor, dim=0)
        return acq_values
    

class SelfCorrectingBayesianOptimization(BaseBayesianOptimizer):
    def __init__(self, config: DictConfig, objective_function):
        super().__init__(config, objective_function)

        self.bo_config = config.bo
        self.gp_config = config.get('gp', OmegaConf.create()) # Handle missing gp config
        self.acq_config = config.get('acquisition', OmegaConf.create())
        self.scorebo_config = config.get('scorebo', OmegaConf.create())

        self.num_hyperparameter_samples_M = self.scorebo_config.get('num_hyperparameter_samples_M', 16)
        self.num_optima_samples_N = self.scorebo_config.get('num_optima_samples_N', 8)
        self.statistical_distance = self.scorebo_config.get('statistical_distance', 'hellinger')
        # self.approximation_method = self.scorebo_config.get('approximation_method', 'moment_matching') # Currently only MM

        self.model: Optional[SingleTaskGP] = None
        self.mll: Optional[ExactMarginalLogLikelihood] = None
        self.sampled_hyperparams_list: List[Dict[str, Any]] = []
        self.sampled_optima_per_hyperparam_list: List[List[Tuple[torch.Tensor, torch.Tensor]]] = []

        self.train_x_gp: Optional[torch.Tensor] = None # Normalized X for GP
        self.train_y_gp: Optional[torch.Tensor] = None # Standardized Y for GP
        self.y_mean: Optional[torch.Tensor] = None
        self.y_std: Optional[torch.Tensor] = None

        log.info(f"SCoreBO configured: M={self.num_hyperparameter_samples_M}, N={self.num_optima_samples_N}, "
                 f"Distance={self.statistical_distance}, Approx=MomentMatching")
        log.warning("SCoreBO Full MCMC hyperparameter sampling is complex; current impl uses fit_gpytorch_mll (MAP-like) "
                    "and uses MAP estimates as stand-ins for posterior samples.")


    def _get_hyperparameter_priors(self) -> Dict[str, gpytorch.priors.Prior]:
        priors = {}
        # Ensure prior parameters are tensors on the correct device
        prior_loc_std_dtype = {'device': self.device, 'dtype': self.dtype}

        default_prior_loc = torch.tensor(self.gp_config.get('default_prior_loc', 0.0), **prior_loc_std_dtype)
        default_prior_scale = torch.tensor(self.gp_config.get('default_prior_scale', 1.732), **prior_loc_std_dtype) # sqrt(3)

        priors['lengthscale'] = LogNormalPrior(
            torch.tensor(self.gp_config.get('lengthscale_prior_mean', default_prior_loc), **prior_loc_std_dtype),
            torch.tensor(self.gp_config.get('lengthscale_prior_std', default_prior_scale), **prior_loc_std_dtype)
        )
        priors['outputscale'] = LogNormalPrior(
            torch.tensor(self.gp_config.get('outputscale_prior_mean', default_prior_loc), **prior_loc_std_dtype),
            torch.tensor(self.gp_config.get('outputscale_prior_std', default_prior_scale), **prior_loc_std_dtype)
        )
        priors['noise'] = LogNormalPrior(
            torch.tensor(self.gp_config.get('noise_prior_mean', default_prior_loc), **prior_loc_std_dtype),
            torch.tensor(self.gp_config.get('noise_prior_std', default_prior_scale), **prior_loc_std_dtype)
        )
        if self.gp_config.get('learn_mean_constant', False):
             priors['mean_constant'] = NormalPrior(
                 torch.tensor(self.gp_config.get('mean_constant_prior_loc', 0.0), **prior_loc_std_dtype),
                 torch.tensor(self.gp_config.get('mean_constant_prior_scale', 1.0), **prior_loc_std_dtype)
             )
        return priors

    def _fit_gp_model(self, train_x_norm: torch.Tensor, train_y_raw: torch.Tensor) -> Tuple[SingleTaskGP, ExactMarginalLogLikelihood]:
        y_squeezed = train_y_raw.squeeze(-1) if train_y_raw.ndim > 1 else train_y_raw
        self.y_mean = y_squeezed.mean()
        self.y_std = y_squeezed.std().clamp(min=1e-6)
        current_train_y_gp = ((y_squeezed - self.y_mean) / self.y_std).unsqueeze(-1)
        # Store standardized Y for use in acquisition function context
        # self.train_y_gp should reflect the Y the *current main model* is trained on
        self.train_y_gp = current_train_y_gp

        gp_priors = self._get_hyperparameter_priors()
        likelihood = GaussianLikelihood(
            noise_prior=gp_priors['noise'],
            noise_constraint=Interval(self.gp_config.get("noise_constraint_low", 1e-8), self.gp_config.get("noise_constraint_high", 1e-3))
        ).to(self.device, self.dtype)

        kernel_type = self.gp_config.get('kernel_type', 'matern').lower()
        if kernel_type == 'matern':
            base_kernel = MaternKernel(
                nu=self.gp_config.get('matern_nu', 2.5),
                ard_num_dims=train_x_norm.shape[-1],
                lengthscale_prior=gp_priors['lengthscale'],
                lengthscale_constraint=Interval(0.005, 4.0)
            )
        elif kernel_type == 'rbf':
             base_kernel = RBFKernel(
                ard_num_dims=train_x_norm.shape[-1],
                lengthscale_prior=gp_priors['lengthscale'],
                lengthscale_constraint=Interval(0.005, 4.0)
            )
        else:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")

        covar_module = ScaleKernel(base_kernel, outputscale_prior=gp_priors['outputscale']).to(self.device, self.dtype)
        
        mean_module = None
        if self.gp_config.get('learn_mean_constant', False):
            mean_module = gpytorch.means.ConstantMean(prior=gp_priors.get('mean_constant')).to(self.device, self.dtype)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SingleTaskGP(
                train_X=train_x_norm, train_Y=current_train_y_gp,
                likelihood=likelihood, covar_module=covar_module, mean_module=mean_module
            ).to(self.device, self.dtype)
        mll = ExactMarginalLogLikelihood(model.likelihood, model).to(self.device, self.dtype)

        try:
            fit_gpytorch_mll(mll, max_retries=self.gp_config.get("fit_max_retries", 5),
                             options={"maxiter": self.gp_config.get("train_iter", 100)})
        except Exception as e:
            log.error(f"GP fitting failed: {e}", exc_info=True)
            raise RuntimeError("GP model fitting failed in _fit_gp_model.") from e
        model.eval()
        return model, mll

    def _sample_hyperparameters_from_model(self, model: SingleTaskGP, mll: ExactMarginalLogLikelihood, num_samples: int) -> List[Dict[str, Any]]:
        log.warning("Using simplified hyperparameter sampling (MAP point). True SCoreBO requires MCMC (e.g., NUTS).")
        
        # Helper to safely get prior parameters or defaults
        def get_prior_params(prior_obj, default_loc=0.0, default_scale=1.0):
            # Convert to float for JSON serializability if configs are saved/logged extensively
            # For internal tensor use, keep as tensor. For dict storage, float is fine if that's the only use.
            loc = prior_obj.loc.item() if hasattr(prior_obj, 'loc') and prior_obj.loc is not None else default_loc
            scale = prior_obj.scale.item() if hasattr(prior_obj, 'scale') and prior_obj.scale is not None else default_scale
            return torch.tensor(loc, device=self.device, dtype=self.dtype), torch.tensor(scale, device=self.device, dtype=self.dtype)

        # noise_loc, noise_scale = get_prior_params(model.likelihood.noise_prior)
        outputscale_loc, outputscale_scale = get_prior_params(model.covar_module.outputscale_prior)
        lengthscale_loc, lengthscale_scale = get_prior_params(model.covar_module.base_kernel.lengthscale_prior)

        map_hyperparams = {
            "likelihood.noise": model.likelihood.noise.detach().clone(),
            "covar_module.outputscale": model.covar_module.outputscale.detach().clone(),
            "covar_module.base_kernel.lengthscale": model.covar_module.base_kernel.lengthscale.detach().clone(),
            # "noise_prior_loc": noise_loc, "noise_prior_scale": noise_scale,
            "outputscale_prior_loc": outputscale_loc, "outputscale_prior_scale": outputscale_scale,
            "lengthscale_prior_loc": lengthscale_loc, "lengthscale_prior_scale": lengthscale_scale,
        }
        if hasattr(model, "mean_module") and hasattr(model.mean_module, "constant") and model.mean_module.constant is not None:
             map_hyperparams["mean_module.constant"] = model.mean_module.constant.detach().clone()
        
        return [map_hyperparams for _ in range(num_samples)] # Return M copies of MAP

    def _initialize_models(self):
        if self.data_x is None or self.data_y is None or self.data_x.shape[0] == 0:
            log.error("Initial data not available for SCoreBO model initialization.")
            raise ValueError("Initial data required.")
        self.train_x_gp = normalize(self.data_x, self.bounds.reshape([2,-1]))
        log.info("SCoreBO: Initializing base GP model...")
        self.model, self.mll = self._fit_gp_model(self.train_x_gp, self.data_y)
        log.info(f"SCoreBO: Sampling {self.num_hyperparameter_samples_M} hyperparameter sets (using MAP estimates)...")
        self.sampled_hyperparams_list = self._sample_hyperparameters_from_model(self.model, self.mll, self.num_hyperparameter_samples_M)
        self._sample_optima_for_hyperparams()
        log.info(f"SCoreBO: Initial model setup complete.")

    def _sample_optima_for_hyperparams(self):
        self.sampled_optima_per_hyperparam_list = []
        if not self.sampled_hyperparams_list:
            log.warning("No hyperparameters sampled, cannot sample optima.")
            return

        # log.info(f"SCoreBO: Sampling {self.num_optima_samples_N} optima for each of {self.num_hyperparameter_samples_M} hyperparameter sets...")
        for hyperparams_m_dict in tqdm(self.sampled_hyperparams_list, desc="Sampling Optima", leave=False, disable=not self.bo_config.get('verbose',True)):
            # Reconstruct model with current hyperparams_m_dict, trained on (self.train_x_gp, self.train_y_gp)
            # self.train_y_gp is already standardized
            temp_model_for_sampling = SCoreBOAcquisitionFunction._reconstruct_gp_model(
                self, self.train_x_gp, self.train_y_gp, hyperparams_m_dict
            )
            optima_for_current_hyperparams = []
            try:
                # MaxPosteriorSampling samples X candidates that maximize some posterior draw.
                # It needs a model whose posterior it can sample from.
                sampler = MaxPosteriorSampling(model=temp_model_for_sampling, replacement=False)
                normalized_bounds_for_mps = torch.tensor([[0.0] * self.input_dim, [1.0] * self.input_dim], device=self.device, dtype=self.dtype)
                X_cand_norm = draw_sobol_samples(bounds=normalized_bounds_for_mps, n=1024, q=1).squeeze(1)

                sampled_x_star_normalized = sampler(X_cand_norm, num_samples=self.num_optima_samples_N)
                posterior_at_x_star = temp_model_for_sampling.posterior(sampled_x_star_normalized)
                sampled_f_star_standardized = posterior_at_x_star.mean # (N, 1)

                for i in range(self.num_optima_samples_N):
                    opt_x_norm = sampled_x_star_normalized[i].unsqueeze(0)
                    opt_f_stand = sampled_f_star_standardized[i].unsqueeze(0)
                    optima_for_current_hyperparams.append((opt_x_norm, opt_f_stand))
            except Exception as e:
                log.error(f"Error during optima sampling for one hyperparameter set: {e}", exc_info=True)
                for _ in range(self.num_optima_samples_N): # Fallback
                     dummy_x = torch.rand(1, self.input_dim, device=self.device, dtype=self.dtype)
                     dummy_f = torch.tensor([[self.train_y_gp.mean().item()]], device=self.device, dtype=self.dtype)
                     optima_for_current_hyperparams.append((dummy_x, dummy_f))
            self.sampled_optima_per_hyperparam_list.append(optima_for_current_hyperparams)

    def _update_models(self):
        if self.data_x is None or self.data_y is None or self.data_x.shape[0] == 0: return
        self.train_x_gp = normalize(self.data_x, self.bounds.reshape([2,-1]))
        #log.debug("SCoreBO: Updating base GP model...")
        self.model, self.mll = self._fit_gp_model(self.train_x_gp, self.data_y)
        #log.debug(f"SCoreBO: Re-sampling {self.num_hyperparameter_samples_M} hyperparameter sets (MAP estimates)...")
        self.sampled_hyperparams_list = self._sample_hyperparameters_from_model(self.model, self.mll, self.num_hyperparameter_samples_M)
        self._sample_optima_for_hyperparams()
        #log.debug("SCoreBO: Model update and re-sampling complete.")

    def _propose_next_candidate(self) -> torch.Tensor:
        if not self.model or not self.sampled_hyperparams_list or not self.sampled_optima_per_hyperparam_list or \
           len(self.sampled_optima_per_hyperparam_list) == 0 or len(self.sampled_optima_per_hyperparam_list[0]) == 0:
            log.error("SCoreBO models/samples not initialized adequately. Cannot propose candidate.")
            return torch.rand(1, self.input_dim, device=self.device, dtype=self.dtype)

        sc_acqf = SCoreBOAcquisitionFunction(
            model=self.model, base_train_x=self.train_x_gp, base_train_y=self.train_y_gp,
            sampled_hyperparams=self.sampled_hyperparams_list,
            sampled_optima_per_hyperparam=self.sampled_optima_per_hyperparam_list,
            input_dim = self.input_dim,
            distance_metric=self.statistical_distance,
            gp_kernel_config=self.gp_config.get('kernel', OmegaConf.create({'type':'matern', 'nu':2.5})), # Pass kernel config
            device=self.device, dtype=self.dtype
        )
        self.acq = sc_acqf
        normalized_bounds = torch.tensor([[0.0] * self.input_dim, [1.0] * self.input_dim], device=self.device, dtype=self.dtype)
        optimizer_type = self.acq_config.get("optimizer_type", "continuous").lower()

        try:
            if optimizer_type == "discrete":
                num_choices = self.acq_config.get("num_discrete_choices", 2000)
                # Ensure bounds for draw_sobol_samples are [2, dim]
                discrete_choices_norm = draw_sobol_samples(bounds=normalized_bounds, n=num_choices, q=1).squeeze(1)
                if discrete_choices_norm.shape[0] == 0: # Fallback if no choices generated
                    log.warning("Generated 0 discrete choices for acquisition function optimization. Proposing random.")
                    return torch.rand(1, self.input_dim, device=self.device, dtype=self.dtype)

                candidates_normalized, acq_value = optimize_acqf_discrete(
                    acq_function=sc_acqf,
                    q=1,
                    choices=discrete_choices_norm,
                    return_best_only=True,
                )
                log.debug(f"SCoreBO discrete acq value: {acq_value.item():.4e} from {num_choices} choices.")

            else: # Default to continuous
                candidates_normalized, acq_value = optimize_acqf(
                    acq_function=sc_acqf, bounds=normalized_bounds, q=1,
                    num_restarts=self.acq_config.get("opt_restarts", 10),
                    raw_samples=self.acq_config.get("opt_samples", 512),
                    options=self.acq_config.get("optimizer_options", {"batch_limit": 5, "maxiter": 200}),
                    return_best_only=True,
                )
        except Exception as e:
            log.error(f"Error optimizing SCoreBO acquisition function: {e}", exc_info=True)
            return torch.rand(1, self.input_dim, device=self.device, dtype=self.dtype) # Fallback

        #log.debug(f"SCoreBO proposed candidate (normalized): {candidates_normalized.cpu().numpy()}, Acq value: {acq_value.item():.4f}")
        return candidates_normalized.detach()

    def run_optimization(self) -> dict:
        start_time = time.time()
        # Ensure optimization_history is initialized here if not in Base
        self.optimization_history = {'all_x': [], 'all_y': [], 'timestamps': []}
        if 'acquisition_values' not in self.optimization_history:
            self.optimization_history['acquisition_values'] = []

        self.sample_initial_points()
        if self.data_x is None:
            log.error("Initial sampling failed.")
            return {'error': 'Initial sampling failed', 'best_x': np.array([]), 'best_y': np.array([]), 'all_x': np.array([]), 'all_y': np.array([]), 'total_time': 0, 'optimization_history': self.optimization_history}

        log.info(f"Collected {self.data_x.shape[0]} initial points.")
        self.optimization_history['all_x'].append(self.data_x.clone().cpu())
        self.optimization_history['all_y'].append(self.data_y.clone().cpu())
        current_ts_val = time.time() - start_time
        ts_shape = self.data_y.squeeze(-1).shape if self.data_y.ndim > 1 else self.data_y.shape
        self.optimization_history['timestamps'].append(torch.full(ts_shape, current_ts_val, device='cpu', dtype=self.dtype).unsqueeze(-1))


        try:
            self._initialize_models()
        except Exception as e:
            log.error(f"SCoreBO model initialization failed: {e}", exc_info=True)
            return {'error': f'SCoreBO model initialization failed: {e}', 'best_x': self.data_x.cpu().numpy(), 'best_y': self.data_y.cpu().numpy(), 'all_x': self.data_x.cpu().numpy(), 'all_y': self.data_y.cpu().numpy(), 'total_time': time.time()-start_time, 'optimization_history': self.optimization_history}


        num_iterations_to_run = self.bo_config.max_iterations
        desc = f"SCoreBO Iter ({self.statistical_distance[0].upper()})"
        iterator = tqdm(range(num_iterations_to_run), desc=desc, unit="iter", disable=not self.bo_config.get('verbose',True))
        current_total_evals = self.data_x.shape[0] # Already includes initial points

        for i in iterator:
            if current_total_evals >= self.bo_config.initial_points + self.bo_config.max_iterations:
                log.info("Reached maximum number of evaluations.")
                break
            try:
                self._update_models()
                X_next_normalized = self._propose_next_candidate()
            except Exception as e:
                log.error(f"Iteration {i}: Failed during model update or proposal: {e}", exc_info=True)
                break
            X_next = unnormalize(X_next_normalized, bounds=self.bounds.reshape([2,-1]))
            Y_next = self.objective_function(X_next)
            self.data_x = torch.cat((self.data_x, X_next), dim=0)
            self.data_y = torch.cat((self.data_y, Y_next), dim=0)
            current_total_evals += X_next.shape[0]

            self.optimization_history['all_x'].append(X_next.clone().cpu())
            self.optimization_history['all_y'].append(Y_next.clone().cpu())
            current_ts_val = time.time() - start_time
            ts_shape_next = Y_next.squeeze(-1).shape if Y_next.ndim > 1 else Y_next.shape
            self.optimization_history['timestamps'].append(torch.full(ts_shape_next, current_ts_val, device='cpu', dtype=self.dtype).unsqueeze(-1))
            self.optimization_history['acquisition_values'].append(float('nan'))

            if self.bo_config.get('verbose',True):
                current_best_y = self.data_y.max().item()
                iterator.set_postfix(best_y=f"{current_best_y:.4f}")

        total_time = time.time() - start_time
        log.info(f"SCoreBO optimization finished after {total_time:.2f} seconds and {current_total_evals} evaluations.")
        
        # Consolidate history from list of tensors to single tensors before returning
        final_all_x = torch.cat(self.optimization_history['all_x'], dim=0).cpu().numpy()
        final_all_y = torch.cat(self.optimization_history['all_y'], dim=0).cpu().numpy()
        final_timestamps = torch.cat(self.optimization_history['timestamps'], dim=0).cpu().numpy()

        # Update optimization_history dict with consolidated numpy arrays
        self.optimization_history['all_x'] = final_all_x
        self.optimization_history['all_y'] = final_all_y
        self.optimization_history['timestamps'] = final_timestamps
        self.optimization_history['acquisition_values'] = np.array(self.optimization_history['acquisition_values'])


        best_y_overall_val, best_idx = torch.from_numpy(final_all_y).max(dim=0)
        best_x_overall_val = torch.from_numpy(final_all_x)[best_idx]

        results = {
            'best_x': best_x_overall_val.cpu().numpy(),
            'best_y': best_y_overall_val.cpu().numpy(),
            'all_x': final_all_x,
            'all_y': final_all_y,
            'total_time': total_time,
            'optimization_history': self.optimization_history # Now contains numpy arrays
        }
        return results