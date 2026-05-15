"""
gp_ensemble.py

Unified GP ensemble factory with a kernel_type dispatch.

Public API
----------
build_gp_ensemble(train_x, train_y, M, ..., kernel_type='rbf_isotropic', dkl_*)
    Returns a list of M dicts: {'model': ..., 'likelihood': ..., 'id': int}
    that is drop-in compatible with dro.py's self.gp_ensemble.

kernel_type options
-------------------
'rbf_isotropic'  Original isotropic RBF ensemble — behaviour UNCHANGED.
'rbf_ard'        ARD RBF ensemble (one lengthscale per input dimension).
'deep_kernel'    DeepKernelGP ensemble sharing a single trained FeatureExtractor.

Default is 'rbf_isotropic' so all existing experiment scripts run identically
without modification.
"""

import warnings
import numpy as np
import torch
import gpytorch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.constraints import GreaterThan
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP
from botorch.exceptions import InputDataWarning

from src.model.deep_kernel_gp import build_deep_kernel_ensemble


def build_gp_ensemble(
    train_x,
    train_y,
    M: int = 5,
    kernel_type: str = "rbf_isotropic",
    # shared RBF / ARD params
    ls_min: float = 0.1,
    ls_max: float = 10.0,
    noise_constraint: float = 1e-6,
    train_iter: int = 50,
    lr: float = 0.1,
    device="cpu",
    dtype=torch.float64,
    # deep-kernel-specific params (dkl_*)
    dkl_hidden_dim: int = 32,
    dkl_latent_dim=None,
    dkl_n_iter: int = 100,
    dkl_lr: float = 0.01,
    dkl_weight_decay: float = 1e-2,
    dkl_warm_start_models=None,
    collect_diagnostics: bool = False,
):
    """Build an ensemble of M GP models and return them in dro.py's dict format.

    Parameters
    ----------
    kernel_type : str
        'rbf_isotropic', 'rbf_ard', or 'deep_kernel'.
    dkl_warm_start_models : list[(model, likelihood)] or None
        If given (deep_kernel only), skips anchor training and runs only
        n_iter // 2 fine-tuning steps per model.

    Returns
    -------
    list of dicts  [{'model': ..., 'likelihood': ..., 'id': i}, ...]
    """
    train_x = train_x.to(device=device, dtype=dtype)
    train_y = train_y.to(device=device, dtype=dtype)

    # ── Deep kernel path ──────────────────────────────────────────────────────
    if kernel_type == "deep_kernel":
        result = build_deep_kernel_ensemble(
            train_x,
            train_y,
            M=M,
            hidden_dim=dkl_hidden_dim,
            latent_dim=dkl_latent_dim,
            n_iter=dkl_n_iter,
            lr=dkl_lr,
            weight_decay=dkl_weight_decay,
            warm_start_models=dkl_warm_start_models,
            collect_diagnostics=collect_diagnostics,
        )
        if collect_diagnostics:
            pairs, diag = result
        else:
            pairs, diag = result, {}
        ensemble = [{"model": m, "likelihood": lik, "id": i}
                    for i, (m, lik) in enumerate(pairs)]
        return (ensemble, diag) if collect_diagnostics else ensemble

    # ── RBF isotropic / ARD path (original behaviour, unchanged) ─────────────
    input_dim = train_x.shape[-1]
    use_ard = kernel_type == "rbf_ard"
    ard_num_dims = input_dim if use_ard else None
    initial_ls = np.linspace(ls_min, ls_max, M)
    ensemble = []

    for i, ls in enumerate(initial_ls):
        likelihood = GaussianLikelihood(noise_constraint=GreaterThan(noise_constraint))
        covar_module = ScaleKernel(RBFKernel(ard_num_dims=ard_num_dims))
        covar_module.base_kernel.initialize(lengthscale=float(ls))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=InputDataWarning)
            model = SingleTaskGP(
                train_X=train_x,
                train_Y=train_y.reshape(-1, 1),
                likelihood=likelihood,
                covar_module=covar_module,
            ).to(device=device, dtype=dtype)

        model.train()
        likelihood.train()
        mll = ExactMarginalLogLikelihood(likelihood, model)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(train_iter):
            opt.zero_grad()
            loss = -mll(model(train_x), train_y)
            loss.backward()
            opt.step()

        model.eval()
        likelihood.eval()
        ensemble.append({"model": model, "likelihood": likelihood, "id": i})

    return ensemble
