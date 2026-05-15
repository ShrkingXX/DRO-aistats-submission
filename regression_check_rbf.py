"""
regression_check_rbf.py

Verify that build_gp_ensemble('rbf_isotropic') produces numerically identical
GP predictions to the original direct SingleTaskGP construction used in
gp_bo_baseline.py.

The two paths share the same random seed, data, Adam hyperparameters, and
initial lengthscales — so their outputs must be bit-for-bit identical.

Prints:
    REGRESSION CHECK: PASS   — if all 10 seeds agree within 1e-6 atol
    REGRESSION CHECK: FAIL   — with details otherwise
"""

import sys
import os
import warnings

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
import gpytorch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.constraints import GreaterThan
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP
from botorch.exceptions import InputDataWarning

from src.model.gp_ensemble import build_gp_ensemble

DTYPE = torch.float64
M = 5
LS_MIN, LS_MAX = 0.1, 10.0
NOISE_CONSTRAINT = 1e-6
TRAIN_ITER = 50
LR = 0.1
N_SEEDS = 10
N_TRAIN = 20
DIM = 2
N_TEST = 50
ATOL = 1e-6


def _reference_ensemble(train_x, train_y):
    """Exact copy of gp_bo_baseline._build_ensemble() — the gold standard."""
    initial_ls = np.linspace(LS_MIN, LS_MAX, M)
    ensemble = []
    for ls in initial_ls:
        likelihood = GaussianLikelihood(noise_constraint=GreaterThan(NOISE_CONSTRAINT))
        covar_module = ScaleKernel(RBFKernel())
        covar_module.base_kernel.initialize(lengthscale=float(ls))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=InputDataWarning)
            model = SingleTaskGP(
                train_X=train_x,
                train_Y=train_y.reshape(-1, 1),
                likelihood=likelihood,
                covar_module=covar_module,
            ).to(dtype=DTYPE)
        model.train(); likelihood.train()
        mll = ExactMarginalLogLikelihood(likelihood, model)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        for _ in range(TRAIN_ITER):
            opt.zero_grad()
            loss = -mll(model(train_x), train_y)
            loss.backward()
            opt.step()
        model.eval(); likelihood.eval()
        ensemble.append(model)
    return ensemble


def _predict_reference(ensemble, test_x):
    """Posterior mean + variance from the reference SingleTaskGP ensemble."""
    means, vars_ = [], []
    for model in ensemble:
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            post = model.posterior(test_x)
            means.append(post.mean.squeeze(-1))   # [N_TEST]
            vars_.append(post.variance.squeeze(-1))
    return torch.stack(means), torch.stack(vars_)   # [M, N_TEST]


def _predict_new(gp_ensemble, test_x):
    """Posterior mean + variance from build_gp_ensemble() output."""
    means, vars_ = [], []
    for gp in gp_ensemble:
        model = gp["model"]
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            post = model.posterior(test_x)
            means.append(post.mean.squeeze(-1))
            vars_.append(post.variance.squeeze(-1))
    return torch.stack(means), torch.stack(vars_)   # [M, N_TEST]


def run_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Synthetic training data: Ackley-like
    train_x = torch.randn(N_TRAIN, DIM, dtype=DTYPE)
    train_y = torch.sin(train_x[:, 0]) + torch.cos(train_x[:, 1]) + 0.05 * torch.randn(N_TRAIN, dtype=DTYPE)
    test_x = torch.randn(N_TEST, DIM, dtype=DTYPE)

    # --- Reference path (original code, fixed seed) ---
    torch.manual_seed(seed)
    ref_ensemble = _reference_ensemble(train_x, train_y)
    ref_mean, ref_var = _predict_reference(ref_ensemble, test_x)

    # --- New path (build_gp_ensemble, same fixed seed) ---
    torch.manual_seed(seed)
    new_ensemble = build_gp_ensemble(
        train_x, train_y, M=M,
        kernel_type="rbf_isotropic",
        ls_min=LS_MIN, ls_max=LS_MAX,
        noise_constraint=NOISE_CONSTRAINT,
        train_iter=TRAIN_ITER,
        lr=LR,
        dtype=DTYPE,
    )
    new_mean, new_var = _predict_new(new_ensemble, test_x)

    mean_ok = torch.allclose(ref_mean, new_mean, atol=ATOL)
    var_ok  = torch.allclose(ref_var,  new_var,  atol=ATOL)

    if not mean_ok or not var_ok:
        max_mean_err = (ref_mean - new_mean).abs().max().item()
        max_var_err  = (ref_var  - new_var ).abs().max().item()
        return False, f"seed={seed}  max_mean_err={max_mean_err:.2e}  max_var_err={max_var_err:.2e}"
    return True, f"seed={seed}  OK"


def main():
    print(f"Regression check: build_gp_ensemble('rbf_isotropic') vs reference")
    print(f"  M={M}  N_train={N_TRAIN}  D={DIM}  N_test={N_TEST}  seeds={N_SEEDS}\n")

    all_pass = True
    for seed in range(N_SEEDS):
        ok, msg = run_seed(seed)
        print(f"  [{('PASS' if ok else 'FAIL')}]  {msg}")
        if not ok:
            all_pass = False

    print()
    print(f"REGRESSION CHECK: {'PASS' if all_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
