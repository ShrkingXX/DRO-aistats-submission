"""
test_deep_kernel_gp.py

Standalone smoke-test for deep_kernel_gp.py.

Checks:
  1. Posterior mean is finite for 5 test points (per model).
  2. Posterior variance is finite for 5 test points (per model).
  3. Posterior variance > 0 for 5 test points (per model).
Prints PASS / FAIL for each check and the final anchor training loss.
"""

import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from gpytorch.likelihoods import GaussianLikelihood

from src.model.deep_kernel_gp import (
    DeepKernelGP,
    build_deep_kernel_ensemble,
    train_deep_kernel_gp,
)

torch.manual_seed(42)

# ── Synthetic dataset: 20 points in 2D, y = sin(x1) + cos(x2) + 0.1*randn ──
N = 20
train_x = torch.randn(N, 2)
train_y = torch.sin(train_x[:, 0]) + torch.cos(train_x[:, 1]) + 0.1 * torch.randn(N)

# ── Single-model training loss ────────────────────────────────────────────────
print("Training single model to obtain final loss ...")
_lik = GaussianLikelihood()
_model = DeepKernelGP(train_x, train_y, _lik, input_dim=2)
_, _, losses = train_deep_kernel_gp(_model, _lik, train_x, train_y, n_iter=100)
print(f"Final training loss (single model, 100 iter): {losses[-1]:.4f}\n")

# ── Build ensemble of M=3 models ──────────────────────────────────────────────
print("Building ensemble (M=3) ...")
ensemble = build_deep_kernel_ensemble(train_x, train_y, M=3, n_iter=100)
print(f"Ensemble built: {len(ensemble)} models\n")

# ── 5 test points ─────────────────────────────────────────────────────────────
test_x = torch.randn(5, 2)

all_pass = True

for k, (model, likelihood) in enumerate(ensemble):
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        pred = likelihood(model(test_x))
        mean = pred.mean        # [5]
        var  = pred.variance    # [5]

    finite_mean = torch.all(torch.isfinite(mean)).item()
    finite_var  = torch.all(torch.isfinite(var)).item()
    pos_var     = torch.all(var > 0).item()

    s_mean = "PASS" if finite_mean else "FAIL"
    s_varf = "PASS" if finite_var  else "FAIL"
    s_varp = "PASS" if pos_var     else "FAIL"

    if not (finite_mean and finite_var and pos_var):
        all_pass = False

    print(f"Model {k}:  mean finite={s_mean}  |  var finite={s_varf}  |  var>0={s_varp}")
    print(f"         mean={mean.tolist()}")
    print(f"         var ={var.tolist()}")

print()
print("Overall:", "PASS" if all_pass else "FAIL")
