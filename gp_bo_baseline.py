"""
gp_bo_baseline.py

Standard Bayesian Optimization with logEI acquisition — no Decision Transformer.
Uses the same M=5 GP ensemble (isotropic RBF, same initialization procedure) as
the DRO experiments, isolating the contribution of the DT component.

At each BO iteration the logEI is optimised independently under each of the 5 GP
models; the candidate with the highest logEI across all models is selected as the
next query point (committee/ensemble acquisition).

Usage
-----
    python gp_bo_baseline.py                          # defaults (all 3 benchmarks)
    python gp_bo_baseline.py --objective Levy
    python gp_bo_baseline.py --num_trials 5 --seed_start 10
"""

import argparse
import os
import sys
import warnings
import json
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.objectives import Ackley, Rosenbrock, Levy

import gpytorch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels  import ScaleKernel, RBFKernel
from gpytorch.constraints import GreaterThan
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.exceptions import InputDataWarning

DEFAULT_DTYPE  = torch.float64
torch.set_default_dtype(DEFAULT_DTYPE)

TRUE_OPTIMA = {"Ackley": 0.0, "Rosenbrock": 0.0, "Levy": 0.0}

# ── GP ensemble helpers ──────────────────────────────────────────────────────

def _build_ensemble(train_x, train_y, num_models=5,
                    ls_min=0.1, ls_max=10.0,
                    noise_constraint=1e-4,
                    device="cpu", dtype=torch.float64):
    """Initialise and fit M GP models with linspace-spaced initial lengthscales."""
    initial_ls = np.linspace(ls_min, ls_max, num_models)
    ensemble = []
    for ls in initial_ls:
        likelihood   = GaussianLikelihood(noise_constraint=GreaterThan(noise_constraint))
        base_kernel  = RBFKernel()
        covar_module = ScaleKernel(base_kernel)
        covar_module.base_kernel.initialize(lengthscale=float(ls))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=InputDataWarning)
            model = SingleTaskGP(
                train_X=train_x,
                train_Y=train_y.reshape(-1, 1),
                likelihood=likelihood,
                covar_module=covar_module,
            ).to(device=device, dtype=dtype)
        # Fit with Adam (same as DRO)
        model.train(); likelihood.train()
        mll = ExactMarginalLogLikelihood(likelihood, model)
        opt = torch.optim.Adam(model.parameters(), lr=0.1)
        for _ in range(50):
            opt.zero_grad()
            loss = -mll(model(train_x), train_y)
            loss.backward()
            opt.step()
        model.eval(); likelihood.eval()
        ensemble.append(model)
    return ensemble


def _refit_ensemble(ensemble, train_x, train_y):
    """Refit all models in-place with updated data."""
    for model in ensemble:
        model.set_train_data(train_x, train_y.squeeze(-1), strict=False)
        model.train(); model.likelihood.train()
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        opt = torch.optim.Adam(model.parameters(), lr=0.1)
        for _ in range(50):
            opt.zero_grad()
            loss = -mll(model(train_x), train_y.squeeze(-1))
            loss.backward()
            opt.step()
        model.eval(); model.likelihood.eval()


def _ensemble_logEI_candidate(ensemble, bounds, best_f, device, dtype):
    """
    For each GP in the ensemble optimise logEI and collect (candidate, logEI).
    Return the candidate with the highest logEI across all models.
    """
    best_val  = -float("inf")
    best_cand = None
    for model in ensemble:
        acq = LogExpectedImprovement(model=model, best_f=best_f)
        try:
            cand, val = optimize_acqf(
                acq_function=acq,
                bounds=bounds,
                q=1,
                num_restarts=5,
                raw_samples=64,
            )
            if val.item() > best_val:
                best_val  = val.item()
                best_cand = cand
        except Exception:
            continue
    if best_cand is None:
        # Fallback: random sample within bounds
        d = bounds.shape[1]
        best_cand = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(1, d, dtype=dtype, device=device)
    return best_cand.squeeze(0)   # [D]


# ── Single trial ─────────────────────────────────────────────────────────────

def run_trial(objective, bounds, initial_points, max_iterations,
              seed, device="cpu", dtype=torch.float64):
    torch.manual_seed(seed); np.random.seed(seed)

    dim = bounds.shape[1]

    # Sobol initial design
    sobol = torch.quasirandom.SobolEngine(dimension=dim, scramble=True, seed=seed)
    X_init = sobol.draw(initial_points).to(device=device, dtype=dtype)
    X_init = bounds[0] + (bounds[1] - bounds[0]) * X_init   # scale to domain
    Y_init = torch.tensor([objective(x.unsqueeze(0)).item()
                            for x in X_init], dtype=dtype, device=device)

    data_x = X_init
    data_y = Y_init
    # Track each initial evaluation individually so the array has
    # exactly (initial_points + max_iterations) entries — matching the DRO format.
    running_best = -float("inf")
    best_so_far  = []
    for v in data_y.tolist():
        running_best = max(running_best, v)
        best_so_far.append(running_best)

    ensemble = _build_ensemble(data_x, data_y, device=device, dtype=dtype)

    for it in range(max_iterations):
        best_f = float(data_y.max())
        cand   = _ensemble_logEI_candidate(ensemble, bounds, best_f, device, dtype)
        new_y  = objective(cand.unsqueeze(0)).item()
        data_x = torch.cat([data_x, cand.unsqueeze(0)], dim=0)
        data_y = torch.cat([data_y, torch.tensor([new_y], dtype=dtype, device=device)])
        best_so_far.append(float(data_y.max()))
        _refit_ensemble(ensemble, data_x, data_y)

    return np.maximum.accumulate(np.array(best_so_far))   # [initial_points + max_iterations]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GP-BO baseline (logEI, no DT)")
    parser.add_argument("--objective",      default=None,
                        choices=["Ackley", "Rosenbrock", "Levy"],
                        help="Single benchmark; omit to run all three")
    parser.add_argument("--dim",            type=int,   default=2)
    parser.add_argument("--domain_min",     type=float, default=-5.0)
    parser.add_argument("--domain_max",     type=float, default=5.0)
    parser.add_argument("--num_trials",     type=int,   default=20)
    parser.add_argument("--initial_points", type=int,   default=10)
    parser.add_argument("--max_iterations", type=int,   default=30)
    parser.add_argument("--seed_start",     type=int,   default=0)
    parser.add_argument("--save_dir",       default="res/gp_bo_baseline")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = "cpu"; dtype = DEFAULT_DTYPE
    total_evals   = args.initial_points + args.max_iterations
    objectives_to_run = ([args.objective] if args.objective
                         else ["Ackley", "Levy", "Rosenbrock"])

    all_results = {}

    for obj_name in objectives_to_run:
        bounds_t = torch.tensor(
            [[args.domain_min]*args.dim, [args.domain_max]*args.dim],
            dtype=dtype, device=device
        )
        cls = {"Ackley": Ackley, "Rosenbrock": Rosenbrock, "Levy": Levy}[obj_name]
        objective = cls(dim=args.dim,
                        bounds=torch.tensor([[args.domain_min, args.domain_max]]*args.dim,
                                             dtype=dtype),
                        negate=True)
        true_opt  = TRUE_OPTIMA[obj_name]

        print(f"\n{'='*60}")
        print(f"  GP-BO Baseline: {obj_name} {args.dim}D")
        print(f"  Trials: {args.num_trials}  |  Budget: {total_evals} evals")
        print(f"{'='*60}\n")

        best_arr = np.full((args.num_trials, total_evals), np.nan)

        for t in range(args.num_trials):
            seed = args.seed_start + t
            print(f"Trial {t+1}/{args.num_trials}  (seed={seed})", end=" ... ", flush=True)
            best = run_trial(objective, bounds_t, args.initial_points,
                             args.max_iterations, seed, device, dtype)
            best_arr[t, :len(best)] = best[:total_evals]
            print(f"best={best[-1]:.4f}")

        regret_arr = true_opt - best_arr
        all_results[obj_name] = regret_arr

        # Save per-benchmark npz
        np.savez(
            os.path.join(args.save_dir, f"gp_bo_{obj_name}_{args.dim}D.npz"),
            best=best_arr, regret=regret_arr, true_optimum=true_opt,
            initial_points=args.initial_points
        )

    # ── Summary report ────────────────────────────────────────────────────────
    lines = [
        "=" * 70,
        "  GP-BO Baseline — logEI, M=5 isotropic RBF ensemble, no DT",
        "=" * 70,
        f"  Dim: {args.dim}D  |  Trials: {args.num_trials}  |  Budget: {total_evals} evals",
        f"  Domain: [{args.domain_min}, {args.domain_max}]^{args.dim}",
        f"  GP: M=5, RBF isotropic, ls_init=linspace(0.1,10,5), Adam 50 iter",
        f"  Acquisition: logEI, ensemble committee (max logEI across 5 models)",
        "=" * 70,
        "",
        f"  {'Benchmark':<14} {'Mean':>10} {'Std':>10} {'Median':>10} {'IQR':>10} {'<0.01':>8}",
        "  " + "-"*62,
    ]
    for obj_name in objectives_to_run:
        reg = all_results[obj_name]
        fr  = reg[:, -1]
        q25, q75 = np.nanpercentile(fr, 25), np.nanpercentile(fr, 75)
        near_zero = int(np.sum(fr < 0.01))
        lines.append(
            f"  {obj_name+' 2D':<14} "
            f"{np.nanmean(fr):>10.4f} "
            f"{np.nanstd(fr):>10.4f} "
            f"{np.nanmedian(fr):>10.4f} "
            f"{q75-q25:>10.4f} "
            f"{near_zero:>5}/20"
        )
    lines += ["  " + "-"*62, ""]

    summary = "\n".join(lines)
    print(f"\n{summary}")
    txt_path = os.path.join(args.save_dir, "gp_bo_baseline_summary.txt")
    with open(txt_path, "w") as f:
        f.write(summary + "\n")
    print(f"  Summary → {txt_path}")


if __name__ == "__main__":
    main()
