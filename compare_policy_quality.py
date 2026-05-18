"""
compare_policy_quality.py

RQ2 Downstream Policy Quality: DT-GP vs DT-GT

Runs two independent DRO loops from identical starting conditions for each trial:

  DT-GP  standard DRO — Decision Transformer trained on GP-simulated trajectories.
         Zero extra true-function evaluations beyond the real BO queries.

  DT-GT  Oracle DRO — Decision Transformer trained on independently-generated
         ground-truth trajectories (Setup B).  Costs
           num_rollouts × max_rollout_length  additional true f(x) calls per BO
         iteration.  Treated as an upper bound, NOT a deployable alternative.

Both DTs are deployed in the same real BO loop: conditioned on the same D_t at
each iteration, they each propose the next query point and the true oracle f is
called at that point.  Because they propose different next points, their
datasets diverge from iteration 1 onward.

The gap between their simple-regret curves quantifies the performance cost of
training on biased (GP-simulated) vs perfect (true-function) trajectories.

Usage
-----
    python compare_policy_quality.py                          # defaults
    python compare_policy_quality.py --objective Rosenbrock --num_trials 10
    python compare_policy_quality.py --max_iterations 20 --num_rollouts 5 --seed_start 100

Output (written to --save_dir)
------
    policy_comparison_<obj>_<dim>D.npz   — raw regret/best arrays
    policy_comparison_<obj>_<dim>D.png   — side-by-side plot
    policy_comparison_<obj>_<dim>D.txt   — summary table
"""

import argparse
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from copy import deepcopy
from omegaconf import OmegaConf

# ── make sure project root is on the path ───────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.policy.dro import DirectRegretOptimization
from src.objectives import Ackley, Rosenbrock, Levy

DEFAULT_DTYPE = torch.float64
torch.set_default_dtype(DEFAULT_DTYPE)

# True optima for the negated objectives used in the codebase
# (negate=True → we maximise −f, so the reachable maximum is 0 for all three)
TRUE_OPTIMA = {
    "Ackley":     0.0,
    "Rosenbrock": 0.0,
    "Levy":       0.0,
}


# ────────────────────────────────────────────────────────────────────────────
# Config helpers
# ────────────────────────────────────────────────────────────────────────────

def _load_base_cfg() -> OmegaConf:
    """Load the dro.yaml and experiment defaults, return a merged DictConfig."""
    dro_yaml = os.path.join(ROOT, "config", "method", "dro.yaml")
    if not os.path.exists(dro_yaml):
        raise FileNotFoundError(f"Could not find DRO config at {dro_yaml}")
    return OmegaConf.load(dro_yaml)


def build_config(
    *,
    objective: str,
    dim: int,
    domain_min: float,
    domain_max: float,
    max_iterations: int,
    initial_points: int,
    num_rollouts: int,
    max_rollout_length: int,
    use_true_trajectories: bool,
    seed: int,
    kernel: str = "rbf",
    ard: bool = False,
    target_rtg=None,
    kernel_type: str = "rbf_isotropic",
    num_models: int = 5,
    dkl_hidden_dim: int = 32,
    dkl_latent_dim=None,
    dkl_n_iter: int = 100,
    dkl_lr: float = 0.01,
    dkl_weight_decay: float = 1e-2,
) -> OmegaConf:
    """Return a fully-specified DRO DictConfig for one trial."""
    cfg = _load_base_cfg()

    # BO settings
    cfg.bo.input_dim       = dim
    cfg.bo.domain_min      = [domain_min] * dim
    cfg.bo.domain_max      = [domain_max] * dim
    cfg.bo.max_iterations  = max_iterations
    cfg.bo.initial_points  = initial_points
    cfg.bo.objective       = "maximize"

    # Simulation settings
    cfg.simulation.num_rollouts        = num_rollouts
    cfg.simulation.max_rollout_length  = max_rollout_length
    cfg.simulation.use_true_trajectories = use_true_trajectories
    cfg.simulation.log_true_trajectories = False   # not needed here

    # GP kernel settings
    cfg.gp.kernel      = kernel
    cfg.gp.ard         = ard
    cfg.gp.num_models  = num_models

    if kernel_type == "deep_kernel":
        cfg.gp.kernel_type      = "deep_kernel"
        cfg.gp.dkl_hidden_dim   = dkl_hidden_dim
        cfg.gp.dkl_latent_dim   = dkl_latent_dim
        cfg.gp.dkl_n_iter       = dkl_n_iter
        cfg.gp.dkl_lr           = dkl_lr
        cfg.gp.dkl_weight_decay = dkl_weight_decay
    elif kernel_type == "rbf_ard":
        cfg.gp.ard = True

    # Target RTG at deployment: None = dynamic, float = fixed constant
    cfg.simulation.target_rtg = target_rtg

    cfg.seed    = seed
    cfg.verbose = False   # suppress per-iteration prints inside the optimizer

    return cfg


def build_objective(name: str, dim: int, domain_min: float, domain_max: float):
    """Instantiate a negated synthetic objective function."""
    bounds = torch.tensor(
        [[domain_min, domain_max]] * dim, dtype=DEFAULT_DTYPE
    )
    cls = {"Ackley": Ackley, "Rosenbrock": Rosenbrock, "Levy": Levy}[name]
    return cls(dim=dim, bounds=bounds, negate=True)


# ────────────────────────────────────────────────────────────────────────────
# Trial runner
# ────────────────────────────────────────────────────────────────────────────

def run_trial(cfg: OmegaConf, objective) -> tuple:
    """
    Run one complete DRO trial.

    Returns
    -------
    best_so_far : np.ndarray, shape [initial_points + max_iterations]
        Cumulative-best objective value at each evaluation.
    reward_diagnostics : list[dict]
        Per-BO-iteration reward/RTG statistics logged before each DT training
        step.  Each dict contains frac_zero_steps, frac_zero_trajs, mean_rtg,
        std_rtg, mean_nonzero_r, max_reward, and mode ('DT-GP' or 'DT-GT').
        Captured here before the dro object goes out of scope.
    """
    dro    = DirectRegretOptimization(cfg, objective)
    result = dro.run_optimization()

    all_y = result["all_y"]
    if isinstance(all_y, torch.Tensor):
        all_y = all_y.cpu().numpy()
    else:
        all_y = np.array(all_y, dtype=float)

    return np.maximum.accumulate(all_y), list(dro.reward_diagnostics), list(dro.dkl_diagnostics)


# ────────────────────────────────────────────────────────────────────────────
# Plotting
# ────────────────────────────────────────────────────────────────────────────

def _plot_comparison(
    iterations: np.ndarray,
    dtgp_regret: np.ndarray,
    dtgt_regret: np.ndarray,
    dtgp_best: np.ndarray,
    dtgt_best: np.ndarray,
    initial_points: int,
    objective: str,
    dim: int,
    num_trials: int,
    save_path: str,
) -> None:
    """
    Two-row figure:
      Top row    — mean ± 1 SE  (sensitive to outlier collapses)
      Bottom row — median + IQR  (robust; use this for the paper)
    Simple regret on the left, best observed value on the right.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    pairs = [
        (dtgp_regret, dtgt_regret, "Simple Regret"),
        (dtgp_best,   dtgt_best,   "Best Observed Value"),
    ]

    for col, (gp_data, gt_data, ylabel) in enumerate(pairs):
        # ── top row: mean ± SE ───────────────────────────────────────────────
        ax = axes[0, col]
        mean_gp = np.nanmean(gp_data, axis=0)
        se_gp   = np.nanstd(gp_data,  axis=0) / max(np.sqrt(num_trials), 1)
        mean_gt = np.nanmean(gt_data, axis=0)
        se_gt   = np.nanstd(gt_data,  axis=0) / max(np.sqrt(num_trials), 1)

        ax.plot(iterations, mean_gp, color="steelblue", label="DT-GP (simulated, free)")
        ax.fill_between(iterations, mean_gp - se_gp, mean_gp + se_gp,
                        color="steelblue", alpha=0.20)
        ax.plot(iterations, mean_gt, color="darkorange", linestyle="--",
                label="DT-GT (true oracle, costly)")
        ax.fill_between(iterations, mean_gt - se_gt, mean_gt + se_gt,
                        color="darkorange", alpha=0.20)
        ax.axvline(initial_points - 0.5, color="gray", linestyle=":", linewidth=1.0,
                   label="End of initial sampling")
        ax.set_xlabel("Function evaluations (incl. initial)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} — Mean ± 1 SE\n"
                     f"{objective} {dim}D, {num_trials} trial(s)  [sensitive to outliers]")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        # ── bottom row: median + IQR ─────────────────────────────────────────
        ax = axes[1, col]
        med_gp  = np.nanmedian(gp_data, axis=0)
        q25_gp  = np.nanpercentile(gp_data, 25, axis=0)
        q75_gp  = np.nanpercentile(gp_data, 75, axis=0)
        med_gt  = np.nanmedian(gt_data, axis=0)
        q25_gt  = np.nanpercentile(gt_data, 25, axis=0)
        q75_gt  = np.nanpercentile(gt_data, 75, axis=0)

        ax.plot(iterations, med_gp, color="steelblue", label="DT-GP (simulated, free)")
        ax.fill_between(iterations, q25_gp, q75_gp, color="steelblue", alpha=0.20)
        ax.plot(iterations, med_gt, color="darkorange", linestyle="--",
                label="DT-GT (true oracle, costly)")
        ax.fill_between(iterations, q25_gt, q75_gt, color="darkorange", alpha=0.20)
        ax.axvline(initial_points - 0.5, color="gray", linestyle=":", linewidth=1.0,
                   label="End of initial sampling")
        ax.set_xlabel("Function evaluations (incl. initial)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} — Median + IQR\n"
                     f"{objective} {dim}D, {num_trials} trial(s)  [robust to collapses — use for paper]")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Plot  → {save_path}")


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare DT-GP vs DT-GT policy quality (RQ2)."
    )
    parser.add_argument("--objective",         default="Ackley",
                        choices=["Ackley", "Rosenbrock", "Levy"])
    parser.add_argument("--dim",               type=int,   default=2)
    parser.add_argument("--domain_min",        type=float, default=-5.0)
    parser.add_argument("--domain_max",        type=float, default=5.0)
    parser.add_argument("--num_trials",        type=int,   default=5,
                        help="Number of independent seeds")
    parser.add_argument("--max_iterations",    type=int,   default=30,
                        help="BO iterations per trial (after initial points)")
    parser.add_argument("--initial_points",    type=int,   default=5)
    parser.add_argument("--num_rollouts",      type=int,   default=10,
                        help="Rollouts per BO iteration (for DT training)")
    parser.add_argument("--max_rollout_length",type=int,   default=4,
                        help="Max steps per rollout")
    parser.add_argument("--seed_start",        type=int,   default=0)
    parser.add_argument("--save_dir",          default="res/policy_comparison")
    parser.add_argument("--kernel",            default="rbf", choices=["rbf", "matern"],
                        help="GP kernel type (rbf or matern)")
    parser.add_argument("--ard",               action="store_true",
                        help="Enable ARD (per-dimension lengthscales) in the GP kernel")
    parser.add_argument("--target_rtg",        default=None, type=float,
                        help="Fixed target RTG at deployment (default: None = dynamic max_rtg)")
    parser.add_argument("--kernel_type",       default="rbf_isotropic",
                        choices=["rbf_isotropic", "rbf_ard", "deep_kernel"],
                        help="GP ensemble kernel type (default: rbf_isotropic)")
    parser.add_argument("--dkl_hidden_dim",    type=int,   default=32,
                        help="DKL feature extractor hidden dim")
    parser.add_argument("--dkl_latent_dim",    type=int,   default=None,
                        help="DKL latent dim (default: 2*input_dim)")
    parser.add_argument("--dkl_n_iter",        type=int,   default=100,
                        help="DKL anchor training iterations")
    parser.add_argument("--dkl_lr",            type=float, default=0.01,
                        help="DKL learning rate")
    parser.add_argument("--dkl_weight_decay",  type=float, default=1e-2,
                        help="DKL weight decay for feature extractor")
    parser.add_argument("--num_models",        type=int,   default=5,
                        help="Number of GPs in the ensemble (default: 5)")
    args = parser.parse_args()

    if args.kernel_type != "rbf_isotropic" and args.save_dir == "res/policy_comparison":
        args.save_dir = f"res/dkl_{args.objective.lower()}_{args.kernel_type}"

    os.makedirs(args.save_dir, exist_ok=True)
    objective = build_objective(
        args.objective, args.dim, args.domain_min, args.domain_max
    )
    true_opt   = TRUE_OPTIMA[args.objective]
    total_evals = args.initial_points + args.max_iterations

    # Extra true-f evaluations consumed by DT-GT per BO iteration
    extra_evals_per_iter = args.num_rollouts * args.max_rollout_length
    total_extra_evals    = extra_evals_per_iter * args.max_iterations

    kernel_label = f"{args.kernel.upper()}" + (" + ARD" if args.ard else " (isotropic)")
    rtg_label    = f"{args.target_rtg:.2f} (fixed)" if args.target_rtg is not None else "dynamic (max_rtg)"
    print(f"\n{'='*65}")
    print(f"  Policy Quality Comparison: DT-GP vs DT-GT")
    print(f"  Objective : {args.objective} {args.dim}D")
    print(f"  GP kernel : {kernel_label}")
    print(f"  Target RTG: {rtg_label}")
    print(f"  Trials    : {args.num_trials}  |  Seeds {args.seed_start}–"
          f"{args.seed_start + args.num_trials - 1}")
    print(f"  BO budget : {args.initial_points} init + {args.max_iterations} iter "
          f"= {total_evals} evaluations")
    print(f"  DT-GT extra eval cost : {extra_evals_per_iter}/iter × "
          f"{args.max_iterations} iter = {total_extra_evals} extra f-calls")
    print(f"{'='*65}\n")

    import json as _json_resume

    # Arrays: [num_trials, total_evals]
    dtgp_best = np.full((args.num_trials, total_evals), np.nan)
    dtgt_best = np.full((args.num_trials, total_evals), np.nan)

    all_diag_gp = []
    all_diag_gt = []
    all_dkl_gp  = []
    all_dkl_gt  = []

    # ── Auto-resume from checkpoint if one exists ─────────────────────────────
    stem_ckpt      = f"policy_comparison_{args.objective}_{args.dim}D"
    ckpt_path      = os.path.join(args.save_dir, f"{stem_ckpt}.ckpt.npz")
    diag_ckpt_path = os.path.join(args.save_dir, f"{stem_ckpt}.ckpt_diag.json")
    dkl_ckpt_path  = os.path.join(args.save_dir, f"{stem_ckpt}.ckpt_dkl_diag.json")

    start_trial = 0
    if os.path.exists(ckpt_path):
        try:
            _ckpt = np.load(ckpt_path, allow_pickle=True)
            _prev = int(_ckpt['trials_done'])
            if 0 < _prev <= args.num_trials:
                dtgp_best[:_prev] = _ckpt['dtgp_best'][:_prev]
                dtgt_best[:_prev] = _ckpt['dtgt_best'][:_prev]
                start_trial = _prev
                print(f"[Resume] Checkpoint found — {_prev}/{args.num_trials} trials already done.")
            if os.path.exists(diag_ckpt_path):
                with open(diag_ckpt_path) as _cf:
                    _dc = _json_resume.load(_cf)
                all_diag_gp = _dc.get('DT-GP', [])
                all_diag_gt = _dc.get('DT-GT', [])
            if os.path.exists(dkl_ckpt_path):
                with open(dkl_ckpt_path) as _cf:
                    _dkc = _json_resume.load(_cf)
                all_dkl_gp = _dkc.get('DT-GP', [])
                all_dkl_gt = _dkc.get('DT-GT', [])
        except Exception as _e:
            print(f"[Resume] Warning: could not load checkpoint ({_e}). Starting fresh.")
            start_trial = 0
            dtgp_best[:] = np.nan
            dtgt_best[:] = np.nan
            all_diag_gp, all_diag_gt, all_dkl_gp, all_dkl_gt = [], [], [], []

    for t in range(start_trial, args.num_trials):
        seed = args.seed_start + t
        print(f"Trial {t + 1}/{args.num_trials}  (seed={seed})")

        # ── DT-GP ──────────────────────────────────────────────────────────
        torch.manual_seed(seed)
        np.random.seed(seed)
        cfg_gp = build_config(
            objective=args.objective,
            dim=args.dim,
            domain_min=args.domain_min,
            domain_max=args.domain_max,
            max_iterations=args.max_iterations,
            initial_points=args.initial_points,
            num_rollouts=args.num_rollouts,
            max_rollout_length=args.max_rollout_length,
            use_true_trajectories=False,
            seed=seed,
            kernel=args.kernel,
            ard=args.ard,
            target_rtg=args.target_rtg,
            kernel_type=args.kernel_type,
            num_models=args.num_models,
            dkl_hidden_dim=args.dkl_hidden_dim,
            dkl_latent_dim=args.dkl_latent_dim,
            dkl_n_iter=args.dkl_n_iter,
            dkl_lr=args.dkl_lr,
            dkl_weight_decay=args.dkl_weight_decay,
        )
        print("  [DT-GP] running...", end=" ", flush=True)
        best_gp, diag_gp, dkl_diag_gp = run_trial(cfg_gp, objective)
        dtgp_best[t, :len(best_gp)] = best_gp[:total_evals]
        all_diag_gp.append(diag_gp)
        all_dkl_gp.append(dkl_diag_gp)
        print(f"final best = {best_gp[-1]:.4f}")

        # ── DT-GT ──────────────────────────────────────────────────────────
        torch.manual_seed(seed)
        np.random.seed(seed)
        cfg_gt = build_config(
            objective=args.objective,
            dim=args.dim,
            domain_min=args.domain_min,
            domain_max=args.domain_max,
            max_iterations=args.max_iterations,
            initial_points=args.initial_points,
            num_rollouts=args.num_rollouts,
            max_rollout_length=args.max_rollout_length,
            use_true_trajectories=True,
            seed=seed,
            kernel=args.kernel,
            ard=args.ard,
            target_rtg=args.target_rtg,
            kernel_type=args.kernel_type,
            num_models=args.num_models,
            dkl_hidden_dim=args.dkl_hidden_dim,
            dkl_latent_dim=args.dkl_latent_dim,
            dkl_n_iter=args.dkl_n_iter,
            dkl_lr=args.dkl_lr,
            dkl_weight_decay=args.dkl_weight_decay,
        )
        print("  [DT-GT] running...", end=" ", flush=True)
        best_gt, diag_gt, dkl_diag_gt = run_trial(cfg_gt, objective)
        dtgt_best[t, :len(best_gt)] = best_gt[:total_evals]
        all_diag_gt.append(diag_gt)
        all_dkl_gt.append(dkl_diag_gt)
        print(f"final best = {best_gt[-1]:.4f}")

        # ── Incremental checkpoint after each completed trial ───────────────
        np.savez(ckpt_path,
                 dtgp_best=dtgp_best,
                 dtgt_best=dtgt_best,
                 trials_done=t + 1,
                 seed_start=args.seed_start,
                 initial_points=args.initial_points)
        with open(diag_ckpt_path, 'w') as _cf:
            _json_resume.dump({'DT-GP': all_diag_gp, 'DT-GT': all_diag_gt,
                               'trials_done': t + 1, 'seed_start': args.seed_start}, _cf, indent=2)
        with open(dkl_ckpt_path, 'w') as _cf:
            _json_resume.dump({'DT-GP': all_dkl_gp, 'DT-GT': all_dkl_gt,
                               'trials_done': t + 1, 'seed_start': args.seed_start}, _cf, indent=2)
        print(f"  [ckpt] {t + 1}/{args.num_trials} trials saved → {ckpt_path}")

    # Simple regret = true_opt − best_so_far
    dtgp_regret = true_opt - dtgp_best   # positive (or 0) for maximisation
    dtgt_regret = true_opt - dtgt_best

    # ── Save arrays ──────────────────────────────────────────────────────────
    stem     = f"policy_comparison_{args.objective}_{args.dim}D"
    npz_path = os.path.join(args.save_dir, f"{stem}.npz")
    np.savez(
        npz_path,
        dtgp_best=dtgp_best,
        dtgt_best=dtgt_best,
        dtgp_regret=dtgp_regret,
        dtgt_regret=dtgt_regret,
        true_optimum=true_opt,
        iterations=np.arange(total_evals),
        initial_points=args.initial_points,
        extra_eval_cost_per_iter=extra_evals_per_iter,
        total_extra_evals=total_extra_evals,
    )
    print(f"\n  Arrays → {npz_path}")

    # ── Save reward diagnostics ───────────────────────────────────────────────
    import json as _json
    diag_path = os.path.join(args.save_dir, f"{stem}_reward_diag.json")
    with open(diag_path, 'w') as _f:
        _json.dump({'DT-GP': all_diag_gp, 'DT-GT': all_diag_gt}, _f, indent=2)
    print(f"  Reward diagnostics → {diag_path}")

    if args.kernel_type == "deep_kernel":
        dkl_diag_path = os.path.join(args.save_dir, f"{stem}_dkl_diagnostics.json")
        with open(dkl_diag_path, 'w') as _f:
            _json.dump({'DT-GP': all_dkl_gp, 'DT-GT': all_dkl_gt}, _f, indent=2)
        print(f"  DKL diagnostics    → {dkl_diag_path}")

    # ── Print cross-trial reward summary ─────────────────────────────────────
    def _diag_summary(all_diag: list, label: str):
        """Print cross-trial summary of key reward/training diagnostics."""
        frac_zeros  = [d['frac_zero_steps']  for trial in all_diag for d in trial if d]
        mean_rtgs   = [d['mean_rtg']         for trial in all_diag for d in trial if d]
        max_rtgs    = [d['max_rtg']          for trial in all_diag for d in trial if d]
        avg_lens    = [d['mean_rollout_len'] for trial in all_diag for d in trial if d]
        dt_losses   = [d['dt_final_loss']    for trial in all_diag for d in trial if d and d['dt_final_loss'] is not None]
        print(
            f"  [{label}] zero_steps={np.mean(frac_zeros):.1%}±{np.std(frac_zeros):.1%}  "
            f"mean_RTG={np.mean(mean_rtgs):.4f}±{np.std(mean_rtgs):.4f}  "
            f"max_RTG={np.mean(max_rtgs):.4f}  "
            f"avg_len={np.mean(avg_lens):.2f}  "
            f"dt_loss={np.mean(dt_losses):.5f}±{np.std(dt_losses):.5f}" if dt_losses else
            f"  [{label}] zero_steps={np.mean(frac_zeros):.1%}  mean_RTG={np.mean(mean_rtgs):.4f}"
        )

    print("\n── Reward density summary (all trials × all BO iters) ──")
    _diag_summary(all_diag_gp, "DT-GP")
    _diag_summary(all_diag_gt, "DT-GT")

    # ── Plot ─────────────────────────────────────────────────────────────────
    iterations = np.arange(total_evals)
    _plot_comparison(
        iterations=iterations,
        dtgp_regret=dtgp_regret,
        dtgt_regret=dtgt_regret,
        dtgp_best=dtgp_best,
        dtgt_best=dtgt_best,
        initial_points=args.initial_points,
        objective=args.objective,
        dim=args.dim,
        num_trials=args.num_trials,
        save_path=os.path.join(args.save_dir, f"{stem}.png"),
    )

    # ── Summary table ─────────────────────────────────────────────────────────
    final_regret_gp = dtgp_regret[:, -1]
    final_regret_gt = dtgt_regret[:, -1]

    # Count DT-GT collapse trials (regret > 2× DT-GP mean — heuristic threshold)
    collapse_thresh  = 2.0 * np.nanmean(final_regret_gp)
    n_collapse       = int(np.sum(final_regret_gt > collapse_thresh))

    gap_mean = np.nanmean(final_regret_gp) - np.nanmean(final_regret_gt)
    gap_med  = np.nanmedian(final_regret_gp) - np.nanmedian(final_regret_gt)
    gap_se   = np.sqrt(
        (np.nanstd(final_regret_gp) ** 2 + np.nanstd(final_regret_gt) ** 2)
        / max(args.num_trials, 1)
    )

    W = 65
    summary_lines = [
        f"Policy Quality Comparison — {args.objective} {args.dim}D",
        f"Trials: {args.num_trials}  |  Budget: {total_evals} evals per DT",
        "=" * W,
        f"{'Metric':<38} {'DT-GP':>8} {'DT-GT':>8} {'Gap':>9}",
        "-" * W,
        # Mean row
        f"{'Final simple regret  (mean)':<38} "
        f"{np.nanmean(final_regret_gp):>8.4f} "
        f"{np.nanmean(final_regret_gt):>8.4f} "
        f"{gap_mean:>+9.4f}",
        f"{'  std':<38} "
        f"{np.nanstd(final_regret_gp):>8.4f} "
        f"{np.nanstd(final_regret_gt):>8.4f} "
        f"{'±'+f'{gap_se:.4f}':>9}",
        # Median row  ← USE THIS FOR THE PAPER (robust to DT-GT collapses)
        f"{'Final simple regret  (median) *':<38} "
        f"{np.nanmedian(final_regret_gp):>8.4f} "
        f"{np.nanmedian(final_regret_gt):>8.4f} "
        f"{gap_med:>+9.4f}",
        f"{'  IQR':<38} "
        f"{np.nanpercentile(final_regret_gp,75)-np.nanpercentile(final_regret_gp,25):>8.4f} "
        f"{np.nanpercentile(final_regret_gt,75)-np.nanpercentile(final_regret_gt,25):>8.4f}",
        "-" * W,
        f"{'DT-GT collapse trials (regret > 2×GP mean)':<38} "
        f"{'—':>8} {n_collapse:>8d} / {args.num_trials}",
        "=" * W,
        "",
        f"* Median is robust to DT-GT collapse seeds — report this in the paper.",
        f"  Mean is inflated by {n_collapse} collapse trial(s).",
        "",
        f"Extra true f-evals consumed by DT-GT: "
        f"{extra_evals_per_iter}/iter × {args.max_iterations} iter "
        f"= {total_extra_evals} calls",
        f"(vs {total_evals} real BO evaluations per DT — "
        f"{total_extra_evals / total_evals:.1f}× overhead)",
        "",
        "Gap interpretation (negative = DT-GP wins, positive = DT-GT wins):",
        "  gap_median > 0  → simulation bias measurably hurts DT-GP",
        "  gap_median ≈ 0  → GP simulation faithful enough; DT-GP justified",
        "  gap_median < 0  → DT-GP competitive with oracle (GP bias may regularise)",
    ]

    summary = "\n".join(summary_lines)
    print(f"\n{summary}\n")

    txt_path = os.path.join(args.save_dir, f"{stem}.txt")
    with open(txt_path, "w") as fh:
        fh.write(summary + "\n")
    print(f"  Summary → {txt_path}")


if __name__ == "__main__":
    main()
