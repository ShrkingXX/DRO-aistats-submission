"""
no_gp_state_experiment.py

Ablation: does removing GP hyperparameters from the state vector
change DT-GT (and DT-GP) performance on Rosenbrock 2D?

GP hyperparameters are 10/19 = 53% of the default state vector.
In DT-GT mode they are INCONSISTENT between training and deployment:
  - training: extracted with GP refitted on n_real + k_rollout points
  - deployment: extracted with GP fitted on n_real points only
This ablation tests whether that inconsistency actually hurts performance
by removing the GP param block entirely.

Conditions run here (10 seeds, Rosenbrock 2D):
  DT-GT-no-gp-state   use_true_traj=True,  state_include_gp_params=False
  DT-GP-no-gp-state   use_true_traj=False, state_include_gp_params=False

Reference conditions (loaded from existing results, NOT rerun):
  DT-GT-with-gp-state  res/ablation_rtg_rosenbrock/dynamic/   (10 seeds)
  DT-GP-with-gp-state  res/ablation_rtg_rosenbrock/dynamic/   (10 seeds)

All other settings identical to the reference experiment:
  40 total evals, 30 rollouts/iter, max_rollout_length=4, isotropic RBF, dynamic RTG.

Usage
-----
    .venv/bin/python no_gp_state_experiment.py
    .venv/bin/python no_gp_state_experiment.py --num_trials 3   # quick test
"""

import argparse, json, os, sys
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.policy.dro import DirectRegretOptimization
from src.objectives import Rosenbrock

DEFAULT_DTYPE = torch.float64
torch.set_default_dtype(DEFAULT_DTYPE)


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_base_cfg():
    return OmegaConf.load(os.path.join(ROOT, "config", "method", "dro.yaml"))


def build_config(*, dim, domain_min, domain_max, max_iterations, initial_points,
                 num_rollouts, max_rollout_length, use_true_trajectories,
                 seed, state_include_gp_params):
    cfg = _load_base_cfg()
    cfg.bo.input_dim      = dim
    cfg.bo.domain_min     = [domain_min] * dim
    cfg.bo.domain_max     = [domain_max] * dim
    cfg.bo.max_iterations = max_iterations
    cfg.bo.initial_points = initial_points
    cfg.bo.objective      = "maximize"
    cfg.simulation.num_rollouts            = num_rollouts
    cfg.simulation.max_rollout_length      = max_rollout_length
    cfg.simulation.use_true_trajectories   = use_true_trajectories
    cfg.simulation.log_true_trajectories   = False
    cfg.simulation.rtg_mode                = "dynamic"
    cfg.simulation.target_rtg              = None
    cfg.simulation.rtg_alpha               = None
    cfg.simulation.state_include_gp_params = state_include_gp_params
    cfg.gp.kernel = "rbf"
    cfg.gp.ard    = False
    cfg.seed      = seed
    cfg.verbose   = False
    return cfg


def build_objective(dim, domain_min, domain_max):
    bounds = torch.tensor([[domain_min, domain_max]] * dim, dtype=DEFAULT_DTYPE)
    return Rosenbrock(dim=dim, bounds=bounds, negate=True)


def run_trial(cfg, objective):
    dro    = DirectRegretOptimization(cfg, objective)
    result = dro.run_optimization()
    all_y  = result["all_y"]
    if isinstance(all_y, torch.Tensor):
        all_y = all_y.cpu().numpy()
    return np.maximum.accumulate(np.array(all_y, dtype=float)), list(dro.reward_diagnostics)


def _stats(arr):
    q25, q75 = np.percentile(arr, 25), np.percentile(arr, 75)
    return dict(mean=float(np.mean(arr)), std=float(np.std(arr)),
                med=float(np.median(arr)),  iqr=float(q75-q25),
                near_zero=int(np.sum(arr < 0.01)),
                catastrophic=int(np.sum(arr > 50)))


def write_summary(results_new, ref, save_dir, num_trials):
    """Write a compact comparison report."""
    W = 80
    lines = [
        "=" * W,
        "  State Ablation: GP Hyperparameters in State vs. Removed",
        "  Rosenbrock 2D — DT-GT and DT-GP, dynamic RTG",
        "=" * W,
        f"  New conditions  : state_include_gp_params=False  ({num_trials} seeds)",
        f"  Reference       : state_include_gp_params=True   (10 seeds, existing data)",
        f"  All other settings identical (40 evals, 30 rollouts, isotropic RBF)",
        "=" * W, "",
        "SECTION 1 — FINAL SIMPLE REGRET",
        "─" * W,
        f"  {'Condition':<28} {'Mean':>8} {'Std':>8} {'Median':>8} "
        f"{'IQR':>8} {'<0.01':>6} {'>50':>6}",
        "  " + "─" * 66,
    ]

    rows = [
        ("DT-GT  with GP state (ref)",  ref['dtgt_regret']),
        ("DT-GT  no   GP state (new)",  results_new['dtgt']['regret']),
        ("DT-GP  with GP state (ref)",  ref['dtgp_regret']),
        ("DT-GP  no   GP state (new)",  results_new['dtgp']['regret']),
    ]
    for label, regret in rows:
        s = _stats(regret)
        lines.append(
            f"  {label:<28} {s['mean']:>8.3f} {s['std']:>8.3f} {s['med']:>8.4f} "
            f"{s['iqr']:>8.4f} {s['near_zero']:>5}/{len(regret)} {s['catastrophic']:>5}/{len(regret)}"
        )

    lines += ["", "  Per-trial final regret:", ""]
    for label, regret in rows:
        lines.append(f"  {label}")
        for i, v in enumerate(regret):
            flag = "  ← catastrophic" if v > 50 else ""
            lines.append(f"    seed {i:2d}: {v:10.4f}{flag}")
        lines.append("")

    # Head-to-head: no-GP vs with-GP
    lines += ["  Head-to-head (no-GP-state wins over with-GP-state, same seeds):"]
    n = min(len(results_new['dtgt']['regret']), len(ref['dtgt_regret']))
    dtgt_wins = int(np.sum(results_new['dtgt']['regret'][:n] < ref['dtgt_regret'][:n]))
    dtgp_wins = int(np.sum(results_new['dtgp']['regret'][:n] < ref['dtgp_regret'][:n]))
    lines.append(f"    DT-GT no-GP wins: {dtgt_wins}/{n}")
    lines.append(f"    DT-GP no-GP wins: {dtgp_wins}/{n}")
    lines += [""]

    lines += [
        "=" * W,
        "SECTION 2 — INTERPRETATION",
        "─" * W, "",
    ]

    gt_new_med = _stats(results_new['dtgt']['regret'])['med']
    gt_ref_med = _stats(ref['dtgt_regret'])['med']
    gp_new_med = _stats(results_new['dtgp']['regret'])['med']
    gp_ref_med = _stats(ref['dtgp_regret'])['med']

    if gt_new_med < gt_ref_med:
        gt_verdict = (f"Removing GP params IMPROVES DT-GT median "
                      f"({gt_new_med:.4f} vs {gt_ref_med:.4f}), consistent with the "
                      f"hypothesis that the training/deployment state inconsistency "
                      f"was hurting performance.")
    elif abs(gt_new_med - gt_ref_med) < 0.5:
        gt_verdict = (f"Removing GP params has NEGLIGIBLE effect on DT-GT median "
                      f"({gt_new_med:.4f} vs {gt_ref_med:.4f}). The DT-GT may already "
                      f"be ignoring the GP hyperparameter dimensions of the state.")
    else:
        gt_verdict = (f"Removing GP params HURTS DT-GT median "
                      f"({gt_new_med:.4f} vs {gt_ref_med:.4f}). The GP hyperparameter "
                      f"signal, despite being inconsistent, provided useful information.")

    if abs(gp_new_med - gp_ref_med) < 0.3:
        gp_verdict = (f"DT-GP median is largely unchanged ({gp_new_med:.4f} vs "
                      f"{gp_ref_med:.4f}), which is expected: DT-GP has no "
                      f"training/deployment GP-state inconsistency, so removing GP "
                      f"params only discards information without fixing a bug.")
    elif gp_new_med < gp_ref_med:
        gp_verdict = (f"DT-GP median also improves ({gp_new_med:.4f} vs {gp_ref_med:.4f}), "
                      f"suggesting the GP hyperparameter block adds noise rather than "
                      f"useful signal for both modes.")
    else:
        gp_verdict = (f"DT-GP median worsens ({gp_new_med:.4f} vs {gp_ref_med:.4f}), "
                      f"suggesting GP hyperparameters are genuinely informative for "
                      f"DT-GP even if not for DT-GT.")

    lines += [f"  DT-GT: {gt_verdict}", "",
              f"  DT-GP: {gp_verdict}", ""]

    lines += ["=" * W,
              f"  Source: {save_dir}/",
              "=" * W]

    path = os.path.join(save_dir, "no_gp_state_summary.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Summary → {path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_trials",         type=int,   default=10)
    parser.add_argument("--seed_start",         type=int,   default=0)
    parser.add_argument("--dim",                type=int,   default=2)
    parser.add_argument("--domain_min",         type=float, default=-5.0)
    parser.add_argument("--domain_max",         type=float, default=5.0)
    parser.add_argument("--initial_points",     type=int,   default=10)
    parser.add_argument("--max_iterations",     type=int,   default=30)
    parser.add_argument("--num_rollouts",       type=int,   default=30)
    parser.add_argument("--max_rollout_length", type=int,   default=4)
    parser.add_argument("--save_dir",           default="res/no_gp_state_rosenbrock")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    total_evals = args.initial_points + args.max_iterations
    objective   = build_objective(args.dim, args.domain_min, args.domain_max)

    # Conditions: (key, use_true_traj, label)
    conditions = [
        ('dtgt', True,  'DT-GT-no-gp-state'),
        ('dtgp', False, 'DT-GP-no-gp-state'),
    ]

    print(f"\n{'='*65}")
    print(f"  State Ablation: No GP Hyperparameters — Rosenbrock 2D")
    print(f"  Trials: {args.num_trials}  |  Seeds {args.seed_start}–"
          f"{args.seed_start + args.num_trials - 1}")
    print(f"  state_dim: 4  (best_val, step_norm, best_pos_x, best_pos_y)")
    print(f"  vs default: 19  (10 GP params + 4 data dims + 5 noise/etc)")
    print(f"{'='*65}\n")

    results = {
        key: {'best': np.full((args.num_trials, total_evals), np.nan),
              'regret': None, 'diag': []}
        for key, *_ in conditions
    }

    for t in range(args.num_trials):
        seed = args.seed_start + t
        print(f"\n── Trial {t+1}/{args.num_trials}  (seed={seed}) {'─'*38}")

        for key, use_gt, label in conditions:
            torch.manual_seed(seed); np.random.seed(seed)
            cfg = build_config(
                dim=args.dim, domain_min=args.domain_min, domain_max=args.domain_max,
                max_iterations=args.max_iterations, initial_points=args.initial_points,
                num_rollouts=args.num_rollouts, max_rollout_length=args.max_rollout_length,
                use_true_trajectories=use_gt, seed=seed,
                state_include_gp_params=False,
            )
            print(f"  [{label}] running...", end=" ", flush=True)
            best, diag = run_trial(cfg, objective)
            results[key]['best'][t, :len(best)] = best[:total_evals]
            results[key]['diag'].append(diag)
            print(f"final best = {best[-1]:.4f}")

        # Checkpoint
        ckpt = {key: results[key]['best'].tolist() for key, *_ in conditions}
        ckpt['trials_done'] = t + 1
        with open(os.path.join(args.save_dir, "no_gp_state.ckpt.json"), 'w') as f:
            json.dump(ckpt, f)
        print(f"  [ckpt] {t+1}/{args.num_trials} saved")

    # Final regret
    for key, *_ in conditions:
        results[key]['regret'] = 0.0 - results[key]['best'][:, -1]

    # Save npz + diag
    for key, use_gt, label in conditions:
        tag = label.lower().replace("-", "_")
        np.savez(os.path.join(args.save_dir, f"{tag}.npz"),
                 best=results[key]['best'], regret=results[key]['regret'],
                 true_optimum=0.0, initial_points=args.initial_points)
        with open(os.path.join(args.save_dir, f"{tag}_diag.json"), 'w') as f:
            json.dump(results[key]['diag'], f, indent=2)

    # Load reference (with GP params, dynamic RTG, same 10 seeds)
    ref_dir  = os.path.join(ROOT, "res", "ablation_rtg_rosenbrock", "dynamic",
                            "policy_comparison_Rosenbrock_2D.npz")
    ref_data = np.load(ref_dir)
    ref = {
        'dtgt_regret': 0.0 - ref_data['dtgt_best'][:, -1],
        'dtgp_regret': 0.0 - ref_data['dtgp_best'][:, -1],
    }

    print(f"\n{'='*65}")
    print("  Writing summary...")
    write_summary(results, ref, args.save_dir, args.num_trials)
    print("  Done.")


if __name__ == "__main__":
    main()
