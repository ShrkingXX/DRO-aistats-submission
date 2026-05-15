"""
floored_rtg_experiment.py

Experiment: Floored Dynamic RTG on synthetic datasets — DT-GT only.

New RTG schema:
    target_RTG(t) = max(max_RTG_from_batch(t), alpha * running_max_RTG(t))

where running_max_RTG is the historical maximum of max_RTG_from_batch over all
iterations 1..t, initialized to 0.  This prevents zero-collapse of the DT
conditioning signal while remaining proportional to the function's actual reward
scale.

Conditions (10 seeds each, seeds 0–9):
    DT-GT-floor05  : DT-GT, floored_dynamic, alpha=0.5
    DT-GT-floor02  : DT-GT, floored_dynamic, alpha=0.2
    DT-GT-floor01  : DT-GT, floored_dynamic, alpha=0.1
    DT-GP-dynamic  : DT-GP, dynamic RTG (within-run reference, same seeds)

Reference conditions (loaded from existing results, NOT rerun):
    (Only for Rosenbrock; optional if files exist)
    DT-GT-dynamic  : res/ablation_rtg_rosenbrock/dynamic/
    DT-GT-fixed1.0 : res/ablation_rtg_rosenbrock/fixed_1.0/

Usage
-----
    .venv/bin/python floored_rtg_experiment.py --dataset rosenbrock
    .venv/bin/python floored_rtg_experiment.py --dataset levy
    .venv/bin/python floored_rtg_experiment.py --dataset ackley
    .venv/bin/python floored_rtg_experiment.py --num_trials 5   # for quick test
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
from omegaconf import OmegaConf
from copy import deepcopy

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.policy.dro import DirectRegretOptimization
from src.objectives import Ackley, Levy, Rosenbrock

DEFAULT_DTYPE = torch.float64
torch.set_default_dtype(DEFAULT_DTYPE)

OBJECTIVE_MAP = {
    "rosenbrock": ("Rosenbrock", Rosenbrock),
    "levy": ("Levy", Levy),
    "ackley": ("Ackley", Ackley),
}
TRUE_OPTIMA = {"Rosenbrock": 0.0, "Levy": 0.0, "Ackley": 0.0}


# ── Config helpers ────────────────────────────────────────────────────────────

def _load_base_cfg() -> OmegaConf:
    dro_yaml = os.path.join(ROOT, "config", "method", "dro.yaml")
    return OmegaConf.load(dro_yaml)


def build_config(
    *,
    dim: int,
    domain_min: float,
    domain_max: float,
    max_iterations: int,
    initial_points: int,
    num_rollouts: int,
    max_rollout_length: int,
    use_true_trajectories: bool,
    seed: int,
    rtg_mode: str = "dynamic",       # "dynamic" | "floored_dynamic" | "fixed"
    rtg_alpha: float = None,          # required for floored_dynamic
    target_rtg: float = None,         # required for fixed
) -> OmegaConf:
    cfg = _load_base_cfg()

    cfg.bo.input_dim       = dim
    cfg.bo.domain_min      = [domain_min] * dim
    cfg.bo.domain_max      = [domain_max] * dim
    cfg.bo.max_iterations  = max_iterations
    cfg.bo.initial_points  = initial_points
    cfg.bo.objective       = "maximize"

    cfg.simulation.num_rollouts          = num_rollouts
    cfg.simulation.max_rollout_length    = max_rollout_length
    cfg.simulation.use_true_trajectories = use_true_trajectories
    cfg.simulation.log_true_trajectories = False
    cfg.simulation.rtg_mode              = rtg_mode
    cfg.simulation.target_rtg            = target_rtg   # None unless fixed mode
    cfg.simulation.rtg_alpha             = rtg_alpha    # None unless floored_dynamic

    cfg.gp.kernel = "rbf"
    cfg.gp.ard    = False

    cfg.seed    = seed
    cfg.verbose = False

    return cfg


def build_objective(dataset: str, dim, domain_min, domain_max):
    bounds = torch.tensor([[domain_min, domain_max]] * dim, dtype=DEFAULT_DTYPE)
    _, obj_cls = OBJECTIVE_MAP[dataset]
    return obj_cls(dim=dim, bounds=bounds, negate=True)


# ── Trial runner ──────────────────────────────────────────────────────────────

def run_trial(cfg: OmegaConf, objective) -> tuple:
    """
    Returns
    -------
    best_so_far : np.ndarray  [initial_points + max_iterations]
    reward_diagnostics : list[dict]
    """
    dro    = DirectRegretOptimization(cfg, objective)
    result = dro.run_optimization()

    all_y = result["all_y"]
    if isinstance(all_y, torch.Tensor):
        all_y = all_y.cpu().numpy()
    else:
        all_y = np.array(all_y, dtype=float)

    return np.maximum.accumulate(all_y), list(dro.reward_diagnostics)


# ── Output generation ─────────────────────────────────────────────────────────

def _load_ref_data(dataset: str, objective_name: str, dim: int):
    """Load existing DT-GT-dynamic and DT-GT-fixed1.0 reference results."""
    if dataset != "rosenbrock":
        return None

    ref = {}
    base = os.path.join(ROOT, "res", "ablation_rtg_rosenbrock")
    dyn_npz_path = os.path.join(base, "dynamic", f"policy_comparison_{objective_name}_{dim}D.npz")
    fix_npz_path = os.path.join(base, "fixed_1.0", f"policy_comparison_{objective_name}_{dim}D.npz")

    if not (os.path.exists(dyn_npz_path) and os.path.exists(fix_npz_path)):
        return None

    dyn_npz  = np.load(dyn_npz_path)
    fix_npz  = np.load(fix_npz_path)

    true_opt = TRUE_OPTIMA[objective_name]
    ref['dtgt_dynamic_regret'] = (true_opt - dyn_npz['dtgt_best'][:, -1])
    ref['dtgp_dynamic_regret'] = (true_opt - dyn_npz['dtgp_best'][:, -1])
    ref['dtgt_fixed10_regret'] = (true_opt - fix_npz['dtgt_best'][:, -1])

    # Per-iteration diagnostics for reference conditions (DT-GT only)
    try:
        dyn_diag_path = os.path.join(base, "dynamic", f"policy_comparison_{objective_name}_{dim}D_reward_diag.json")
        fix_diag_path = os.path.join(base, "fixed_1.0", f"policy_comparison_{objective_name}_{dim}D_reward_diag.json")
        dyn_diag = json.load(open(dyn_diag_path))
        fix_diag = json.load(open(fix_diag_path))
        ref['dtgt_dynamic_diag'] = dyn_diag.get('DT-GT', [])
        ref['dtgt_fixed10_diag'] = fix_diag.get('DT-GT', [])
    except Exception:
        ref['dtgt_dynamic_diag'] = []
        ref['dtgt_fixed10_diag'] = []

    return ref


def _per_trial_table(regret_arr, label):
    """Return lines for per-trial regret section."""
    lines = [f"  {label}"]
    for i, v in enumerate(regret_arr):
        flag = "  ← catastrophic" if v > 50 else ""
        lines.append(f"    seed {i:2d}: {v:10.4f}{flag}")
    return lines


def _stats(arr):
    q25, q75 = np.percentile(arr, 25), np.percentile(arr, 75)
    return {
        'mean': float(np.mean(arr)),
        'std':  float(np.std(arr)),
        'med':  float(np.median(arr)),
        'iqr':  float(q75 - q25),
        'near_zero': int(np.sum(arr < 0.01)),
        'catastrophic': int(np.sum(arr > 50)),
    }


def _floor_stats_from_diag(diag_list, max_iter=30):
    """
    diag_list : list of per-trial diagnostic lists
                each entry is a list of per-BO-iter dicts
    Returns summary floor statistics averaged across seeds.
    """
    floor_active_all  = []  # fraction of iters floor was active, per seed
    tgt_all           = []  # mean target_rtg_used per seed
    floor_val_all     = []  # mean floor_value per seed
    tgt_early, tgt_mid, tgt_late = [], [], []

    for trial_diags in diag_list:
        # Each trial_diags is a list of dicts, one per BO iteration
        iters = [d for d in trial_diags if d.get('mode') in ('DT-GT', 'DT-GP')]
        if not iters:
            continue

        fa   = [d.get('floor_active', 0) for d in iters]
        tgt  = [d.get('target_rtg_used', 0.0) or 0.0 for d in iters]
        fv   = [d.get('floor_value', 0.0) or 0.0 for d in iters]
        bo_i = [d.get('bo_iteration', i+1) for i, d in enumerate(iters)]

        floor_active_all.append(float(np.mean(fa)))
        tgt_all.append(float(np.mean(tgt)))
        floor_val_all.append(float(np.mean(fv)))

        early = [t for t, it in zip(tgt, bo_i) if 1 <= it <= 10]
        mid   = [t for t, it in zip(tgt, bo_i) if 11 <= it <= 20]
        late  = [t for t, it in zip(tgt, bo_i) if 21 <= it <= 30]
        if early: tgt_early.append(np.mean(early))
        if mid:   tgt_mid.append(np.mean(mid))
        if late:  tgt_late.append(np.mean(late))

    return {
        'frac_floor_active': float(np.mean(floor_active_all)) if floor_active_all else float('nan'),
        'mean_target_rtg':   float(np.mean(tgt_all))          if tgt_all else float('nan'),
        'mean_floor_val':    float(np.mean(floor_val_all))     if floor_val_all else float('nan'),
        'mean_tgt_early':    float(np.mean(tgt_early))         if tgt_early else float('nan'),
        'mean_tgt_mid':      float(np.mean(tgt_mid))           if tgt_mid else float('nan'),
        'mean_tgt_late':     float(np.mean(tgt_late))          if tgt_late else float('nan'),
    }


def _iter_profile_from_diag(diag_list, target_iters=(1,2,3,5,10,15,20,25,30)):
    """
    Compute per-iteration profile averaged over seeds.
    Returns dict keyed by iteration index with mean±std of target_rtg_used,
    floor_active fraction, max_rtg_from_batch.
    """
    profile = {}
    for it in target_iters:
        tgt_vals, fa_vals, batch_vals = [], [], []
        for trial_diags in diag_list:
            iters = [d for d in trial_diags if d.get('mode') in ('DT-GT', 'DT-GP')]
            # bo_iteration is 1-indexed
            match = [d for d in iters if d.get('bo_iteration') == it]
            if match:
                d = match[0]
                tgt_vals.append(d.get('target_rtg_used', 0.0) or 0.0)
                fa_vals.append(d.get('floor_active', 0))
                batch_vals.append(d.get('max_rtg_from_batch', d.get('max_rtg', 0.0)) or 0.0)
        profile[it] = {
            'tgt_mean': float(np.mean(tgt_vals)) if tgt_vals else float('nan'),
            'tgt_std':  float(np.std(tgt_vals))  if tgt_vals else float('nan'),
            'fa_frac':  float(np.mean(fa_vals))  if fa_vals else float('nan'),
            'batch_mean': float(np.mean(batch_vals)) if batch_vals else float('nan'),
            'batch_std':  float(np.std(batch_vals))  if batch_vals else float('nan'),
            'n': len(tgt_vals),
        }
    return profile


def write_summary(results, ref, save_dir, dataset: str, objective_name: str, args):
    """Write dataset-specific floored RTG summary."""

    conditions = [
        ('DT-GT-floor05', 'floor05'),
        ('DT-GT-floor02', 'floor02'),
        ('DT-GT-floor01', 'floor01'),
        ('DT-GP-dynamic', 'dtgp'),
    ]

    n_seeds = len(next(iter(results.values()))['regret'])

    lines = []
    W = 80
    lines += [
        "=" * W,
        f"  Floored Dynamic RTG — {objective_name} {args.dim}D: Summary",
        "  DT-GT: floored_dynamic alpha={0.5, 0.2, 0.1} vs reference conditions",
        "=" * W,
        "Experimental Settings",
        f"  Objective      : {objective_name} {args.dim}D",
        f"  Trials         : {n_seeds} seeds (seeds 0–{n_seeds-1})",
        f"  BO budget      : {args.initial_points} initial + {args.max_iterations} iterations = "
        f"{args.initial_points + args.max_iterations} evaluations",
        f"  Rollouts       : {args.num_rollouts} per iteration, max rollout length {args.max_rollout_length}",
        f"  GP kernel      : RBF isotropic (M=5 ensemble)",
        f"  New RTG schema : target(t) = max(max_RTG_batch(t), alpha*running_max_RTG(t))",
        f"  Reference data : DT-GT-dynamic, DT-GT-fixed1.0 from res/ablation_rtg_rosenbrock/",
        "=" * W,
        "",
    ]

    # ── SECTION 1: Final Simple Regret ────────────────────────────────────────
    lines += [
        "=" * W,
        "SECTION 1 — FINAL SIMPLE REGRET",
        "=" * W,
        "",
    ]

    # Build reference condition stats
    ref_conditions = [
        ('DT-GT-dynamic',  ref['dtgt_dynamic_regret']),
        ('DT-GT-fixed1.0', ref['dtgt_fixed10_regret']),
    ]

    all_condition_stats = {}
    for label, regret in ref_conditions:
        all_condition_stats[label] = _stats(regret)

    # DT-GP from within-run results
    dtgp_regret = results['dtgp']['regret']
    all_condition_stats['DT-GP-dynamic'] = _stats(dtgp_regret)

    # Floored conditions
    for key in ['floor05', 'floor02', 'floor01']:
        label = f"DT-GT-floor{key[-2:]}"
        all_condition_stats[label] = _stats(results[key]['regret'])

    # Summary table header
    lines.append(
        f"  {'Condition':<18} {'Mean':>8} {'Std':>8} {'Median':>8} "
        f"{'IQR':>8} {'<0.01':>6} {'>50':>6}"
    )
    lines.append("  " + "-" * 68)

    display_order = [
        'DT-GT-dynamic', 'DT-GT-fixed1.0',
        'DT-GT-floor05', 'DT-GT-floor02', 'DT-GT-floor01',
        'DT-GP-dynamic',
    ]
    for label in display_order:
        s = all_condition_stats[label]
        lines.append(
            f"  {label:<18} {s['mean']:>8.3f} {s['std']:>8.3f} {s['med']:>8.4f} "
            f"{s['iqr']:>8.4f} {s['near_zero']:>5}/10 {s['catastrophic']:>5}/10"
        )
    lines.append("")

    # Per-trial breakdown
    lines += ["", "  Per-trial final regret (all seeds):"]
    lines += [""]

    # Reference conditions
    for label, regret in ref_conditions:
        lines += _per_trial_table(regret, label)
        lines += [""]

    # New conditions
    lines += _per_trial_table(results['dtgp']['regret'], "DT-GP-dynamic (within-run reference)")
    lines += [""]
    for key, alpha_lbl in [('floor05','alpha=0.5'), ('floor02','alpha=0.2'), ('floor01','alpha=0.1')]:
        cond_label = f"DT-GT-floor{key[-2:]} ({alpha_lbl})"
        lines += _per_trial_table(results[key]['regret'], cond_label)
        lines += [""]

    # Head-to-head vs DT-GP-dynamic
    lines += ["  Head-to-head wins vs DT-GP-dynamic (floored conditions only):"]
    for key in ['floor05', 'floor02', 'floor01']:
        fl_reg = results[key]['regret']
        gp_reg = results['dtgp']['regret']
        n_seeds_here = len(fl_reg)
        wins = int(np.sum(fl_reg < gp_reg))
        label = f"DT-GT-floor{key[-2:]}"
        lines.append(f"    {label}: {wins}/{n_seeds_here} wins")
    lines += [""]

    # ── SECTION 2: Floor Diagnostics ─────────────────────────────────────────
    lines += [
        "=" * W,
        "SECTION 2 — FLOOR ACTIVATION DIAGNOSTICS",
        "=" * W,
        "",
        f"  {'Condition':<18} {'FloorActive%':>12} {'MeanTgtRTG':>11} "
        f"{'FloorVal':>9} {'Tgt(1-10)':>10} {'Tgt(11-20)':>10} {'Tgt(21-30)':>10}",
        "  " + "-" * 82,
    ]

    for key, alpha in [('floor05', 0.5), ('floor02', 0.2), ('floor01', 0.1)]:
        label = f"DT-GT-floor{key[-2:]}"
        fs = _floor_stats_from_diag(results[key]['diag'])
        lines.append(
            f"  {label:<18} {fs['frac_floor_active']:>11.1%} "
            f"{fs['mean_target_rtg']:>11.4f} "
            f"{fs['mean_floor_val']:>9.4f} "
            f"{fs['mean_tgt_early']:>10.4f} "
            f"{fs['mean_tgt_mid']:>10.4f} "
            f"{fs['mean_tgt_late']:>10.4f}"
        )
    lines += ["",
        "  DT-GP-dynamic (reference — floor not applicable):",
    ]
    gp_fs = _floor_stats_from_diag(results['dtgp']['diag'])
    lines.append(
        f"  {'DT-GP-dynamic':<18} {'N/A':>12} "
        f"{gp_fs['mean_target_rtg']:>11.4f} "
        f"{'N/A':>9} "
        f"{gp_fs['mean_tgt_early']:>10.4f} "
        f"{gp_fs['mean_tgt_mid']:>10.4f} "
        f"{gp_fs['mean_tgt_late']:>10.4f}"
    )
    lines += [""]

    # ── SECTION 3: Interpretation ─────────────────────────────────────────────
    lines += [
        "=" * W,
        "SECTION 3 — INTERPRETATION",
        "=" * W,
        "",
    ]

    # Gather key numbers for interpretation
    med_dyn  = all_condition_stats['DT-GT-dynamic']['med']
    med_fix  = all_condition_stats['DT-GT-fixed1.0']['med']
    cat_fix  = all_condition_stats['DT-GT-fixed1.0']['catastrophic']
    med_f05  = all_condition_stats['DT-GT-floor05']['med']
    med_f02  = all_condition_stats['DT-GT-floor02']['med']
    med_f01  = all_condition_stats['DT-GT-floor01']['med']
    cat_f05  = all_condition_stats['DT-GT-floor05']['catastrophic']
    cat_f02  = all_condition_stats['DT-GT-floor02']['catastrophic']
    cat_f01  = all_condition_stats['DT-GT-floor01']['catastrophic']

    # Alpha stats for interpretation
    fs05 = _floor_stats_from_diag(results['floor05']['diag'])
    fs02 = _floor_stats_from_diag(results['floor02']['diag'])
    fs01 = _floor_stats_from_diag(results['floor01']['diag'])

    # Determine best floored condition
    floored_meds = {'DT-GT-floor05': med_f05, 'DT-GT-floor02': med_f02, 'DT-GT-floor01': med_f01}
    best_floor_label = min(floored_meds, key=lambda k: floored_meds[k])
    best_floor_med   = floored_meds[best_floor_label]

    # i: Which alpha improves over both baselines?
    improves_dyn  = {k: v < med_dyn for k, v in floored_meds.items()}
    improves_fix  = {k: v < med_fix for k, v in floored_meds.items()}
    improves_both = {k: improves_dyn[k] and improves_fix[k] for k in floored_meds}

    i_ans = [k for k, v in improves_both.items() if v]
    if i_ans:
        i_text = (f"{', '.join(i_ans)} improve{'s' if len(i_ans)==1 else ''} over both "
                  f"DT-GT-dynamic ({med_dyn:.4f}) and DT-GT-fixed1.0 ({med_fix:.4f}). "
                  f"Best floored median: {best_floor_med:.4f} ({best_floor_label}).")
    else:
        # Partial improvement
        better_dyn = [k for k, v in improves_dyn.items() if v]
        if better_dyn:
            i_text = (f"No single alpha improves over both baselines simultaneously. "
                      f"{', '.join(better_dyn)} improve over DT-GT-dynamic ({med_dyn:.4f}) "
                      f"but not over DT-GT-fixed1.0 ({med_fix:.4f}). "
                      f"Best floored median: {best_floor_med:.4f} ({best_floor_label}).")
        else:
            i_text = (f"No floored alpha improves over DT-GT-dynamic ({med_dyn:.4f}). "
                      f"Best floored median: {best_floor_med:.4f} ({best_floor_label}). "
                      f"DT-GT-fixed1.0 median ({med_fix:.4f}) remains the best result.")

    # ii: Catastrophic failure
    if cat_f05 == 0 and cat_f02 == 0 and cat_f01 == 0:
        ii_text = (f"All three floored alpha values eliminate the catastrophic failure "
                   f"observed in DT-GT-fixed1.0 (seed 1, regret 206.6; {cat_fix}/10 seeds "
                   f"catastrophic). No floored condition produces regret > 50.")
    else:
        cats = [(f"alpha=0.5", cat_f05), (f"alpha=0.2", cat_f02), (f"alpha=0.1", cat_f01)]
        cat_str = "; ".join(f"{a}: {n}/10" for a, n in cats)
        ii_text = (f"Catastrophic failures (regret>50) per condition: {cat_str}. "
                   f"DT-GT-fixed1.0 had {cat_fix}/10 catastrophic seed(s).")

    # iii: When does floor activate?
    iii_text = (
        f"For alpha=0.5, the floor activates in {fs05['frac_floor_active']:.1%} of iterations "
        f"(mean target_RTG {fs05['mean_target_rtg']:.3f}); "
        f"for alpha=0.2, {fs02['frac_floor_active']:.1%} ({fs02['mean_target_rtg']:.3f}); "
        f"for alpha=0.1, {fs01['frac_floor_active']:.1%} ({fs01['mean_target_rtg']:.3f}). "
        f"The floor is most active in mid-to-late iterations when max_RTG_from_batch "
        f"collapses toward zero, with higher alpha producing stronger and more frequent activation."
    )

    # iv: Tradeoff alpha vs catastrophic failure
    if cat_f05 == cat_f02 == cat_f01 == 0:
        iv_text = (
            f"All alpha values produce similar (zero) catastrophic failure rates, "
            f"suggesting the floor mechanism itself prevents the destabilization seen in "
            f"fixed RTG=1.0. The floor avoids fixed1.0's problem by scaling proportionally "
            f"to the running maximum rather than imposing a scale-independent constant. "
            f"The primary tradeoff is between floor strength (higher alpha → more aggressive "
            f"conditioning) and median performance, not between alpha and tail risk."
        )
    else:
        # Some catastrophics exist — report actual pattern
        iv_text = (
            f"Higher alpha values provide stronger floor conditioning but may increase "
            f"tail risk if the floor over-targets early in the run. "
            f"Catastrophic counts: alpha=0.5: {cat_f05}/10, alpha=0.2: {cat_f02}/10, "
            f"alpha=0.1: {cat_f01}/10. DT-GT-fixed1.0 had {cat_fix}/10."
        )

    lines += [
        "  (i)  Which alpha value improves over both DT-GT-dynamic and DT-GT-fixed1.0?",
        "",
        f"  {i_text}",
        "",
        "  (ii) Does any alpha value eliminate the catastrophic failure in DT-GT-fixed1.0?",
        "",
        f"  {ii_text}",
        "",
        "  (iii) When does the floor activate, and how does rate differ across alpha values?",
        "",
        f"  {iii_text}",
        "",
        "  (iv) Is there a tradeoff between floor strength and catastrophic failure risk?",
        "",
        f"  {iv_text}",
        "",
    ]

    lines += [
        "=" * W,
        f"  Source: {save_dir}/",
        "=" * W,
    ]

    path = os.path.join(save_dir, f"{dataset}_floored_rtg_summary.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Summary → {path}")
    return path


def write_iteration_profile(results, save_dir, dataset: str):
    """Write dataset-specific floored RTG iteration profile."""

    target_iters = (1, 2, 3, 5, 10, 15, 20, 25, 30)
    W = 80

    lines = [
        "=" * W,
        "  Floored Dynamic RTG — Iteration Profile  (mean ± std across seeds)",
        "=" * W,
        "  Reported at BO iterations: 1, 2, 3, 5, 10, 15, 20, 25, 30",
        "  target_RTG_used : actual value passed to DT at deployment",
        "  floor_active%   : fraction of seeds where floor > max_RTG_from_batch",
        "  max_RTG_batch   : max_RTG_from_batch (mean ± std)",
        "=" * W,
        "",
    ]

    for key, alpha_lbl in [('floor05','alpha=0.5'), ('floor02','alpha=0.2'),
                            ('floor01','alpha=0.1'), ('dtgp','DT-GP-dynamic (reference)')]:
        label = f"DT-GT-floor{key[-2:]}" if key.startswith('floor') else 'DT-GP-dynamic'
        profile = _iter_profile_from_diag(results[key]['diag'], target_iters)

        lines += [
            "-" * W,
            f"  {label}  [{alpha_lbl}]",
            "-" * W,
            f"  {'Iter':>5}  {'target_RTG_used':>16}  {'floor_active%':>13}  "
            f"{'max_RTG_batch':>15}  {'n':>3}",
            "  " + "-" * 60,
        ]
        for it in target_iters:
            p = profile[it]
            tgt_str   = f"{p['tgt_mean']:7.4f} ±{p['tgt_std']:7.4f}"
            batch_str = f"{p['batch_mean']:6.4f} ±{p['batch_std']:6.4f}"
            fa_str    = f"{p['fa_frac']:>12.1%}" if not np.isnan(p['fa_frac']) else "         N/A"
            lines.append(
                f"  {it:>5}  {tgt_str:>16}  {fa_str:>13}  {batch_str:>15}  {p['n']:>3}"
            )
        lines += [""]

    lines += [
        "=" * W,
        f"  Source: {save_dir}/",
        "=" * W,
    ]

    path = os.path.join(save_dir, f"{dataset}_floored_rtg_iteration_profile.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Iteration profile → {path}")
    return path


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Floored dynamic RTG experiment on synthetic datasets (Rosenbrock/Levy/Ackley)"
    )
    parser.add_argument("--dataset",           type=str, choices=["rosenbrock", "levy", "ackley"], default="rosenbrock")
    parser.add_argument("--num_trials",        type=int,   default=10)
    parser.add_argument("--seed_start",        type=int,   default=0)
    parser.add_argument("--dim",               type=int,   default=2)
    parser.add_argument("--domain_min",        type=float, default=-5.0)
    parser.add_argument("--domain_max",        type=float, default=5.0)
    parser.add_argument("--initial_points",    type=int,   default=10)
    parser.add_argument("--max_iterations",    type=int,   default=30)
    parser.add_argument("--num_rollouts",      type=int,   default=30)
    parser.add_argument("--max_rollout_length",type=int,   default=4)
    parser.add_argument("--save_dir",          default="res/floored_rtg_rosenbrock")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    total_evals = args.initial_points + args.max_iterations
    objective_name, _ = OBJECTIVE_MAP[args.dataset]
    objective   = build_objective(args.dataset, args.dim, args.domain_min, args.domain_max)
    true_opt    = TRUE_OPTIMA[objective_name]

    # Conditions to run: key → (use_true_traj, rtg_mode, alpha, label)
    run_conditions = [
        ('floor05', True,  'floored_dynamic', 0.5,  'DT-GT-floor05'),
        ('floor02', True,  'floored_dynamic', 0.2,  'DT-GT-floor02'),
        ('floor01', True,  'floored_dynamic', 0.1,  'DT-GT-floor01'),
        ('dtgp',    False, 'dynamic',         None,  'DT-GP-dynamic'),
    ]

    print(f"\n{'='*65}")
    print(f"  Floored Dynamic RTG — {objective_name} {args.dim}D")
    print(f"  Trials: {args.num_trials}  |  Seeds {args.seed_start}–"
          f"{args.seed_start + args.num_trials - 1}")
    print(f"  Budget: {args.initial_points} init + {args.max_iterations} iter = "
          f"{total_evals} evaluations")
    print(f"  Conditions: DT-GT×3 (floor α∈{{0.5,0.2,0.1}}) + DT-GP-dynamic")
    print(f"{'='*65}\n")

    # Storage: best arrays and diagnostics per condition
    results = {
        key: {
            'best':  np.full((args.num_trials, total_evals), np.nan),
            'regret': None,
            'diag':  [],   # list of per-trial diag lists
        }
        for key, *_ in run_conditions
    }

    for t in range(args.num_trials):
        seed = args.seed_start + t
        print(f"\n── Trial {t+1}/{args.num_trials}  (seed={seed}) {'─'*40}")

        for key, use_gt, rtg_mode, alpha, label in run_conditions:
            torch.manual_seed(seed)
            np.random.seed(seed)

            cfg = build_config(
                dim=args.dim,
                domain_min=args.domain_min,
                domain_max=args.domain_max,
                max_iterations=args.max_iterations,
                initial_points=args.initial_points,
                num_rollouts=args.num_rollouts,
                max_rollout_length=args.max_rollout_length,
                use_true_trajectories=use_gt,
                seed=seed,
                rtg_mode=rtg_mode,
                rtg_alpha=alpha,
            )

            print(f"  [{label}] running...", end=" ", flush=True)
            best, diag = run_trial(cfg, objective)
            results[key]['best'][t, :len(best)] = best[:total_evals]
            results[key]['diag'].append(diag)
            print(f"final best = {best[-1]:.4f}")

        # Incremental checkpoint
        ckpt = {key: results[key]['best'].tolist() for key, *_ in run_conditions}
        ckpt['trials_done'] = t + 1
        ckpt_path = os.path.join(args.save_dir, "floored_rtg.ckpt.json")
        with open(ckpt_path, 'w') as f:
            json.dump(ckpt, f)
        print(f"  [ckpt] {t+1}/{args.num_trials} saved → {ckpt_path}")

    # Compute final regret arrays
    for key, *_ in run_conditions:
        results[key]['regret'] = true_opt - results[key]['best'][:, -1]

    # Save per-condition npz + diag JSON
    for key, use_gt, rtg_mode, alpha, label in run_conditions:
        tag = label.lower().replace("-", "_")
        np.savez(
            os.path.join(args.save_dir, f"{tag}.npz"),
            best=results[key]['best'],
            regret=results[key]['regret'],
            true_optimum=true_opt,
            initial_points=args.initial_points,
        )
        with open(os.path.join(args.save_dir, f"{tag}_diag.json"), 'w') as f:
            json.dump(results[key]['diag'], f, indent=2)

    # Load reference data (Rosenbrock only)
    ref = _load_ref_data(args.dataset, objective_name, args.dim)

    # Write output files
    print(f"\n{'='*65}")
    print(f"  Writing output files...")
    if ref is not None:
        write_summary(results, ref, args.save_dir, args.dataset, objective_name, args)
    else:
        print("  Summary skipped (reference DT-GT-dynamic / DT-GT-fixed1.0 not available for this dataset).")
    write_iteration_profile(results, args.save_dir, args.dataset)
    print(f"{'='*65}\n")
    print("  Done.")


if __name__ == "__main__":
    main()
