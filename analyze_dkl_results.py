"""
analyze_dkl_results.py

Generates DKL analysis output files after the experiment runs complete.
Covers Rosenbrock 2D, Ackley 2D, and Levy 2D.

Output files:
  res/dkl_rosenbrock/policy_comparison_Rosenbrock_2D_dkl.txt
  res/dkl_ackley/policy_comparison_Ackley_2D_dkl.txt
  res/dkl_levy/policy_comparison_Levy_2D_dkl.txt
  res/dkl_rosenbrock/dkl_reward_diagnostics_summary.txt
  res/dkl_ackley/dkl_reward_diagnostics_summary.txt
  res/dkl_levy/dkl_reward_diagnostics_summary.txt
  res/dkl_comparison_summary.txt

Run with:
    .venv/bin/python analyze_dkl_results.py
"""

import os
import json
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _final_regret(npz):
    """Return (gp_regret, gt_regret) as 1-D arrays of final simple regret."""
    gp = (npz['true_optimum'] - npz['dtgp_best'])[:, -1]
    gt = (npz['true_optimum'] - npz['dtgt_best'])[:, -1]
    return gp, gt


def _stats(arr):
    arr = np.asarray(arr, dtype=float)
    valid = arr[np.isfinite(arr)]
    n = len(valid)
    if n == 0:
        return dict(n=0, mean=np.nan, std=np.nan, se=np.nan, median=np.nan,
                    q25=np.nan, q75=np.nan, iqr=np.nan, min=np.nan, max=np.nan)
    return dict(
        n=n,
        mean=np.mean(valid),
        std=np.std(valid),
        se=np.std(valid) / max(np.sqrt(n), 1),
        median=np.median(valid),
        q25=np.percentile(valid, 25),
        q75=np.percentile(valid, 75),
        iqr=np.percentile(valid, 75) - np.percentile(valid, 25),
        min=np.min(valid),
        max=np.max(valid),
    )


def _h2h_wins(arr_a, arr_b):
    valid = [(a, b) for a, b in zip(arr_a, arr_b) if np.isfinite(a) and np.isfinite(b)]
    if not valid:
        return np.nan, np.nan, np.nan
    N = len(valid)
    wins_a = sum(a < b for a, b in valid) / N
    wins_b = sum(b < a for a, b in valid) / N
    ties   = sum(a == b for a, b in valid) / N
    return wins_a, wins_b, ties


def _near_zero(arr, thresh=0.1):
    v = np.asarray(arr, dtype=float)
    return float(np.mean(v[np.isfinite(v)] < thresh))


def _catastrophic(arr, thresh=50.0):
    v = np.asarray(arr, dtype=float)
    return int(np.sum(v[np.isfinite(v)] > thresh))


# ─────────────────────────────────────────────────────────────────────────────
# DKL diagnostics analysis
# ─────────────────────────────────────────────────────────────────────────────

def _analyze_dkl_diag(diag_path):
    with open(diag_path) as f:
        raw = json.load(f)

    result = {}
    for mode, trials in raw.items():
        train_losses, latent_norms = [], []
        ls_per_member = []
        warm_start_flags = []
        early, mid, late = [], [], []

        for trial in trials:
            n = len(trial)
            for idx, rec in enumerate(trial):
                tl = rec.get("dkl_train_loss")
                ln = rec.get("dkl_latent_norm")
                ws = rec.get("dkl_warm_start_used", False)
                ls_raw = rec.get("dkl_lengthscales", [])

                if tl is not None and np.isfinite(tl):
                    train_losses.append(tl)
                    frac = idx / max(n - 1, 1)
                    (early if frac < 0.33 else mid if frac < 0.67 else late).append(tl)

                if ln is not None and np.isfinite(ln):
                    latent_norms.append(ln)

                warm_start_flags.append(ws)

                if ls_raw:
                    ls_per_member.append([float(np.mean(dims)) for dims in ls_raw])

        ls_diversity = [float(np.std(pm)) for pm in ls_per_member if len(pm) > 1]

        result[mode] = {
            "train_loss": {
                "mean":       float(np.mean(train_losses)) if train_losses else np.nan,
                "std":        float(np.std(train_losses))  if train_losses else np.nan,
                "early_mean": float(np.mean(early)) if early else np.nan,
                "mid_mean":   float(np.mean(mid))   if mid   else np.nan,
                "late_mean":  float(np.mean(late))  if late  else np.nan,
            },
            "latent_norm": {
                "mean": float(np.mean(latent_norms)) if latent_norms else np.nan,
                "std":  float(np.std(latent_norms))  if latent_norms else np.nan,
            },
            "ls_diversity": {
                "mean": float(np.mean(ls_diversity)) if ls_diversity else np.nan,
                "std":  float(np.std(ls_diversity))  if ls_diversity else np.nan,
            },
            "warm_start_fraction": float(np.mean(warm_start_flags)) if warm_start_flags else np.nan,
        }
    return result


def _analyze_reward_diag(diag_path):
    with open(diag_path) as f:
        raw = json.load(f)

    result = {}
    for mode, trials in raw.items():
        zero_steps, mean_rtgs, max_rtgs = [], [], []
        for trial in trials:
            for rec in trial:
                if not rec:
                    continue
                zs = rec.get("frac_zero_steps")
                mr = rec.get("mean_rtg")
                mx = rec.get("max_rtg")
                if zs is not None:
                    zero_steps.append(zs)
                if mr is not None and np.isfinite(mr):
                    mean_rtgs.append(mr)
                if mx is not None and np.isfinite(mx):
                    max_rtgs.append(mx)

        floor_active = float(np.mean([z == 1.0 for z in zero_steps])) if zero_steps else np.nan
        result[mode] = {
            "floor_active_pct": floor_active * 100,
            "mean_rtg_mean":   float(np.mean(mean_rtgs)) if mean_rtgs else np.nan,
            "mean_rtg_std":    float(np.std(mean_rtgs))  if mean_rtgs else np.nan,
            "max_rtg_mean":    float(np.mean(max_rtgs))  if max_rtgs else np.nan,
        }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# File writers
# ─────────────────────────────────────────────────────────────────────────────

def _write_comparison_txt(path, objective, dim, num_trials, total_evals,
                          initial_points, gp_regret, gt_regret,
                          label_gp="DT-GP-DKL", label_gt="DT-GT-DKL"):
    sg  = _stats(gp_regret)
    sgt = _stats(gt_regret)
    wins_gp, wins_gt, ties = _h2h_wins(gp_regret, gt_regret)
    W = 70
    sep, dash = "=" * W, "-" * W

    lines = [
        f"DKL Policy Comparison — {objective} {dim}D",
        f"Conditions : {label_gp}  |  {label_gt}",
        f"Trials     : {num_trials}  |  Budget: {total_evals} evals per condition",
        sep,
        f"{'Metric':<42} {label_gp:>12} {label_gt:>12}",
        dash,
        f"{'Final simple regret  (mean)':<42} {sg['mean']:>12.4f} {sgt['mean']:>12.4f}",
        f"{'  ± std':<42} {sg['std']:>12.4f} {sgt['std']:>12.4f}",
        f"{'  ± SE':<42} {sg['se']:>12.4f} {sgt['se']:>12.4f}",
        f"{'Final simple regret  (median) *':<42} {sg['median']:>12.4f} {sgt['median']:>12.4f}",
        f"{'  IQR [Q25 – Q75]':<42} "
        f"[{sg['q25']:.4f}–{sg['q75']:.4f}]  [{sgt['q25']:.4f}–{sgt['q75']:.4f}]",
        f"{'  range [min – max]':<42} "
        f"[{sg['min']:.3f}–{sg['max']:.3f}]    [{sgt['min']:.3f}–{sgt['max']:.3f}]",
        dash,
        f"{'Near-zero regret  (<0.1, %)':<42} "
        f"{_near_zero(gp_regret)*100:>11.1f}% {_near_zero(gt_regret)*100:>11.1f}%",
        f"{'Catastrophic failures (>50)':<42} "
        f"{_catastrophic(gp_regret):>12d} {_catastrophic(gt_regret):>12d}",
        f"{'Head-to-head wins (lower regret)':<42} "
        f"{wins_gp*100:>11.1f}% {wins_gt*100:>11.1f}%",
        f"{'  ties':<42} {ties*100:>11.1f}%",
        sep,
        "",
        "* Median is robust to collapse seeds — use this for the paper.",
        "",
        "Per-trial final simple regret:",
        f"  {'Seed':>4}  {label_gp:>12}  {label_gt:>12}",
        "  " + "-" * 32,
    ]

    for i, (g, t) in enumerate(zip(gp_regret, gt_regret)):
        tag = ""
        if np.isfinite(g) and g > 50:
            tag += f" ← {label_gp} catastrophic"
        if np.isfinite(t) and t > 50:
            tag += f" ← {label_gt} catastrophic"
        lines.append(f"  {i:>4}  {g:>12.4f}  {t:>12.4f}{tag}")

    lines += ["", sep]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Written: {path}")


def _write_diag_summary(path, objective, dkl_diag, reward_diag):
    W = 72
    sep = "=" * W

    lines = [
        f"DKL Reward + Training Diagnostics Summary — {objective}",
        "Averaged over all trials and all BO iterations",
        sep,
    ]

    for mode in ("DT-GP", "DT-GT"):
        label = "DT-GP-DKL" if mode == "DT-GP" else "DT-GT-DKL"
        dd  = dkl_diag.get(mode, {})
        rd  = reward_diag.get(mode, {})
        tl  = dd.get("train_loss", {})
        ln  = dd.get("latent_norm", {})
        lsd = dd.get("ls_diversity", {})

        lines += [
            "",
            f"── {label} ─────────────────────────────────────────────",
            "",
            "  DKL Training Loss (negative MLL):",
            f"    early-iterations mean : {tl.get('early_mean', np.nan):.4g}",
            f"    mid-iterations  mean  : {tl.get('mid_mean',   np.nan):.4g}",
            f"    late-iterations mean  : {tl.get('late_mean',  np.nan):.4g}",
            f"    overall  mean ± std   : {tl.get('mean', np.nan):.4g} ± {tl.get('std', np.nan):.4g}",
            "",
            "  Latent Norm (should be ≈ 1.0, L2-normalised extractor):",
            f"    mean ± std : {ln.get('mean', np.nan):.6f} ± {ln.get('std', np.nan):.2e}",
            "",
            "  Ensemble Lengthscale Diversity (std across M members):",
            f"    mean std across members : {lsd.get('mean', np.nan):.4f}",
            f"    ± (std of that std)     : {lsd.get('std',  np.nan):.4f}",
            "",
            "  RTG / Reward density:",
            f"    floor_active (zero_steps=100%) : {rd.get('floor_active_pct', np.nan):.1f}%",
            f"    mean_RTG  mean ± std           : {rd.get('mean_rtg_mean', np.nan):.4f} ± {rd.get('mean_rtg_std', np.nan):.4f}",
            f"    max_RTG   mean                 : {rd.get('max_rtg_mean', np.nan):.4f}",
        ]

    lines += ["", sep]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Written: {path}")


def _write_consolidated_summary(path,
                                rosenb_dkl_gp, rosenb_dkl_gt,
                                ackley_dkl_gp,  ackley_dkl_gt,
                                levy_dkl_gp,    levy_dkl_gt):
    def _load_regret_1d(fpath):
        d = np.load(fpath)
        r = d['regret']
        return r[:, -1] if r.ndim == 2 else r

    base = os.path.join(ROOT, "res")

    refs = {}
    refs["rosenb_gpbo"]     = _load_regret_1d(f"{base}/gp_bo_baseline/gp_bo_Rosenbrock_2D.npz")
    refs["rosenb_dtgp_iso"] = 0.0 - np.load(f"{base}/policy_comparison_rosenbrock/policy_comparison_Rosenbrock_2D.npz")["dtgp_best"][:, -1]
    refs["rosenb_dtgt_iso"] = 0.0 - np.load(f"{base}/policy_comparison_rosenbrock/policy_comparison_Rosenbrock_2D.npz")["dtgt_best"][:, -1]
    refs["rosenb_dtgp_ard"] = 0.0 - np.load(f"{base}/ablation_ard_rosenbrock/policy_comparison_Rosenbrock_2D.npz")["dtgp_best"][:, -1]
    refs["rosenb_dtgt_ard"] = 0.0 - np.load(f"{base}/ablation_ard_rosenbrock/policy_comparison_Rosenbrock_2D.npz")["dtgt_best"][:, -1]
    refs["rosenb_floor05"]  = _load_regret_1d(f"{base}/floored_rtg_rosenbrock/dt_gt_floor05.npz")
    refs["ackley_gpbo"]     = _load_regret_1d(f"{base}/gp_bo_baseline/gp_bo_Ackley_2D.npz")
    refs["ackley_dtgp_iso"] = 0.0 - np.load(f"{base}/policy_comparison_ackley/policy_comparison_Ackley_2D.npz")["dtgp_best"][:, -1]
    refs["ackley_dtgt_iso"] = 0.0 - np.load(f"{base}/policy_comparison_ackley/policy_comparison_Ackley_2D.npz")["dtgt_best"][:, -1]
    refs["levy_gpbo"]       = _load_regret_1d(f"{base}/gp_bo_baseline/gp_bo_Levy_2D.npz")
    # Levy isotropic reference — use policy_comparison_levy if it exists
    levy_iso_path = f"{base}/policy_comparison_levy/policy_comparison_Levy_2D.npz"
    if os.path.exists(levy_iso_path):
        refs["levy_dtgp_iso"] = 0.0 - np.load(levy_iso_path)["dtgp_best"][:, -1]
        refs["levy_dtgt_iso"] = 0.0 - np.load(levy_iso_path)["dtgt_best"][:, -1]
    else:
        refs["levy_dtgp_iso"] = np.full(1, np.nan)
        refs["levy_dtgt_iso"] = np.full(1, np.nan)

    def _row(label, arr, source, W_label=26):
        s = _stats(arr)
        return (f"  {label:<{W_label}} | {s['median']:>7.3f} | {s['mean']:>7.3f} | "
                f"{s['std']:>8.3f} | {source:<12} | n={s['n']}")

    W = 78
    sep  = "=" * W
    hdr  = (f"  {'Condition':<26} | {'Median':>7} | {'Mean':>7} | "
            f"{'Std':>8} | {'Source':<12} | Trials")
    dash = "  " + "-" * (W - 2)

    lines = [
        "DKL Deep Kernel Comparison — Consolidated Results",
        "Final simple regret after full BO budget (10 Sobol + 30 BO = 40 evals)",
        sep,
        "",
        "══ ROSENBROCK 2D ══════════════════════════════════════════════════════════",
        hdr, dash,
        _row("GP-BO (logEI)",       refs["rosenb_gpbo"],     "existing"),
        _row("DT-GP isotropic",     refs["rosenb_dtgp_iso"], "existing"),
        _row("DT-GT isotropic",     refs["rosenb_dtgt_iso"], "existing"),
        _row("DT-GP ARD",           refs["rosenb_dtgp_ard"], "existing"),
        _row("DT-GT ARD",           refs["rosenb_dtgt_ard"], "existing"),
        _row("DT-GT floor05",       refs["rosenb_floor05"],  "existing"),
        _row("DT-GP-DKL  [NEW]",    rosenb_dkl_gp,           "new"),
        _row("DT-GT-DKL  [NEW]",    rosenb_dkl_gt,           "new"),
        "",
        "══ ACKLEY 2D ══════════════════════════════════════════════════════════════",
        hdr, dash,
        _row("GP-BO (logEI)",       refs["ackley_gpbo"],     "existing"),
        _row("DT-GP isotropic",     refs["ackley_dtgp_iso"], "existing"),
        _row("DT-GT isotropic",     refs["ackley_dtgt_iso"], "existing"),
        _row("DT-GP-DKL  [NEW]",    ackley_dkl_gp,           "new"),
        _row("DT-GT-DKL  [NEW]",    ackley_dkl_gt,           "new"),
        "",
        "══ LEVY 2D ════════════════════════════════════════════════════════════════",
        hdr, dash,
        _row("GP-BO (logEI)",       refs["levy_gpbo"],       "existing"),
        _row("DT-GP isotropic",     refs["levy_dtgp_iso"],   "existing"),
        _row("DT-GT isotropic",     refs["levy_dtgt_iso"],   "existing"),
        _row("DT-GP-DKL  [NEW]",    levy_dkl_gp,             "new"),
        _row("DT-GT-DKL  [NEW]",    levy_dkl_gt,             "new"),
        "",
        sep,
        "",
        "Notes:",
        "  - floor05 is DT-GT with RTG floor=0.5 (10-seed run); not directly comparable.",
        "  - All DKL runs: M=5, dkl_hidden_dim=32, dkl_latent_dim=4,",
        "    dkl_n_iter=100 (first iter) / 50 (warm-start), dkl_lr=0.01.",
        "  - Budget: 10 Sobol + 30 BO = 40 evaluations per condition.",
        "  - Levy isotropic reference shown as n=1 if run file not found.",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Written: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    base = os.path.join(ROOT, "res")

    benchmarks = [
        ("Rosenbrock", "2D", "dkl_rosenbrock"),
        ("Ackley",     "2D", "dkl_ackley"),
        ("Levy",       "2D", "dkl_levy"),
    ]

    # Check all required files exist
    missing = []
    for obj, dim, dname in benchmarks:
        stem = f"policy_comparison_{obj}_{dim}"
        d = os.path.join(base, dname)
        for fname in [f"{stem}.npz", f"{stem}_dkl_diagnostics.json", f"{stem}_reward_diag.json"]:
            p = os.path.join(d, fname)
            if not os.path.exists(p):
                missing.append(p)
    if missing:
        print("ERROR: the following expected files do not exist yet:")
        for f in missing:
            print(f"  {f}")
        return

    results = {}
    for obj, dim, dname in benchmarks:
        stem = f"policy_comparison_{obj}_{dim}"
        d    = os.path.join(base, dname)
        npz  = np.load(os.path.join(d, f"{stem}.npz"))
        results[obj] = {
            "dir":        d,
            "stem":       stem,
            "npz":        npz,
            "dkl_gp":     _final_regret(npz)[0],
            "dkl_gt":     _final_regret(npz)[1],
            "dkl_diag":   _analyze_dkl_diag(os.path.join(d, f"{stem}_dkl_diagnostics.json")),
            "reward_diag": _analyze_reward_diag(os.path.join(d, f"{stem}_reward_diag.json")),
        }

    print("\nGenerating DKL analysis files...")

    # Files 1-3: per-benchmark comparison + diagnostics
    for obj, _, _ in benchmarks:
        r = results[obj]
        npz = r["npz"]
        _write_comparison_txt(
            path=os.path.join(r["dir"], f"{r['stem']}_dkl.txt"),
            objective=obj, dim=2,
            num_trials=int(npz["dtgp_best"].shape[0]),
            total_evals=int(npz["dtgp_best"].shape[1]),
            initial_points=int(npz["initial_points"]),
            gp_regret=r["dkl_gp"], gt_regret=r["dkl_gt"],
        )
        _write_diag_summary(
            path=os.path.join(r["dir"], "dkl_reward_diagnostics_summary.txt"),
            objective=f"{obj} 2D",
            dkl_diag=r["dkl_diag"], reward_diag=r["reward_diag"],
        )

    # File 4: consolidated cross-benchmark summary
    _write_consolidated_summary(
        path=os.path.join(base, "dkl_comparison_summary.txt"),
        rosenb_dkl_gp=results["Rosenbrock"]["dkl_gp"],
        rosenb_dkl_gt=results["Rosenbrock"]["dkl_gt"],
        ackley_dkl_gp=results["Ackley"]["dkl_gp"],
        ackley_dkl_gt=results["Ackley"]["dkl_gt"],
        levy_dkl_gp=results["Levy"]["dkl_gp"],
        levy_dkl_gt=results["Levy"]["dkl_gt"],
    )

    print("\nDone. Output files:")
    all_paths = []
    for obj, _, _ in benchmarks:
        r = results[obj]
        all_paths += [
            os.path.join(r["dir"], f"{r['stem']}_dkl.txt"),
            os.path.join(r["dir"], "dkl_reward_diagnostics_summary.txt"),
        ]
    all_paths.append(os.path.join(base, "dkl_comparison_summary.txt"))
    for p in all_paths:
        print(f"  [{'OK' if os.path.exists(p) else 'MISSING'}] {p}")


if __name__ == "__main__":
    main()
