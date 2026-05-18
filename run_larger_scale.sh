#!/bin/bash
# =============================================================================
#  Larger-scale experiment suite
#  max_iterations : 100  (was 30)
#  num_models     : 10   (was 5)
#  num_trials     : 20 for policy/dkl/ard,  10 for ablation_rtg
#  save_dir       : res_larger_scale/<experiment>
# =============================================================================

set -e
PYTHON=.venv/bin/python

# ── 1. GP-BO Baseline ─────────────────────────────────────────────────────────
$PYTHON gp_bo_baseline.py \
    --objective Rosenbrock --max_iterations 100 --num_models 10 --num_trials 20 \
    --save_dir res_larger_scale/gp_bo_baseline

$PYTHON gp_bo_baseline.py \
    --objective Ackley --max_iterations 100 --num_models 10 --num_trials 20 \
    --save_dir res_larger_scale/gp_bo_baseline

$PYTHON gp_bo_baseline.py \
    --objective Levy --max_iterations 100 --num_models 10 --num_trials 20 \
    --save_dir res_larger_scale/gp_bo_baseline

# ── 2. Policy Comparison (DT-GP vs DT-GT, dynamic RTG) ───────────────────────
$PYTHON compare_policy_quality.py \
    --objective Rosenbrock --max_iterations 100 --num_models 10 --num_trials 20 \
    --num_rollouts 30 --max_rollout_length 4 \
    --save_dir res_larger_scale/policy_comparison_rosenbrock

$PYTHON compare_policy_quality.py \
    --objective Ackley --max_iterations 100 --num_models 10 --num_trials 20 \
    --num_rollouts 30 --max_rollout_length 4 \
    --save_dir res_larger_scale/policy_comparison_ackley

$PYTHON compare_policy_quality.py \
    --objective Levy --max_iterations 100 --num_models 10 --num_trials 20 \
    --num_rollouts 30 --max_rollout_length 4 \
    --save_dir res_larger_scale/policy_comparison_levy

# ── 3. DKL (Deep Kernel Learning) ────────────────────────────────────────────
$PYTHON compare_policy_quality.py \
    --objective Rosenbrock --max_iterations 100 --num_models 10 --num_trials 20 \
    --num_rollouts 30 --max_rollout_length 4 \
    --kernel_type deep_kernel \
    --save_dir res_larger_scale/dkl_rosenbrock

$PYTHON compare_policy_quality.py \
    --objective Ackley --max_iterations 100 --num_models 10 --num_trials 20 \
    --num_rollouts 30 --max_rollout_length 4 \
    --kernel_type deep_kernel \
    --save_dir res_larger_scale/dkl_ackley

$PYTHON compare_policy_quality.py \
    --objective Levy --max_iterations 100 --num_models 10 --num_trials 20 \
    --num_rollouts 30 --max_rollout_length 4 \
    --kernel_type deep_kernel \
    --save_dir res_larger_scale/dkl_levy

# ── 4. ARD Kernel Ablation (Rosenbrock only) ──────────────────────────────────
$PYTHON compare_policy_quality.py \
    --objective Rosenbrock --max_iterations 100 --num_models 10 --num_trials 20 \
    --num_rollouts 30 --max_rollout_length 4 \
    --kernel_type rbf_ard \
    --save_dir res_larger_scale/ablation_ard_rosenbrock

# ── 5. RTG Ablation: Rosenbrock ───────────────────────────────────────────────
$PYTHON compare_policy_quality.py \
    --objective Rosenbrock --max_iterations 100 --num_models 10 --num_trials 10 \
    --num_rollouts 30 --max_rollout_length 4 \
    --save_dir res_larger_scale/ablation_rtg_rosenbrock/dynamic

$PYTHON compare_policy_quality.py \
    --objective Rosenbrock --max_iterations 100 --num_models 10 --num_trials 10 \
    --num_rollouts 30 --max_rollout_length 4 \
    --target_rtg 1.0 \
    --save_dir res_larger_scale/ablation_rtg_rosenbrock/fixed_1.0

# ── 6. RTG Ablation: Ackley ───────────────────────────────────────────────────
$PYTHON compare_policy_quality.py \
    --objective Ackley --max_iterations 100 --num_models 10 --num_trials 10 \
    --num_rollouts 30 --max_rollout_length 4 \
    --save_dir res_larger_scale/ablation_rtg_ackley/dynamic

$PYTHON compare_policy_quality.py \
    --objective Ackley --max_iterations 100 --num_models 10 --num_trials 10 \
    --num_rollouts 30 --max_rollout_length 4 \
    --target_rtg 1.0 \
    --save_dir res_larger_scale/ablation_rtg_ackley/fixed_1.0

# ── 7. RTG Ablation: Levy ─────────────────────────────────────────────────────
$PYTHON compare_policy_quality.py \
    --objective Levy --max_iterations 100 --num_models 10 --num_trials 10 \
    --num_rollouts 30 --max_rollout_length 4 \
    --save_dir res_larger_scale/ablation_rtg_levy/dynamic

$PYTHON compare_policy_quality.py \
    --objective Levy --max_iterations 100 --num_models 10 --num_trials 10 \
    --num_rollouts 30 --max_rollout_length 4 \
    --target_rtg 1.0 \
    --save_dir res_larger_scale/ablation_rtg_levy/fixed_1.0

echo ""
echo "All experiments complete. Results in res_larger_scale/"
