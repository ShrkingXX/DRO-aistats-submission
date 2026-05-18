@echo off
:: =============================================================================
::  run_larger_scale.bat  —  Windows equivalent of run_larger_scale.sh
::  max_iterations : 100    num_models : 10    initial_points : 5
::  num_trials     : 20 for policy/dkl/ard,  10 for ablation_rtg
::  save_dir       : res_larger_scale\<experiment>
::
::  Run verify_env.bat first to confirm environment is ready.
:: =============================================================================

setlocal enabledelayedexpansion
set PYTHON=.venv\Scripts\python.exe

:: Abort on any error
set ERRORLEVEL=0

echo.
echo =============================================================
echo   Larger-Scale Experiment Suite
echo =============================================================
echo.

:: ── 1. GP-BO Baseline (all 3 benchmarks in one call) ─────────────────────────
echo [1/8] GP-BO Baseline...
%PYTHON% gp_bo_baseline.py ^
    --max_iterations 100 --initial_points 5 ^
    --num_models 10 --num_trials 20 ^
    --save_dir res_larger_scale\gp_bo_baseline
if %errorlevel% neq 0 goto :fail

:: ── 2. Policy Comparison: Rosenbrock ─────────────────────────────────────────
echo [2/8] Policy Comparison - Rosenbrock...
%PYTHON% compare_policy_quality.py ^
    --objective Rosenbrock --max_iterations 100 --initial_points 5 ^
    --num_models 10 --num_trials 20 ^
    --num_rollouts 30 --max_rollout_length 4 ^
    --save_dir res_larger_scale\policy_comparison_rosenbrock
if %errorlevel% neq 0 goto :fail

:: ── 3. Policy Comparison: Ackley ─────────────────────────────────────────────
echo [3/8] Policy Comparison - Ackley...
%PYTHON% compare_policy_quality.py ^
    --objective Ackley --max_iterations 100 --initial_points 5 ^
    --num_models 10 --num_trials 20 ^
    --num_rollouts 30 --max_rollout_length 4 ^
    --save_dir res_larger_scale\policy_comparison_ackley
if %errorlevel% neq 0 goto :fail

:: ── 4. Policy Comparison: Levy ───────────────────────────────────────────────
echo [4/8] Policy Comparison - Levy...
%PYTHON% compare_policy_quality.py ^
    --objective Levy --max_iterations 100 --initial_points 5 ^
    --num_models 10 --num_trials 20 ^
    --num_rollouts 30 --max_rollout_length 4 ^
    --save_dir res_larger_scale\policy_comparison_levy
if %errorlevel% neq 0 goto :fail

:: ── 5. DKL ───────────────────────────────────────────────────────────────────
echo [5/8] DKL - Rosenbrock...
%PYTHON% compare_policy_quality.py ^
    --objective Rosenbrock --max_iterations 100 --initial_points 5 ^
    --num_models 10 --num_trials 20 ^
    --num_rollouts 30 --max_rollout_length 4 ^
    --kernel_type deep_kernel ^
    --save_dir res_larger_scale\dkl_rosenbrock
if %errorlevel% neq 0 goto :fail

echo [5/8] DKL - Ackley...
%PYTHON% compare_policy_quality.py ^
    --objective Ackley --max_iterations 100 --initial_points 5 ^
    --num_models 10 --num_trials 20 ^
    --num_rollouts 30 --max_rollout_length 4 ^
    --kernel_type deep_kernel ^
    --save_dir res_larger_scale\dkl_ackley
if %errorlevel% neq 0 goto :fail

echo [5/8] DKL - Levy...
%PYTHON% compare_policy_quality.py ^
    --objective Levy --max_iterations 100 --initial_points 5 ^
    --num_models 10 --num_trials 20 ^
    --num_rollouts 30 --max_rollout_length 4 ^
    --kernel_type deep_kernel ^
    --save_dir res_larger_scale\dkl_levy
if %errorlevel% neq 0 goto :fail

:: ── 6. ARD Kernel Ablation ───────────────────────────────────────────────────
echo [6/8] ARD Kernel Ablation - Rosenbrock...
%PYTHON% compare_policy_quality.py ^
    --objective Rosenbrock --max_iterations 100 --initial_points 5 ^
    --num_models 10 --num_trials 20 ^
    --num_rollouts 30 --max_rollout_length 4 ^
    --kernel_type rbf_ard ^
    --save_dir res_larger_scale\ablation_ard_rosenbrock
if %errorlevel% neq 0 goto :fail

:: ── 7. RTG Ablation ──────────────────────────────────────────────────────────
echo [7/8] RTG Ablation - Rosenbrock dynamic...
%PYTHON% compare_policy_quality.py ^
    --objective Rosenbrock --max_iterations 100 --initial_points 5 ^
    --num_models 10 --num_trials 10 ^
    --num_rollouts 30 --max_rollout_length 4 ^
    --save_dir res_larger_scale\ablation_rtg_rosenbrock\dynamic
if %errorlevel% neq 0 goto :fail

echo [7/8] RTG Ablation - Rosenbrock fixed=1.0...
%PYTHON% compare_policy_quality.py ^
    --objective Rosenbrock --max_iterations 100 --initial_points 5 ^
    --num_models 10 --num_trials 10 ^
    --num_rollouts 30 --max_rollout_length 4 ^
    --target_rtg 1.0 ^
    --save_dir res_larger_scale\ablation_rtg_rosenbrock\fixed_1.0
if %errorlevel% neq 0 goto :fail

echo [7/8] RTG Ablation - Ackley dynamic...
%PYTHON% compare_policy_quality.py ^
    --objective Ackley --max_iterations 100 --initial_points 5 ^
    --num_models 10 --num_trials 10 ^
    --num_rollouts 30 --max_rollout_length 4 ^
    --save_dir res_larger_scale\ablation_rtg_ackley\dynamic
if %errorlevel% neq 0 goto :fail

echo [7/8] RTG Ablation - Ackley fixed=1.0...
%PYTHON% compare_policy_quality.py ^
    --objective Ackley --max_iterations 100 --initial_points 5 ^
    --num_models 10 --num_trials 10 ^
    --num_rollouts 30 --max_rollout_length 4 ^
    --target_rtg 1.0 ^
    --save_dir res_larger_scale\ablation_rtg_ackley\fixed_1.0
if %errorlevel% neq 0 goto :fail

echo [7/8] RTG Ablation - Levy dynamic...
%PYTHON% compare_policy_quality.py ^
    --objective Levy --max_iterations 100 --initial_points 5 ^
    --num_models 10 --num_trials 10 ^
    --num_rollouts 30 --max_rollout_length 4 ^
    --save_dir res_larger_scale\ablation_rtg_levy\dynamic
if %errorlevel% neq 0 goto :fail

echo [8/8] RTG Ablation - Levy fixed=1.0...
%PYTHON% compare_policy_quality.py ^
    --objective Levy --max_iterations 100 --initial_points 5 ^
    --num_models 10 --num_trials 10 ^
    --num_rollouts 30 --max_rollout_length 4 ^
    --target_rtg 1.0 ^
    --save_dir res_larger_scale\ablation_rtg_levy\fixed_1.0
if %errorlevel% neq 0 goto :fail

echo.
echo =============================================================
echo   All experiments complete. Results in res_larger_scale\
echo =============================================================
goto :end

:fail
echo.
echo =============================================================
echo   EXPERIMENT FAILED. Check the error above.
echo =============================================================
exit /b 1

:end
endlocal
