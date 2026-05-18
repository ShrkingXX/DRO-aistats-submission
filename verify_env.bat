@echo off
:: =============================================================================
::  verify_env.bat
::  Run this before any experiment to confirm the environment is correct.
::  Usage: verify_env.bat
:: =============================================================================

echo.
echo ============================================================
echo   Environment Verification
echo ============================================================
echo.

:: ── 1. Check Python is reachable ─────────────────────────────────────────────
echo [1] Checking Python...
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo     FAIL: python not found in PATH.
    echo     Install Python 3.9+ from https://www.python.org/downloads/
    goto :fail
)
python --version
echo     OK
echo.

:: ── 2. Create virtual environment if it does not exist ───────────────────────
echo [2] Checking virtual environment...
if not exist ".venv\Scripts\python.exe" (
    echo     .venv not found. Creating...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo     FAIL: could not create .venv
        goto :fail
    )
    echo     Created .venv
) else (
    echo     .venv already exists  OK
)
echo.

:: ── 3. Install / verify requirements ─────────────────────────────────────────
echo [3] Installing requirements (skips already-installed packages)...
.venv\Scripts\pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo     FAIL: pip install failed. Check requirements.txt and network access.
    goto :fail
)
echo     OK
echo.

:: ── 4. Verify core package versions ─────────────────────────────────────────
echo [4] Verifying core package versions...
.venv\Scripts\python -c "import torch; print('    torch     :', torch.__version__)"
.venv\Scripts\python -c "import gpytorch; print('    gpytorch  :', gpytorch.__version__)"
.venv\Scripts\python -c "import botorch; print('    botorch   :', botorch.__version__)"
.venv\Scripts\python -c "import numpy as np; print('    numpy     :', np.__version__)"
.venv\Scripts\python -c "import scipy; print('    scipy     :', scipy.__version__)"
.venv\Scripts\python -c "import omegaconf; print('    omegaconf :', omegaconf.__version__)"
echo.

:: ── 5. Smoke-test: import the project source ─────────────────────────────────
echo [5] Smoke-test: importing project modules...
.venv\Scripts\python -c ^
    "import sys; sys.path.insert(0, '.'); ^
     from src.policy.dro import DirectRegretOptimization; ^
     from src.objectives import Ackley, Rosenbrock, Levy; ^
     from src.model.deep_kernel_gp import DeepKernelGP; ^
     print('    src.policy.dro            OK'); ^
     print('    src.objectives            OK'); ^
     print('    src.model.deep_kernel_gp  OK')"
if %errorlevel% neq 0 (
    echo     FAIL: project import failed. Check that you are in the repo root directory.
    goto :fail
)
echo.

:: ── 6. Quick config load test ─────────────────────────────────────────────────
echo [6] Loading DRO config...
.venv\Scripts\python -c ^
    "from omegaconf import OmegaConf; ^
     cfg = OmegaConf.load('config/method/dro.yaml'); ^
     print('    config/method/dro.yaml    OK')"
if %errorlevel% neq 0 (
    echo     FAIL: could not load config/method/dro.yaml
    goto :fail
)
echo.

:: ── 7. Create output directory ────────────────────────────────────────────────
echo [7] Creating output directory res_larger_scale\...
if not exist "res_larger_scale" mkdir res_larger_scale
echo     OK
echo.

echo ============================================================
echo   All checks passed. Ready to run experiments.
echo   Next step: run_larger_scale.bat
echo ============================================================
echo.
goto :end

:fail
echo.
echo ============================================================
echo   Verification FAILED. Fix the error above before running.
echo ============================================================
exit /b 1

:end
