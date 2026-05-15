import torch
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import sys # For checking module imports in __main__
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import logging # Use logging instead of just print for better Hydra integration
import math # For math.isclose if needed, and used by new_benchmarks

from src.objectives import Ackley, Rosenbrock, Levy # Use correct path if needed
# Attempt to import BayesmarkBoTorchWrapper, assuming it's in src.objectives.bayesmark
try:
    from src.objectives.bayesmark_objectives import BayesmarkBoTorchWrapper
    BAYESMARK_WRAPPER_AVAILABLE = True
except ImportError:
    BAYESMARK_WRAPPER_AVAILABLE = False
    # BayesmarkBoTorchWrapper will be None, handled in main

# --- Import New Benchmarks ---
try:
    # Assuming new_benchmarks.py is in src.objectives
    # If in the same directory as main.py, change to: from new_benchmarks import ...
    from src.objectives.yahoo_gym import (
         iaml_xgboostSF 
    )
    from src.objectives.openaigym import LunarLanderProblem
    NEW_BENCHMARKS_AVAILABLE = True
    # Map names to classes for easier instantiation
    NEW_BENCHMARK_CLASSES = {
        "XGBoost": iaml_xgboostSF,
        "LunarLander": LunarLanderProblem,
    }
except ImportError as e:
    print(f"Warning: Could not import new benchmarks from src.objectives.new_benchmarks: {e}")
    NEW_BENCHMARKS_AVAILABLE = False
    NEW_BENCHMARK_CLASSES = {}


# Import necessary optimizer classes
from src.policy import DirectRegretOptimization
from src.policy import StandardBayesianOptimization, TrustRegionBayesianOptimization, PFNS4BayesianOptimization
from src.policy import SelfCorrectingBayesianOptimization

DEFAULT_DTYPE = torch.float64
torch.set_default_dtype(DEFAULT_DTYPE)

# Configure logging
log = logging.getLogger(__name__)

@hydra.main(config_path='config', config_name="experiment", version_base=None)
def main(cfg: DictConfig):
    # --- Configuration & Logging ---
    log.info("--- Experiment Configuration ---")
    log.info(OmegaConf.to_yaml(cfg))
    log.info("-----------------------------")
    hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    log.info(f"Hydra working directory : {os.getcwd()}")
    log.info(f"Hydra output directory  : {hydra_output_dir}")
    log.info(f"Using dtype             : {DEFAULT_DTYPE}")

    # --- Objective Function Setup ---
    objective_name = cfg.test_function.name
    noise_std_config = cfg.test_function.get('noise_std', 0.0)
    negate_cfg = bool(cfg.test_function.get('negate', True)) # Default negate from config

    # Parameters for original synthetic functions (Ackley, etc.)
    input_dim_cfg = cfg.test_function.get('input_dim', None) # Configured input_dim for some functions
    domain_min_cfg = cfg.test_function.get('domain_min', None)
    domain_max_cfg = cfg.test_function.get('domain_max', None)
    shift_cfg = cfg.test_function.get('shift', None)

    objective_function = None
    bounds_list = None      # Will be List[Tuple[float,float]]
    domain_min = None       # List[float]
    domain_max = None       # List[float]
    input_dim = None        # int, will be derived from the objective_function
    negate = negate_cfg     # bool, might be updated by objective_function

    log.info(f"Attempting to set up Objective: {objective_name}")

    if objective_name.lower() == "bayesmark":
        if not BAYESMARK_WRAPPER_AVAILABLE:
            log.error("Bayesmark objective specified, but BayesmarkBoTorchWrapper could not be imported.")
            raise ImportError("BayesmarkBoTorchWrapper not found. Please ensure it's in src/objectives/bayesmark.py")

        bm_cfg = cfg.test_function.get('bayesmark_config', None)
        if bm_cfg is None:
            log.error("Objective is 'Bayesmark', but 'test_function.bayesmark_config' is missing.")
            raise ValueError("Missing 'bayesmark_config' for Bayesmark objective.")

        objective_function = BayesmarkBoTorchWrapper(
            model_name=bm_cfg.model,
            dataset=bm_cfg.dataset,
            scorer=bm_cfg.scorer,
            bayesmark_path=bm_cfg.get('path', None),
            noise_std=noise_std_config if noise_std_config > 0 else None
        )
        input_dim = objective_function.dim
        bounds_list = objective_function._bounds
        negate = objective_function.negate # Use negate from wrapper
        log.info(f"Bayesmark wrapper: dim={input_dim}, bounds={bounds_list}, negate={negate}, noise_std={getattr(objective_function, 'noise_std', 'N/A')}")

    elif objective_name in {"Ackley", "Rosenbrock", "Levy"}:
        if input_dim_cfg is None or domain_min_cfg is None or domain_max_cfg is None:
            log.error(f"For '{objective_name}', input_dim, domain_min, domain_max must be specified in config.")
            raise ValueError(f"Missing required parameters for {objective_name}")
        input_dim = input_dim_cfg # Use configured dim for these

        if isinstance(domain_min_cfg, (int, float)):
            domain_min_list = [float(domain_min_cfg)] * input_dim
            domain_max_list = [float(domain_max_cfg)] * input_dim
        else:
            domain_min_list = [float(b) for b in domain_min_cfg]
            domain_max_list = [float(b) for b in domain_max_cfg]
        bounds_list = list(zip(domain_min_list, domain_max_list))
        negate = negate_cfg # Use negate from main config

        log.info(f"Synthetic Objective: {objective_name} (dim={input_dim}, negate={negate}, noise_cfg={noise_std_config})")
        bounds_tensor = torch.tensor(bounds_list, dtype=DEFAULT_DTYPE)

        common_kwargs = {
            "dim": input_dim, "bounds": bounds_tensor, "negate": negate,
            "noise_std": noise_std_config if noise_std_config > 0 else None,
            "shift": shift_cfg
        }
        if objective_name == "Ackley":
            objective_function = Ackley(**common_kwargs)
        elif objective_name == "Rosenbrock":
            objective_function = Rosenbrock(**common_kwargs)
        elif objective_name == "Levy":
            objective_function = Levy(**common_kwargs)
        
        # Confirm actual properties from the instantiated object
        if objective_function:
            input_dim = objective_function.dim
            negate = objective_function.negate
            # Convert BoTorch bounds (list of tensors) to list of tuples if needed
            if hasattr(objective_function, "_bounds") and isinstance(objective_function._bounds, list) and \
               len(objective_function._bounds) > 0 and isinstance(objective_function._bounds[0], torch.Tensor):
                 bounds_list = [(b[0].item(), b[1].item()) for b in objective_function._bounds]
            elif hasattr(objective_function, "bounds") and isinstance(objective_function.bounds, torch.Tensor): 
                if len(objective_function.bounds) == objective_function.dim:
                    bounds_list = [(objective_function.bounds[0,i].item(), objective_function.bounds[1,i].item()) for i in range(objective_function.dim)]
                else:
                    bounds_list = [(objective_function.bounds[0,0].item(), objective_function.bounds[1,0].item()) for i in range(objective_function.dim)]


    elif NEW_BENCHMARKS_AVAILABLE and objective_name in NEW_BENCHMARK_CLASSES:
        log.info(f"New Benchmark: {objective_name} (negate_cfg={negate_cfg}, noise_cfg={noise_std_config})")
        BenchmarkClass = NEW_BENCHMARK_CLASSES[objective_name]
        obj_kwargs = {'negate': negate_cfg} # Common argument

        # Add benchmark-specific arguments from cfg.test_function
        if objective_name in ["iaml_xgboostSF"]:
            instance = cfg.test_function.get('instance')
            if instance is None:
                raise ValueError(f"'instance' must be provided for {objective_name}")
            obj_kwargs['instance'] = instance
            if objective_name == "iaml_xgboostSF":
                obj_kwargs['booster'] = cfg.test_function.get('booster', 'dart')
        elif objective_name == "LunarLander":
            obj_kwargs['n_runs'] = cfg.test_function.get('n_runs', 10) # Default from LunarLanderProblem if not in config
            obj_kwargs['steps_limit'] = cfg.test_function.get('steps_limit', 1000)
            obj_kwargs['timeout_reward'] = cfg.test_function.get('timeout_reward', -100.0)
            # problem_seed: if None, LunarLanderProblem uses its internal default.
            # Pass it from config if specified, otherwise let the class handle its default.
            if 'problem_seed' in cfg.test_function: # Only pass if explicitly in config
                 obj_kwargs['problem_seed'] = cfg.test_function.problem_seed
        
        objective_function = BenchmarkClass(**obj_kwargs)

        # Apply BoTorch-level observation noise if configured and applicable
        if hasattr(objective_function, 'noise_std') and noise_std_config > 0:
            objective_function.noise_std = noise_std_config # SyntheticTestFunction base handles this
            log.info(f"Applied BoTorch-level observation noise_std={noise_std_config} to {objective_name}")

        input_dim = objective_function.dim
        bounds_list = objective_function._bounds # Should be List[Tuple[float,float]]
        negate = objective_function.negate # Actual negate status from the object
        log.info(f"{objective_name} initialized: dim={input_dim}, bounds={bounds_list}, negate={negate}, effective_noise_std={getattr(objective_function, 'noise_std', 'N/A')}")

    else:
        err_msg = f"Unsupported objective function: {objective_name}"
        if objective_name in NEW_BENCHMARK_CLASSES and not NEW_BENCHMARKS_AVAILABLE:
            err_msg += ". New benchmarks were found by name but import failed. Check src/objectives/new_benchmarks.py."
        log.error(err_msg)
        raise ValueError(err_msg)

    if objective_function is None:
        log.critical("Objective function was NOT initialized!")
        raise RuntimeError("Objective function setup failed.")

    # Derive domain_min and domain_max from bounds_list for all cases
    if bounds_list:
        domain_min = [b[0] for b in bounds_list]
        domain_max = [b[1] for b in bounds_list]
    else:
        log.error(f"bounds_list was not set for objective {objective_name}")
        raise ValueError("Failed to derive bounds_list.")


    # --- Experiment Parameters ---
    n_trials = cfg.experiment_params.num_trials
    seed_start = cfg.experiment_params.seed_start
    initial_points = cfg.experiment_params.initial_points
    max_iterations = cfg.experiment_params.max_iterations
    method_name = cfg.method.name
    save_dir = cfg.get("save_dir", "./res_main_py_raw_trials")

    if n_trials <= 0:
        log.warning("cfg.experiment_params.num_trials is <= 0. No trials will be run.")
        return None

    # --- File Prefix for Saving ---
    specific_details = ""
    if objective_name.lower() == "bayesmark":
        bm_cfg_for_filename = cfg.test_function.bayesmark_config
        specific_details = f"_{bm_cfg_for_filename.model}_{bm_cfg_for_filename.dataset}_{bm_cfg_for_filename.scorer}"
    elif objective_name in NEW_BENCHMARK_CLASSES:
        if objective_name in ["iaml_xgboostSF"]:
            specific_details = f"_{cfg.test_function.instance}"
            if objective_name == "iaml_xgboostSF":
                specific_details += f"_{cfg.test_function.get('booster', 'dart')}"
    file_prefix = f"{method_name}_{objective_name}{specific_details}_{input_dim}D"


    # --- Trial Loop ---
    all_trials_results = []
    all_trials_all_y = []
    all_trials_best_y = []

    log.info(f"Running {n_trials} trials for method '{method_name}' on '{file_prefix}'...")
    iterator = tqdm(range(n_trials), desc=f"Trials ({file_prefix})", unit="trial")

    for trial_idx in iterator:
        trial_seed = seed_start + trial_idx
        log.info(f"--- Starting Trial {trial_idx + 1}/{n_trials} (Seed: {trial_seed}) ---")
        torch.manual_seed(trial_seed)
        np.random.seed(trial_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(trial_seed)

        current_method_cfg = cfg.method.copy()
        current_method_cfg.bo.domain_min = domain_min
        current_method_cfg.bo.domain_max = domain_max
        current_method_cfg.bo.input_dim = input_dim
        current_method_cfg.bo.initial_points = initial_points
        current_method_cfg.bo.max_iterations = max_iterations
        current_method_cfg.seed = trial_seed

        log.debug(f"Instantiating optimizer: {method_name}")
        optimizer = None
        if method_name.lower() == 'dro':
            optimizer = DirectRegretOptimization(current_method_cfg, objective_function)
        elif method_name.lower() == 'bo':
            optimizer = StandardBayesianOptimization(current_method_cfg, objective_function)
        elif method_name.lower() == 'turbo':
            optimizer = TrustRegionBayesianOptimization(current_method_cfg, objective_function)
        elif method_name.lower() == 'pfns4bo':
            optimizer = PFNS4BayesianOptimization(current_method_cfg, objective_function)
        elif method_name.lower() == 'scorebo':
            optimizer = SelfCorrectingBayesianOptimization(current_method_cfg, objective_function)
        else:
            log.error(f"Unsupported method specified in config: {method_name}")
            raise ValueError(f"Unsupported method specified in config: {method_name}")

        try:
            result = optimizer.run_optimization()
            # Ensure results are numpy arrays for consistent saving
            trial_data_y = result['all_y'].cpu().numpy() if isinstance(result['all_y'], torch.Tensor) else np.array(result['all_y'])
            trial_results = {
                'best_x': result['best_x'].cpu().numpy() if isinstance(result['best_x'], torch.Tensor) else np.array(result['best_x']),
                'best_y': result['best_y'].cpu().numpy() if isinstance(result['best_y'], torch.Tensor) else np.array(result['best_y']),
                'all_x': result['all_x'].cpu().numpy() if isinstance(result['all_x'], torch.Tensor) else np.array(result['all_x']),
                'all_y': trial_data_y,
                'seed': trial_seed
            }
            all_trials_results.append(trial_results)
            all_trials_all_y.append(trial_data_y) # Append processed numpy array
            all_trials_best_y.append(trial_results['best_y'])
            log.info(f"Trial {trial_idx + 1} Complete! Best Y: {trial_results['best_y']:.6f}")

            # RQ2 — save fidelity logs produced by this DRO trial
            if method_name.lower() == 'dro' and hasattr(optimizer, 'save_fidelity_logs'):
                fidelity_prefix = f"{file_prefix}_trial{trial_idx}_seed{trial_seed}"
                optimizer.save_fidelity_logs(save_dir, fidelity_prefix)           # Setup A
                optimizer.save_true_trajectory_logs(save_dir, fidelity_prefix)    # Setup B (if enabled)

            if cfg.experiment_params.get("save_individual_trials", False):
                 trial_save_dir = os.path.join(hydra_output_dir, f"trial_{trial_idx}")
                 os.makedirs(trial_save_dir, exist_ok=True)
                 plt.figure(figsize=(10, 6))
                 plt.plot(trial_results['all_y'], marker='.', linestyle='-')
                 plt.xlabel('Iteration (incl. initial points)')
                 plt.ylabel('Objective Value')
                 plt.title(f'Optimization Progress (Trial {trial_idx + 1}, Seed {trial_seed}, {file_prefix})')
                 plt.grid(True); plt.tight_layout()
                 plt.savefig(os.path.join(trial_save_dir, f"{file_prefix}_trial_{trial_idx}_progress.png"))
                 plt.close()
                 np.savez(os.path.join(trial_save_dir, f"{file_prefix}_trial_{trial_idx}_results.npz"), **trial_results)
        except Exception as e:
            log.error(f"Error during trial {trial_idx + 1} for {file_prefix}: {e}", exc_info=True)
            all_trials_all_y.append(np.array([np.nan] * (initial_points + max_iterations)))
            all_trials_best_y.append(np.nan)

    log.info("\n--- Aggregating Results Across Trials ---")
    if not all_trials_all_y:
        log.warning("No trial results available for aggregation.")
        return np.nan

    expected_len = initial_points + max_iterations
    processed_all_y = []
    valid_trials = 0
    for y_seq in all_trials_all_y:
        if y_seq is not None and hasattr(y_seq, '__len__') and len(y_seq) > 0 and not np.isnan(y_seq).all():
            if len(y_seq) > expected_len: processed_all_y.append(y_seq[:expected_len])
            elif len(y_seq) < expected_len:
                padding_val = y_seq[-1] if len(y_seq) > 0 else np.nan
                processed_all_y.append(np.pad(y_seq, (0, expected_len - len(y_seq)), 'constant', constant_values=padding_val))
            else: processed_all_y.append(y_seq)
            valid_trials += 1
        else: processed_all_y.append(np.full(expected_len, np.nan)) # Ensure consistent shape for failed trials

    if valid_trials == 0:
        log.warning("No valid trial results found after processing.")
        return np.nan

    all_trials_all_y_np = np.array(processed_all_y)
    # Robust calculation of mean, std, stderr, ignoring NaNs
    mean_all_y = np.nanmean(all_trials_all_y_np, axis=0)
    std_all_y = np.nanstd(all_trials_all_y_np, axis=0)
    # Count non-NaNs per iteration for stderr
    counts_per_iteration = np.sum(~np.isnan(all_trials_all_y_np), axis=0)
    # Avoid division by zero if an iteration has no valid data (though unlikely if valid_trials > 0)
    stderr_all_y = np.divide(std_all_y, np.sqrt(counts_per_iteration), 
                             out=np.full_like(std_all_y, np.nan), 
                             where=counts_per_iteration > 0)

    iterations = np.arange(len(mean_all_y))

    plt.figure(figsize=(12, 7))
    plt.plot(iterations, mean_all_y, label=f'Mean Objective Value ({method_name} on {file_prefix})', color='blue')
    plt.fill_between(iterations, mean_all_y - stderr_all_y, mean_all_y + stderr_all_y, color='blue', alpha=0.2, label='Mean +/- 1 Std Error')
    plt.xlabel('Iteration (incl. initial points)'); plt.ylabel('Objective Value')
    plt.title(f'Aggregate Optimization Progress ({file_prefix}, {valid_trials}/{n_trials} Trials)')
    plt.legend(); plt.grid(True); plt.tight_layout()
    aggregate_plot_path = os.path.join(hydra_output_dir, f"aggregate_progress_{file_prefix}.png")
    plt.savefig(aggregate_plot_path)
    log.info(f"Saved aggregate plot to: {aggregate_plot_path}")
    plt.close()

    aggregate_save_path = os.path.join(hydra_output_dir, f"aggregate_results_{file_prefix}.npz")
    valid_best_y = np.array([y for y in all_trials_best_y if y is not None and not np.isnan(y)])
    mean_best_y = np.mean(valid_best_y) if len(valid_best_y) > 0 else np.nan
    std_best_y = np.std(valid_best_y) if len(valid_best_y) > 0 else np.nan

    np.savez(
        aggregate_save_path,
        mean_all_y=mean_all_y, std_all_y=std_all_y, stderr_all_y=stderr_all_y,
        all_trials_best_y=np.array(all_trials_best_y),
        mean_final_best_y=mean_best_y, std_final_best_y=std_best_y,
        valid_trials_count = valid_trials, iterations=iterations,
        config=OmegaConf.to_container(cfg, resolve=True),
        method_name=method_name, objective_name_full=file_prefix, input_dim=input_dim
    )
    log.info(f"Saved aggregate results to: {aggregate_save_path}")

    log.info(f"\n--- Saving Raw Trial Data to {save_dir} ---")
    if all_trials_results:
        try:
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_results_filename = f"{file_prefix}_{timestamp}_all_trials.pkl"
            raw_results_filepath = os.path.join(save_dir, raw_results_filename)
            with open(raw_results_filepath, 'wb') as f: pickle.dump(all_trials_results, f)
            log.info(f"Saved raw results for all trials to: {raw_results_filepath}")
        except Exception as e: log.error(f"Failed to save raw trial results to {save_dir}: {e}", exc_info=True)
    else: log.warning("No trial results were generated, skipping raw data saving.")

    log.info("\n--- Overall Summary for this Run ---")
    log.info(f"Method: {method_name}, Objective Details: {file_prefix}")
    log.info(f"Ran {valid_trials}/{n_trials} successful trials.")
    log.info(f"Mean Best Objective Value across trials: {mean_best_y:.6f}")
    log.info(f"Std Dev Best Objective Value across trials: {std_best_y:.6f}")
    log.info(f"Results saved in: {hydra_output_dir}")
    log.info(f"Raw trial data (if any) attempted save in: {save_dir}")
    log.info("----------------------------------")

    return float(mean_best_y)

if __name__ == "__main__":
    main()