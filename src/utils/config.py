import hydra
import os
from omegaconf import DictConfig, OmegaConf, OmegaConf

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))
# Import the objective function
from src.policy import StandardBOConfig 

def create_dro_config(base_hydra_cfg: DictConfig, method_params: dict, common_params: dict) -> DictConfig:
    """Creates the Hydra config for DRO, applying necessary overrides."""
    # Resolve interpolations first IF base_cfg might contain them
    # trial_cfg_dict = OmegaConf.to_container(base_hydra_cfg, resolve=True)
    # trial_cfg = OmegaConf.create(trial_cfg_dict)
    # More direct copy if no complex interpolations expected at this stage:
    trial_cfg = base_hydra_cfg.copy()

    # Apply common overrides
    OmegaConf.update(trial_cfg, "bo.max_iterations", common_params['max_iterations'], merge=True)
    OmegaConf.update(trial_cfg, "bo.input_dim", common_params['input_dim'], merge=True)
    OmegaConf.update(trial_cfg, "bo.domain_min", common_params['domain_min_list'], merge=True)
    OmegaConf.update(trial_cfg, "bo.domain_max", common_params['domain_max_list'], merge=True)
    OmegaConf.update(trial_cfg, "seed", common_params['seed'], merge=True)
    # Do not override save_dir from common_params here, method might save artifacts elsewhere
    OmegaConf.update(trial_cfg, "verbose", common_params['verbose'], merge=True)
    OmegaConf.update(trial_cfg, "bo.initial_points", common_params['initial_points'], merge=True)

    # Apply DRO specific overrides from method_params
    # Example: trial_cfg.gp.num_models = method_params.get('gp_num_models', trial_cfg.gp.num_models) # Allow override
    OmegaConf.update(trial_cfg, "gp.num_models", method_params.get('gp_num_models', trial_cfg.gp.num_models), merge=True)
    OmegaConf.update(trial_cfg, "acquisition.function", method_params.get('acquisition', trial_cfg.acquisition.function), merge=True)
    OmegaConf.update(trial_cfg, "transformer.num_epochs", method_params.get('transformer_epochs', trial_cfg.transformer.num_epochs), merge=True)
    OmegaConf.update(trial_cfg, "simulation.num_rollouts", method_params.get('num_rollouts', trial_cfg.simulation.num_rollouts), merge=True)

    # Important: Set a specific save_dir for DRO's internal savings if needed
    # This should ideally point within the main experiment output directory
    method_artifact_dir = os.path.join(common_params['experiment_output_dir'], "dro_artifacts", f"trial_{common_params['trial_index']}")
    os.makedirs(method_artifact_dir, exist_ok=True)
    OmegaConf.update(trial_cfg, "save_dir", method_artifact_dir, merge=True)

    return trial_cfg


def create_standard_bo_config(base_hydra_cfg: DictConfig, method_params: dict, common_params: dict) -> StandardBOConfig:
    """Creates the StandardBOConfig dataclass."""
    initial_points = common_params['initial_points']

    # Standard BO might save plots/data internally, decide where they should go
    # Example: save within the main experiment output dir
    method_artifact_dir = os.path.join(common_params['experiment_output_dir'], "standard_bo_artifacts", f"trial_{common_params['trial_index']}")
    os.makedirs(method_artifact_dir, exist_ok=True)

    return StandardBOConfig(
        max_iterations=common_params['max_iterations'],
        input_dim=common_params['input_dim'],
        domain_min=common_params['domain_min_list'][0], # Assumes StandardBO takes single floats
        domain_max=common_params['domain_max_list'][0], # Assumes StandardBO takes single floats
        initial_points=initial_points,
        objective=method_params.get('objective', "maximize"),
        acquisition=method_params.get('acquisition', "ei"),
        seed=common_params['seed'],
        verbose=common_params['verbose'],
        save_dir=method_artifact_dir # Point its savings to the specific dir
    )