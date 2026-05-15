"""
Example usage of DirectRegretOptimization
"""
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
import os

# Import the main implementation
from src.policy import DirectRegretOptimization
from src.objectives import ackley_function

@hydra.main(config_path="config/methods", config_name="dro_config")
def main(cfg: DictConfig):
    # Print the configuration
    print(OmegaConf.to_yaml(cfg))
    
    # Create the output directory if it doesn't exist
    os.makedirs(cfg.save_dir, exist_ok=True)
    
    # Select the test function
    test_function = ackley_function
    
    # Set the function name for the output
    function_name = "Ackley"
    
    # Ensure domain bounds are correct for the input dimension
    if isinstance(cfg.bo.domain_min, float):
        cfg.bo.domain_min = [cfg.bo.domain_min] * cfg.bo.input_dim
    if isinstance(cfg.bo.domain_max, float):
        cfg.bo.domain_max = [cfg.bo.domain_max] * cfg.bo.input_dim
    
    if len(cfg.bo.domain_min) != cfg.bo.input_dim:
        cfg.bo.domain_min = [cfg.bo.domain_min[0]] * cfg.bo.input_dim
    if len(cfg.bo.domain_max) != cfg.bo.input_dim:
        cfg.bo.domain_max = [cfg.bo.domain_max[0]] * cfg.bo.input_dim
    
    print(f"Optimizing {function_name} function using Direct Regret Optimization")
    
    # Initialize the optimizer
    dro = DirectRegretOptimization(cfg, test_function)
    
    # Run optimization
    result = dro.run_optimization()
    
    # Print results
    print("\nOptimization Results:")
    print(f"Best point: {result['best_x']}")
    print(f"Best value: {result['best_y']}")
    print(f"Total evaluations: {len(result['all_y'])}")
    
    # Create a progress plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(result['all_y'])), result['all_y'], 'o-', markersize=6)
    plt.axhline(y=result['best_y'], color='r', linestyle='--', label=f'Best value: {result["best_y"]:.4f}')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.title(f'Optimization Progress for {function_name} Function')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{cfg.save_dir}/{function_name.lower()}_progress.png")
    
    print(f"Results saved to {cfg.save_dir}/")
    return result

if __name__ == "__main__":
    main()