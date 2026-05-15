from __future__ import annotations

import math
from typing import Optional, List, Tuple

import torch
from botorch.test_functions.synthetic import SyntheticTestFunction
# from botorch.utils.transforms import unnormalize # Not needed for SF definition directly
from torch import Tensor
import numpy as np

class AugmentedBranin(SyntheticTestFunction):
    r"""Augmented Branin test function for multi-fidelity optimization."""
    dim = 3
    _bounds = [(-5.0, 10.0), (0.0, 15.0), (0.0, 1.0)]
    _optimal_value = 0.397887
    _optimizers = [
        (-math.pi, 12.275, 1.0), # Target fidelity optimizer
        (math.pi, 2.275, 1.0),   # Another target fidelity optimizer (approx, from Branin)
        (3*math.pi, 2.275, 1.0), # Another target fidelity optimizer (approx, from Branin)
        # Original optimizers might be for various fidelities, let's focus on target
    ]
    def __init__(self, negate: Optional[bool] = False) -> None:
        super().__init__(negate=negate)
        # Ensure optimizers are correctly defined for the non-negated case
        if negate:
            self._optimal_value = -self._optimal_value

    def evaluate_true(self, X: Tensor) -> Tensor:
        # Ensure X has 3 columns
        if X.ndim == 1: X = X.unsqueeze(0)
        # assert X.shape[1] == self.dim, f"Input X should have {self.dim} columns"
        t1 = (
            X[..., 1]
            - (5.1 / (4 * math.pi**2) - 0.1 * (1 - X[..., 2])) * X[..., 0].pow(2)
            + 5 / math.pi * X[..., 0]
            - 6
        )
        t2 = 10 * (1 - 1 / (8 * math.pi)) * torch.cos(X[..., 0])
        return t1.pow(2) + t2 + 10

class AugmentedRastrigin(SyntheticTestFunction):
    r"""Augmented Rastrigin test function for multi-fidelity optimization."""
    dim = 2
    _bounds = [(-10.0, 10.0), (0.0, 1.0)] # Last is fidelity
    _optimal_value = 0.0 # Max value is 0 at x=0, fid=1
    _optimizers = [(0.0, 1.0)]
    _scale = 1.0

    def __init__(self, negate: Optional[bool] = False) -> None:
        super().__init__(negate=negate)
        if negate: # If negated, we are minimizing, optimal value becomes -0
            self._optimal_value = -self._optimal_value
            # Optimizers remain the same point, but value changes
        # else: maximizing, original optimal value is correct (for x=0, fid=1, base=0, correction=0)

    def evaluate_true(self, X: Tensor) -> Tensor:
        if X.ndim == 1: X = X.unsqueeze(0)
        # assert X.shape[1] == self.dim, f"Input X should have {self.dim} columns"
        single_fid_x = X[..., :-1].squeeze(-1) # Handle single and batch inputs
        base = (single_fid_x**2 - 10 * torch.cos(2 * math.pi * single_fid_x) + 10)
        fid = X[..., -1]
        # Simplified fidelity correction for clarity, original had complex conditions
        # Correction should be zero at fid=1 for _optimal_value=0 to hold
        is_low_fidelity = (1 - fid) > 1e-10
        # Example: fid_correction_mu = is_low_fidelity * (torch.abs(single_fid_x) * (1-fid) * 5)
        # To ensure optimal_value = 0 at fid=1, let correction terms be zero at fid=1
        fid_correction_mu_val = 2 * ((fid) <= 0.5) * ((-.1) ** ((fid) <= 0.25))
        fid_correction_mu_val = fid_correction_mu_val * (torch.abs(single_fid_x) **((1-fid)/0.25 + 1e-6)) # added small epsilon
        
        # Ensure fidelity_correction is zero when fid is 1.0
        fidelity_correction = torch.normal(mean=fid_correction_mu_val, std=8.0) * is_low_fidelity
        return (- base - fidelity_correction) * self._scale

class AugmentedRastrigin20D(SyntheticTestFunction):
    dim = 21 # 20 features + 1 fidelity
    _bounds = [(-5.0, 5.0) for _ in range(dim-1)]
    _bounds.append((0.0, 1.0)) # Fidelity bound
    _optimal_value = 0.0 # Max value at x=all_zeros, fid=1
    _optimizers = [(0.0,) * (dim -1) + (1.0,)] # (0,0,...,0, 1.0)
    _scale = 1.0

    def __init__(self, negate: Optional[bool] = False) -> None:
        super().__init__(negate=negate)
        if negate:
            self._optimal_value = -self._optimal_value

    def evaluate_true(self, X: Tensor) -> Tensor:
        if X.ndim == 1: X = X.unsqueeze(0)
        # assert X.shape[1] == self.dim, f"Input X should have {self.dim} columns"
        
        single_fid_x = X[..., :-1] # Shape (batch_size, 20) or (20,)
        
        # Ensure term1 calculation is robust to batch vs single instance
        if single_fid_x.ndim == 1: # Single instance (20,)
            term1 = torch.sum(single_fid_x**2 - 10 * torch.cos(2 * math.pi * single_fid_x), dim=-1) + 10 * single_fid_x.shape[0]
        else: # Batch of instances (batch_size, 20)
            term1 = torch.sum(single_fid_x**2 - 10 * torch.cos(2 * math.pi * single_fid_x), dim=-1) + 10 * single_fid_x.shape[1]

        base = term1 # This is sum version of Rastrigin, not norm like original error
                      # Original description used x**2 - 10cos(...) not norm(x)**2
                      # Let's stick to the sum version like standard Rastrigin

        fid = X[..., -1]
        is_low_fidelity = (1 - fid) > 1e-10

        # Simplified fidelity correction, ensure it's zero at fid=1
        # Example from original:
        fid_correction_mu_val = ((1 - fid) > 1e-10) * (2 ** ((fid) <= 0.5)) * ((-.1) ** ((fid) <= 0.25))
        # The norm part might make values very large for 20D, let's use sum of abs values
        sum_abs_x = torch.sum(torch.abs(single_fid_x), dim=-1) if single_fid_x.ndim > 1 else torch.sum(torch.abs(single_fid_x))
        fid_correction_mu_val = fid_correction_mu_val * (sum_abs_x **((1-fid)/0.25 + 1e-6))
        
        fidelity_correction = torch.normal(mean=fid_correction_mu_val, std=8.0) * is_low_fidelity
        return (- base - fidelity_correction) * self._scale


# --- YAHPO Gym Base Class ---
class YahooGYM(SyntheticTestFunction):
    def __init__(self, negate: Optional[bool]=False):
        super().__init__(negate=negate)
        # yahpo_gym related initializations would go into specific subclasses
        self.bench = None # Placeholder
        self.instance_id = None # Placeholder
        self.input_keys: List[str] = []
        self.int_keys: List[str] = []
        self.fidelity_key: Optional[str] = None # To be defined by subclasses
        self.target_fidelity_value: Optional[float] = None # To be defined by subclasses

    def _setup_yahpo(self, scenario: str, instance: str, data_path: str = "./data/yahpo_data"):
        from yahpo_gym import local_config, benchmark_set
        local_config.init_config() # Initialize if not already
        try: # check if data path is already set
            local_config.get_data_path()
        except: # if not, set it
             local_config.set_data_path(data_path)
        self.bench = benchmark_set.BenchmarkSet(scenario)
        if instance not in self.bench.instances:
            raise ValueError(f"Instance '{instance}' not found for scenario '{scenario}'. Available: {self.bench.instances}")
        self.bench.set_instance(instance)
        self.instance_id = instance


class LCBench(YahooGYM):
    dim = 8 # 7 features + 1 fidelity (epoch)
    _bounds = [ # From original code
        (16, 512), (0.0001001, 0.1), (0.0, 1.0), (64, 1024),
        (0.1, 0.99), (1, 5), (0.0000101, 0.1), (1, 52) # epoch (fidelity)
    ]
    # _optimal_value: Depends on instance and whether we negate. YAHPO provides values.
    # For 'test_balanced_accuracy' (higher is better), original _optimal_value would be max.

    def __init__(self, negate: Optional[bool] = False, instance: str = '3945') -> None:
        super().__init__(negate=negate)
        self._setup_yahpo("lcbench", instance)
        self.input_keys = ['batch_size', 'learning_rate', 'max_dropout', 'max_units', 
                           'momentum', 'num_layers', 'weight_decay', 'epoch']
        self.int_keys = ['batch_size', 'max_units', 'num_layers', 'epoch']
        self.fidelity_key = 'epoch'
        self.target_fidelity_value = 52.0 # Max epochs

        # Note: _optimal_value for YAHPO is tricky as it's per instance and not easily available
        # without querying the benchmark. For now, we won't set it generically.

    def evaluate_true(self, X: Tensor) -> Tensor:
        if X.ndim == 1: X = X.unsqueeze(0)
        assert X.shape[1] == self.dim, f"Input X should have {self.dim} columns"
        
        configs = []
        for i in range(X.shape[0]):
            x_numpy = X[i].cpu().numpy()
            config = {k: v.item() if hasattr(v, 'item') else v for k, v in zip(self.input_keys, x_numpy)}
            for k_int in self.int_keys:
                if k_int in config:
                    config[k_int] = int(round(config[k_int])) # Ensure integer types
            config['OpenML_task_id'] = self.instance_id # Correct key for lcbench
            configs.append(config)

        outputs = self.bench.objective_function(configs)
        objectives = [inst['test_balanced_accuracy'] for inst in outputs]
        return torch.tensor(objectives, dtype=X.dtype, device=X.device)


class iaml_rpart(YahooGYM):
    dim = 5 # 4 features + 1 fidelity (trainsize)
    _bounds = [(0.001, 1.0), (1, 30), (1, 100), (1, 100), (0.03, 1.0)] # trainsize (fidelity)

    def __init__(self, negate: Optional[bool] = False, instance: str = '1489') -> None:
        super().__init__(negate=negate)
        self._setup_yahpo("iaml_rpart", instance)
        self.input_keys = ['cp', 'maxdepth', 'minbucket', 'minsplit', 'trainsize']
        self.int_keys = ['maxdepth', 'minbucket', 'minsplit']
        self.fidelity_key = 'trainsize'
        self.target_fidelity_value = 1.0

    def evaluate_true(self, X: Tensor) -> Tensor:
        if X.ndim == 1: X = X.unsqueeze(0)
        assert X.shape[1] == self.dim
        configs = []
        for i in range(X.shape[0]):
            x_numpy = X[i].cpu().numpy()
            config = {k: v.item() if hasattr(v, 'item') else v for k, v in zip(self.input_keys, x_numpy)}
            for k_int in self.int_keys:
                 if k_int in config:
                    config[k_int] = int(round(config[k_int]))
            config['task_id'] = self.instance_id # Correct key
            configs.append(config)
        outputs = self.bench.objective_function(configs)
        objectives = [inst['auc'] for inst in outputs] # Objective is auc
        return torch.tensor(objectives, dtype=X.dtype, device=X.device)


class iaml_xgboost(YahooGYM):
    dim = 13 # 12 features + 1 fidelity (trainsize)
    _bounds = [
        (0.0001001, 999.9999), (0.01001, 1.0), (0.01001, 1.0), (0.0001001, 1.0),
        (0.0001001, 6.9999), (0.0001001, 999.9999), (1.0, 15.0), (2.7183, 149.9999),
        (3.0, 2000.0), (0.0, 1.0), (0.0, 1.0), (0.1001, 1.0), (0.03, 1.0) # trainsize (fidelity)
    ]
    def __init__(self, negate: Optional[bool] = False, instance: str = '1489', booster:str = 'dart') -> None:
        super().__init__(negate=negate)
        self._setup_yahpo("iaml_xgboost", instance)
        self.booster = booster
        self.input_keys = ['alpha','colsample_bylevel', 'colsample_bytree', 'eta', 'gamma', 'lambda',
                           'max_depth', 'min_child_weight', 'nrounds', 'rate_drop', 'skip_drop',
                           'subsample', 'trainsize']
        self.int_keys = ['max_depth', 'nrounds']
        self.fidelity_key = 'trainsize'
        self.target_fidelity_value = 1.0

    def evaluate_true(self, X: Tensor) -> Tensor:
        if X.ndim == 1: X = X.unsqueeze(0)
        assert X.shape[1] == self.dim
        configs = []
        for i in range(X.shape[0]):
            x_numpy = X[i].cpu().numpy()
            config = {k: v.item() if hasattr(v, 'item') else v for k, v in zip(self.input_keys, x_numpy)}
            for k_int in self.int_keys:
                if k_int in config:
                    config[k_int] = int(round(config[k_int]))
            config['task_id'] = self.instance_id
            config['booster'] = self.booster
            configs.append(config)
        outputs = self.bench.objective_function(configs)
        objectives = [inst['auc'] for inst in outputs]
        return torch.tensor(objectives, dtype=X.dtype, device=X.device)

# --- HPO_Benchmark (placeholder) ---
class HPO_Benchmark(SyntheticTestFunction):
    pass


# ============================================================
# --- Single-Fidelity (SF) Versions ---
# ============================================================

class AugmentedBraninSF(AugmentedBranin):
    target_fidelity_value = 1.0

    def __init__(self, negate: Optional[bool] = False):
        super().__init__(negate=negate) # Initializes original dim, bounds, optimizers
        self.dim = super().dim - 1
        self._bounds = super()._bounds[:-1]
        
        # Adjust optimizers: keep only those at target_fidelity_value and remove fidelity component
        new_optimizers = []
        if hasattr(super(), "_optimizers") and super()._optimizers is not None:
            for opt in super()._optimizers:
                if len(opt) == super().dim and math.isclose(opt[-1], self.target_fidelity_value):
                    new_optimizers.append(opt[:-1]) # remove fidelity component
        # If Branin has standard optimizers independent of our fidelity augmentation (at target fid)
        # (x1, x2) pairs for Branin: (-pi, 12.275), (pi, 2.275), (9.42478, 2.475)
        branin_optimizers_x_only = [(-math.pi, 12.275), (math.pi, 2.275), (3 * math.pi, 2.475)]
        self._optimizers = branin_optimizers_x_only # Use known Branin optimizers directly
        # _optimal_value is already defined for target fidelity by parent

    def evaluate_true(self, X: Tensor) -> Tensor:
        # X is (num_samples, self.dim) which is (num_samples, 2)
        if X.ndim == 1: X = X.unsqueeze(0)
        assert X.shape[1] == self.dim, f"Input X for SF should have {self.dim} columns"
        
        fidelity_tensor = torch.full(
            (X.shape[0], 1), self.target_fidelity_value, dtype=X.dtype, device=X.device
        )
        X_augmented = torch.cat((X, fidelity_tensor), dim=1)
        return super().evaluate_true(X_augmented)


class AugmentedRastriginSF(AugmentedRastrigin):
    target_fidelity_value = 1.0

    def __init__(self, negate: Optional[bool] = False):
        super().__init__(negate=negate)
        self.dim = super().dim - 1 # Becomes 1
        self._bounds = super()._bounds[:-1]
        # _optimal_value is already defined for target fidelity (0.0 for maximization)
        # Adjust optimizer
        if hasattr(super(), "_optimizers") and super()._optimizers is not None and len(super()._optimizers[0]) == super().dim +1:
             self._optimizers = [opt[:-1] for opt in super()._optimizers if math.isclose(opt[-1], self.target_fidelity_value)] # [(0.0,)]
        else: # Fallback if not set or wrong format
            self._optimizers = [(0.0,)]


    def evaluate_true(self, X: Tensor) -> Tensor:
        if X.ndim == 1: X = X.unsqueeze(0)
        assert X.shape[1] == self.dim, f"Input X for SF should have {self.dim} columns"
        fidelity_tensor = torch.full(
            (X.shape[0], 1), self.target_fidelity_value, dtype=X.dtype, device=X.device
        )
        X_augmented = torch.cat((X, fidelity_tensor), dim=1)
        return super().evaluate_true(X_augmented)


class AugmentedRastrigin20DSF(AugmentedRastrigin20D):
    target_fidelity_value = 1.0

    def __init__(self, negate: Optional[bool] = False):
        super().__init__(negate=negate)
        self.dim = super().dim - 1 # Becomes 20
        self._bounds = super()._bounds[:-1]
        # _optimal_value is already defined for target fidelity
        if hasattr(super(), "_optimizers") and super()._optimizers is not None and len(super()._optimizers[0]) == super().dim+1 :
            self._optimizers = [opt[:-1] for opt in super()._optimizers if math.isclose(opt[-1], self.target_fidelity_value)]
        else: # Fallback
            self._optimizers = [(0.0,) * self.dim]


    def evaluate_true(self, X: Tensor) -> Tensor:
        if X.ndim == 1: X = X.unsqueeze(0)
        assert X.shape[1] == self.dim, f"Input X for SF should have {self.dim} columns"
        fidelity_tensor = torch.full(
            (X.shape[0], 1), self.target_fidelity_value, dtype=X.dtype, device=X.device
        )
        X_augmented = torch.cat((X, fidelity_tensor), dim=1)
        return super().evaluate_true(X_augmented)



# --- SF YAHPO ---
class LCBenchSF(LCBench):
    def __init__(self, negate: Optional[bool] = False, instance: str = '3945'):
        super().__init__(negate=negate, instance=instance) # Sets up MF version
        
        self.original_input_keys = list(self.input_keys)
        self.original_bounds = list(self._bounds)
        self.original_dim = self.dim

        # Remove fidelity 'epoch' from input_keys and adjust dim and bounds
        fidelity_idx = self.input_keys.index(self.fidelity_key) # 'epoch'
        
        self.dim = self.original_dim - 1
        self.input_keys = self.original_input_keys[:fidelity_idx] + self.original_input_keys[fidelity_idx+1:]
        self._bounds = self.original_bounds[:fidelity_idx] + self.original_bounds[fidelity_idx+1:]
        # _optimal_value and _optimizers are not set by parent YAHPO, so nothing to adjust here.

    def evaluate_true(self, X: Tensor) -> Tensor:
        # X is (num_samples, self.dim) which is (num_samples, 7)
        if X.ndim == 1: X = X.unsqueeze(0)
        assert X.shape[1] == self.dim, f"Input X for SF should have {self.dim} columns"

        configs = []
        for i in range(X.shape[0]):
            x_numpy = X[i].cpu().numpy()
            # Zip with self.input_keys (which doesn't have fidelity key anymore)
            config = {k: v.item() if hasattr(v, 'item') else v for k, v in zip(self.input_keys, x_numpy)}
            # Add the fixed target fidelity value
            config[self.fidelity_key] = self.target_fidelity_value # e.g., epoch = 52

            # Ensure correct types for all original keys that are still present or fixed
            for k_orig in self.original_input_keys:
                if k_orig in config and k_orig in self.int_keys:
                     config[k_orig] = int(round(config[k_orig]))

            config['OpenML_task_id'] = self.instance_id
            configs.append(config)

        outputs = self.bench.objective_function(configs)
        objectives = [inst['test_balanced_accuracy'] for inst in outputs]
        return torch.tensor(objectives, dtype=X.dtype, device=X.device)


class iaml_rpartSF(iaml_rpart):
    def __init__(self, negate: Optional[bool] = False, instance: str = '1489'):
        super().__init__(negate=negate, instance=instance)
        self.original_input_keys = list(self.input_keys)
        self.original_bounds = list(self._bounds)
        self.original_dim = self.dim

        fidelity_idx = self.input_keys.index(self.fidelity_key) # 'trainsize'
        self.dim = self.original_dim - 1
        self.input_keys = self.original_input_keys[:fidelity_idx] + self.original_input_keys[fidelity_idx+1:]
        self._bounds = self.original_bounds[:fidelity_idx] + self.original_bounds[fidelity_idx+1:]

    def evaluate_true(self, X: Tensor) -> Tensor:
        if X.ndim == 1: X = X.unsqueeze(0)
        assert X.shape[1] == self.dim
        configs = []
        for i in range(X.shape[0]):
            x_numpy = X[i].cpu().numpy()
            config = {k: v.item() if hasattr(v, 'item') else v for k, v in zip(self.input_keys, x_numpy)}
            config[self.fidelity_key] = self.target_fidelity_value # e.g., trainsize = 1.0

            for k_orig in self.original_input_keys:
                 if k_orig in config and k_orig in self.int_keys:
                    config[k_orig] = int(round(config[k_orig]))
            config['task_id'] = self.instance_id
            configs.append(config)
        outputs = self.bench.objective_function(configs)
        objectives = [inst['auc'] for inst in outputs]
        return torch.tensor(objectives, dtype=X.dtype, device=X.device)


class iaml_xgboostSF(iaml_xgboost):
    def __init__(self, negate: Optional[bool] = False, instance: str = '1489', booster:str = 'dart'):
        super().__init__(negate=negate, instance=instance, booster=booster)
        self.original_input_keys = list(self.input_keys)
        self.original_bounds = list(self._bounds)
        self.original_dim = self.dim

        fidelity_idx = self.input_keys.index(self.fidelity_key) # 'trainsize'
        self.dim = self.original_dim - 1
        self.input_keys = self.original_input_keys[:fidelity_idx] + self.original_input_keys[fidelity_idx+1:]
        self._bounds = self.original_bounds[:fidelity_idx] + self.original_bounds[fidelity_idx+1:]

    def evaluate_true(self, X: Tensor) -> Tensor:
        if X.ndim == 1: X = X.unsqueeze(0)
        assert X.shape[1] == self.dim
        configs = []
        for i in range(X.shape[0]):
            x_numpy = X[i].cpu().numpy()
            config = {k: v.item() if hasattr(v, 'item') else v for k, v in zip(self.input_keys, x_numpy)}
            config[self.fidelity_key] = self.target_fidelity_value # e.g., trainsize = 1.0

            for k_orig in self.original_input_keys:
                if k_orig in config and k_orig in self.int_keys:
                    config[k_orig] = int(round(config[k_orig]))
            config['task_id'] = self.instance_id
            config['booster'] = self.booster
            configs.append(config)
        outputs = self.bench.objective_function(configs)
        objectives = [inst['auc'] for inst in outputs]
        return torch.tensor(objectives, dtype=X.dtype, device=X.device)


if __name__ == '__main__':
    tkwargs = {
        "dtype": torch.double,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

    print("Testing Single-Fidelity AugmentedBraninSF...")
    branin_sf = AugmentedBraninSF(negate=False).to(**tkwargs)
    print(f"  dim: {branin_sf.dim}, bounds: {branin_sf._bounds}")
    print(f"  Optimal value: {branin_sf._optimal_value}, Optimizers: {branin_sf._optimizers}")
    test_X_branin = torch.tensor([[-math.pi, 12.275], [0.0, 0.0]], **tkwargs)
    test_Y_branin = branin_sf(test_X_branin)
    print(f"  X_sf:\n{test_X_branin}\n  Y_sf (target fid):\n{test_Y_branin}")
    assert torch.isclose(test_Y_branin[0], torch.tensor(branin_sf._optimal_value, **tkwargs))

    print("\nTesting Single-Fidelity AugmentedRastriginSF...")
    rastrigin_sf = AugmentedRastriginSF(negate=False).to(**tkwargs) # Maximize
    print(f"  dim: {rastrigin_sf.dim}, bounds: {rastrigin_sf._bounds}")
    print(f"  Optimal value: {rastrigin_sf._optimal_value}, Optimizers: {rastrigin_sf._optimizers}") # Should be 0
    test_X_rastrigin = torch.tensor([[0.0], [1.0]], **tkwargs)
    test_Y_rastrigin = rastrigin_sf(test_X_rastrigin)
    print(f"  X_sf:\n{test_X_rastrigin}\n  Y_sf (target fid):\n{test_Y_rastrigin}")
    assert torch.isclose(test_Y_rastrigin[0], torch.tensor(rastrigin_sf._optimal_value, **tkwargs))

    rastrigin_sf_neg = AugmentedRastriginSF(negate=True).to(**tkwargs) # Minimize
    print(f"  Negated Optimal value: {rastrigin_sf_neg._optimal_value}, Optimizers: {rastrigin_sf_neg._optimizers}")
    test_Y_rastrigin_neg = rastrigin_sf_neg(test_X_rastrigin)
    print(f"  X_sf:\n{test_X_rastrigin}\n  Y_sf_neg (target fid):\n{test_Y_rastrigin_neg}")
    assert torch.isclose(test_Y_rastrigin_neg[0], torch.tensor(rastrigin_sf_neg._optimal_value, **tkwargs))


    print("\nTesting Single-Fidelity AugmentedRastrigin20DSF...")
    rastrigin20d_sf = AugmentedRastrigin20DSF(negate=False).to(**tkwargs)
    print(f"  dim: {rastrigin20d_sf.dim}, bounds: {len(rastrigin20d_sf._bounds)} pairs")
    print(f"  Optimal value: {rastrigin20d_sf._optimal_value}, Optimizers: {rastrigin20d_sf._optimizers[0][:2]}... (truncated)")
    test_X_rastrigin20d = torch.zeros(1, rastrigin20d_sf.dim, **tkwargs)
    test_Y_rastrigin20d = rastrigin20d_sf(test_X_rastrigin20d)
    print(f"  X_sf (zeros):\n{test_X_rastrigin20d[0,:5]}...\n  Y_sf (target fid):\n{test_Y_rastrigin20d}")
    assert torch.isclose(test_Y_rastrigin20d[0], torch.tensor(rastrigin20d_sf._optimal_value, **tkwargs), atol=1e-5)




    # --- YAHPO SF Tests ---
    # Need yahpo_gym installed and data downloaded for these to run
    print("\nTesting Single-Fidelity LCBenchSF...")
    try:
        lcbench_sf = LCBenchSF(negate=False, instance='3945').to(**tkwargs) # Test with a small instance
        print(f"  dim: {lcbench_sf.dim}, bounds: {len(lcbench_sf._bounds)} pairs")
        print(f"  Fidelity key: {lcbench_sf.fidelity_key}, Target value: {lcbench_sf.target_fidelity_value}")
        # Construct a valid input (mid-point of bounds)
        mid_point_X = torch.tensor([(b[0] + b[1]) / 2 for b in lcbench_sf._bounds], **tkwargs).unsqueeze(0)
        if lcbench_sf.int_keys: # round integer params for yahpo
            for i, key in enumerate(lcbench_sf.input_keys): # use SF input keys
                if key in lcbench_sf.int_keys: # use original int_keys for checking
                    mid_point_X[0, i] = round(mid_point_X[0, i].item())

        print(f"  Test X (mid bounds): {mid_point_X[:,:3]}...")
        test_Y_lcbench = lcbench_sf(mid_point_X)
        print(f"  Y_sf (target fid):\n{test_Y_lcbench}")
    except Exception as e:
        print(f"Could not run LCBenchSF test: {e}. Make sure yahpo_gym is installed and data is available.")

    print("\nTesting Single-Fidelity iaml_rpartSF...")
    try:
        rpart_sf = iaml_rpartSF(negate=False, instance='1489').to(**tkwargs)
        print(f"  dim: {rpart_sf.dim}, bounds: {len(rpart_sf._bounds)} pairs")
        mid_point_X_rpart = torch.tensor([(b[0] + b[1]) / 2 for b in rpart_sf._bounds], **tkwargs).unsqueeze(0)
        if rpart_sf.int_keys:
             for i, key in enumerate(rpart_sf.input_keys):
                if key in rpart_sf.int_keys:
                    mid_point_X_rpart[0, i] = round(mid_point_X_rpart[0, i].item())
        print(f"  Test X (mid bounds): {mid_point_X_rpart[:,:3]}...")
        test_Y_rpart = rpart_sf(mid_point_X_rpart)
        print(f"  Y_sf (target fid):\n{test_Y_rpart}")
    except Exception as e:
        print(f"Could not run iaml_rpartSF test: {e}.")

    print("\nTesting Single-Fidelity iaml_xgboostSF...")
    try:
        xgboost_sf = iaml_xgboostSF(negate=False, instance='1489').to(**tkwargs)
        print(f"  dim: {xgboost_sf.dim}, bounds: {len(xgboost_sf._bounds)} pairs")
        mid_point_X_xgb = torch.tensor([(b[0] + b[1]) / 2 for b in xgboost_sf._bounds], **tkwargs).unsqueeze(0)
        if xgboost_sf.int_keys:
            for i, key in enumerate(xgboost_sf.input_keys):
                if key in xgboost_sf.int_keys:
                    mid_point_X_xgb[0, i] = round(mid_point_X_xgb[0, i].item())
        print(f"  Test X (mid bounds): {mid_point_X_xgb[:,:3]}...")
        test_Y_xgb = xgboost_sf(mid_point_X_xgb)
        print(f"  Y_sf (target fid):\n{test_Y_xgb}")
    except Exception as e:
        print(f"Could not run iaml_xgboostSF test: {e}.")