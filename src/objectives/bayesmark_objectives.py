# Copyright (c) 2019 Uber Technologies, Inc.
import json
import sys
import logging
import random as pyrandom
import uuid
import warnings
from collections import OrderedDict # Import OrderedDict
from time import sleep, time
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import torch
import xarray as xr # Retained from original context
from botorch.test_functions.synthetic import SyntheticTestFunction

# Attempt to import bayesmark components. These are expected to be in the environment.
try:
    import bayesmark.cmd_parse as cmd
    import bayesmark.constants as cc
    import bayesmark.random_search as rs
    from bayesmark.builtin_opt.config import CONFIG 
    from bayesmark.cmd_parse import CmdArgs 
    from bayesmark.constants import ARG_DELIM, ITER, OBJECTIVE, SUGGEST 
    from bayesmark.data import METRICS_LOOKUP, get_problem_type 
    # linear_rescale is not directly used by the wrapper's core logic anymore,
    # but kept for potential use in mock or by user.
    from bayesmark.np_util import argmin_2d, linear_rescale, random_seed 
    from bayesmark.serialize import XRSerializer 
    from bayesmark.signatures import analyze_signature_pair, get_func_signature 
    from bayesmark.sklearn_funcs import SklearnModel, SklearnSurrogate 
    from bayesmark.space import JointSpace 
    from bayesmark.util import chomp, str_join_safe 
    from bayesmark.experiment import _build_test_problem # User's requested import

except ImportError as e:
    print(f"Warning: Bayesmark library components could not be imported. "
          f"The BayesmarkBoTorchWrapper may not function correctly. Error: {e}")
    # Define dummy/placeholder classes/variables if bayesmark is not available
    class SklearnModel: 
        api_config: Dict[str, Any]
        def evaluate(self, params: Dict[str, Any]) -> Any: pass
    class SklearnSurrogate: 
        api_config: Dict[str, Any]
        space: Any
        def evaluate(self, params: Dict[str, Any]) -> Any: pass

    class JointSpace: # User's provided dummy JointSpace for mock problem setup
        def __init__(self, api_config: Dict[str, Any]):
            self.api_config = api_config
            self.dimension = 0 # Matches user's dummy
            self.bounds: List[Tuple[float, float]] = []
            self.param_names: List[str] = []
            for name, details in api_config.items():
                if "range" in details and isinstance(details["range"], tuple):
                    self.dimension +=1
                    self.bounds.append(details["range"])
                    self.param_names.append(name)
            
    METRICS_LOOKUP = {}
    def chomp(s, suffix): return s
    def linear_rescale(x, old_min, old_max, new_min=0.0, new_max=1.0, clip=True): 
        if old_min == old_max: return new_min if x <= old_min else new_max
        res = (x - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
        if clip:
            b_min, b_max = min(new_min, new_max), max(new_min, new_max)
            return min(max(res, b_min), b_max)
        return res

logger = logging.getLogger(__name__)

class BayesmarkBoTorchWrapper(SyntheticTestFunction):
    r"""A BoTorch SyntheticTestFunction wrapper for bayesmark problems.
    """

    def __init__(
        self,
        model_name: str,
        dataset: str,
        scorer: str,
        bayesmark_path: Optional[str] = None,
        noise_std: Optional[float] = None,
    ) -> None:
        """
        Initialize the BayesmarkBoTorchWrapper.
        """
        self.model_name = model_name
        self.dataset = dataset
        self.scorer = scorer
        self.bayesmark_path = bayesmark_path

        # Build the underlying bayesmark problem
        try:
            self.bayesmark_problem = _build_test_problem( # Uses imported _build_test_problem
                model_name, dataset, scorer, bayesmark_path
            )
        except Exception as e:
            logger.error(f"Failed to build bayesmark problem: {e}")
            raise

        # Ensure the bayesmark problem has an api_config dictionary
        if not hasattr(self.bayesmark_problem, 'api_config') or \
           not isinstance(self.bayesmark_problem.api_config, dict):
            raise AttributeError(
                "Bayesmark problem instance must have an 'api_config' attribute (dict) "
                "to determine dimensions and bounds for BoTorch."
            )
        
        current_api_config = self.bayesmark_problem.api_config

        # Parse api_config to get dim, bounds (original scale), and param_names for BoTorch
        _dim = 0
        _bounds_list: List[Tuple[float, float]] = []
        _param_names_for_botorch: List[str] = []

        # Use OrderedDict if current_api_config is not already one, or sort keys for consistent order
        # Sorting by key is a common practice for reproducibility.
        # The original bayesmark configs are dicts, so sorting is good.
        param_items = sorted(current_api_config.items())

        for param_name, param_details in param_items:
            param_type = param_details.get("type")
            param_range = param_details.get("range")

            # We are interested in numerical parameters that BoTorch will optimize.
            if (param_type == "real" or param_type == "int") and isinstance(param_range, tuple):
                _dim += 1
                _bounds_list.append(param_range)
                _param_names_for_botorch.append(param_name)
            else:
                # Log parameters not being directly optimized by BoTorch in this setup
                logger.debug(f"Parameter '{param_name}' of type '{param_type}' or without a numerical 'range' "
                             "will not be part of BoTorch's optimization dimensions for this wrapper.")
        
        if _dim == 0:
            raise ValueError(
                "No numerical parameters with 'range' found in api_config. "
                "BoTorch wrapper requires at least one such parameter for its optimization dimensions."
            )

        self.dim = _dim
        self._bounds = _bounds_list # List of (lower, upper) tuples in original scale
        self.param_names = _param_names_for_botorch # Names of parameters BoTorch will vary

        # Store the full api_config for reference, e.g., for int casting in evaluate_true
        self.full_api_config = current_api_config


        # Determine if the problem is maximization or minimization
        try:
            if self.scorer in {'mean_squared_error', 'mse', 'rmse'}:
                is_maximization = False
            elif self.scorer in {'acc', 'f1', 'precision', 'recall', 'accuracy'}:
                is_maximization = True
        except Exception as e: 
            logger.error(f"Error determining optimization direction for scorer '{self.scorer}': {e}. Assuming minimization.")
            is_maximization = False



        # Initialize BoTorch specific attributes before super().__init__
        self._optimizers: Optional[List[Tuple[float, ...]]] = None 
        # self.optimal_value: Optional[float] = None


        super().__init__(noise_std=noise_std, negate=is_maximization)
        # SyntheticTestFunction's __init__ sets self.bounds (tensor) from self._bounds (list of tuples).


    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the underlying bayesmark problem.
        X is a (batch_shape) x d tensor from BoTorch, with values in original parameter scales.
        """
        if X.ndim == 1: X = X.unsqueeze(0) # Ensure X is at least 2D
        if X.shape[-1] != self.dim:
             raise ValueError(f"Input tensor X last dim ({X.shape[-1]}) must match wrapper dim ({self.dim})")

        scores = []
        for i in range(X.shape[0]): # Iterate over batch
            x_row_values = X[i].tolist() # Get i-th parameter configuration
            
            # Create parameter dictionary in original scale for bayesmark_problem.evaluate
            params_dict_original_scale: Dict[str, Any] = {}
            for idx, param_name_for_botorch in enumerate(self.param_names):
                val = x_row_values[idx]
                # Handle integer types: BoTorch provides floats, round them for bayesmark.
                if self.full_api_config[param_name_for_botorch].get("type") == "int":
                    val = int(round(val))
                params_dict_original_scale[param_name_for_botorch] = val
            
            # Note: bayesmark_problem.evaluate might also need other fixed parameters
            # or parameters not varied by BoTorch (e.g., booleans, categoricals not in self.param_names).
            # The current setup assumes bayesmark_problem.evaluate can handle a dict
            # containing only the tunable numerical parameters, or that it merges them
            # with fixed/default values internally. This matches typical bayesmark usage.

            try:
                # bayesmark_problem.evaluate expects a dict of params in original scale
                evaluation_output = self.bayesmark_problem.evaluate(params_dict_original_scale)
                
                current_score: float
                if isinstance(evaluation_output, tuple): # Handle multiple objectives if returned
                    current_score = float(evaluation_output[0]) # Assume first is primary
                else:
                    current_score = float(evaluation_output)
                
                if not np.isfinite(current_score): # Check for NaN/inf
                    logger.warning(f"Non-finite score ({current_score}) "
                                   f"for params: {params_dict_original_scale}. Replacing with float('nan').")
                    current_score = float('nan')
                scores.append(current_score)

            except Exception as e:
                logger.error(f"Error during bayesmark problem evaluation for "
                               f"params {params_dict_original_scale}: {e}")
                scores.append(float('nan')) # Propagate error as NaN

        return torch.tensor(scores, dtype=X.dtype, device=X.device)


if __name__ == '__main__': # User's provided __main__ block
    logging.basicConfig(level=logging.INFO) 
    logger.setLevel(logging.DEBUG) 

    if not METRICS_LOOKUP: 
        print("Mocking METRICS_LOOKUP for example execution.")
        class MockMetricInfo:
            def __init__(self, greater_is_better): self.greater_is_better = greater_is_better
        METRICS_LOOKUP['accuracy'] = MockMetricInfo(greater_is_better=True)
        METRICS_LOOKUP['neg_mean_squared_error'] = MockMetricInfo(greater_is_better=True)
        METRICS_LOOKUP['rmse'] = MockMetricInfo(greater_is_better=False)
        METRICS_LOOKUP['mse'] = MockMetricInfo(greater_is_better=True) # mse is often neg_mean_squared_error, higher is better

    if 'linear_rescale' not in globals() and 'bayesmark' not in sys.modules: # Check if real one was imported
        print("Mocking linear_rescale for example execution.")
        def linear_rescale(x, old_min, old_max, new_min=0.0, new_max=1.0, clip=True):
            if old_min == old_max: 
                return new_min if x <= old_min else new_max
            res = (x - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
            if clip:
                bound_min = min(new_min, new_max)
                bound_max = max(new_min, new_max)
                return min(max(res, bound_min), bound_max)
            return res
    
    # --- Mocking for Bayesmark Problem Classes (SklearnModel, SklearnSurrogate) ---
    # This section uses the user's provided mock structure.
    if 'SklearnModel' not in globals() or SklearnModel.__module__ == '__main__': 
        print("Mocking Bayesmark problem classes (SklearnModel, SklearnSurrogate) for example execution.")
        
        class MockBayesmarkProblem: # User's mock class name
            def __init__(self, model_name_arg, dataset_arg, scorer_arg, data_root=None, path=None):
                self.model_name = model_name_arg 
                self.dataset = dataset_arg
                self.scorer = scorer_arg
                
                if model_name_arg == "ada": # Specific case from user's __main__
                     self.api_config = OrderedDict([
                        ("n_estimators", {"type": "int", "range": (10, 100)}),
                        ("learning_rate", {"type": "real", "range": (1e-4, 1e1)}),
                    ])
                elif model_name_arg == "mock_model_continuous": 
                    self.api_config = OrderedDict([ 
                        ("C", {"type": "real", "space": "log", "range": (0.1, 10.0)}),
                        ("gamma", {"type": "real", "space": "log", "range": (0.01, 1.0)}),
                    ])
                elif model_name_arg == "mock_model_mixed": 
                     self.api_config = OrderedDict([
                        ("n_estimators", {"type": "int", "space": "linear", "range": (10, 100)}),
                        ("learning_rate", {"type": "real", "space": "log", "range": (0.001, 0.1)}),
                    ])
                elif model_name_arg == "mock_surrogate_model": # User's name for surrogate mock
                    self.api_config = OrderedDict([
                        ("s_alpha", {"type": "real", "range": (1.0, 5.0)}),
                    ])
                    self.space = JointSpace(self.api_config) if 'JointSpace' in globals() else None
                else: # Default fallback for other model names
                    self.api_config = OrderedDict([
                        ("param_default", {"type": "real", "range": (0.0,1.0)})
                    ])


            def evaluate(self, params: Dict[str, Any]):
                logger.debug(f"MockBayesmarkProblem '{self.model_name}' evaluating with original scale params: {params}")
                score = 0.0
                if self.model_name == "ada":
                    n_est = params.get("n_estimators", 50)
                    lr = params.get("learning_rate", 0.1)
                    if not isinstance(n_est, int): logger.warning(f"n_estimators not int: {n_est}")
                    score = -( (n_est/50.0 -1 )**2 + (np.log10(lr) + 1)**2 ) # Example for ada
                elif self.model_name == "mock_model_continuous":
                    score = -(params.get("C", 0)**2 + params.get("gamma", 0)**2)
                elif self.model_name == "mock_model_mixed":
                    n_est = params.get("n_estimators", 10)
                    lr = params.get("learning_rate", 0.01)
                    if not isinstance(n_est, int): logger.warning(f"n_estimators not int: {n_est}")
                    score = -( (n_est/50.0 - 1)**2 + (lr*100 - 5)**2 ) 
                elif self.model_name == "mock_surrogate_model":
                    score = -params.get("s_alpha", 0)**2
                else: # Default fallback
                    score = -sum(v**2 for v in params.values() if isinstance(v, (int, float)))
                
                logger.debug(f"Mock evaluation score: {score}")
                return score 

        globals()['SklearnModel'] = MockBayesmarkProblem 
        globals()['SklearnSurrogate'] = MockBayesmarkProblem 

        # Check if _build_test_problem was successfully imported from bayesmark.experiment
        # If not (e.g., bayesmark not installed), then mock it.
        if '_build_test_problem' not in globals() or \
           (hasattr(sys.modules.get('bayesmark.experiment'), '_build_test_problem') and \
            globals()['_build_test_problem'] != sys.modules['bayesmark.experiment']._build_test_problem):

            _original_build_test_problem_ref = globals().get('_build_test_problem') # Save if it exists from import

            def _mock_build_test_problem_for_main(model_name_arg, dataset_arg, scorer_arg, path_arg):
                print(f"Using MOCK _build_test_problem (for __main__) for {model_name_arg}, {dataset_arg}, {scorer_arg}")
                if model_name_arg.endswith("-surr"):
                    # Use the specific name for surrogate mock recognized by MockBayesmarkProblem
                    return SklearnSurrogate("mock_surrogate_model", dataset_arg, scorer_arg, path=path_arg)
                # For non-surrogate, just pass the model_name_arg; MockBayesmarkProblem will handle it.
                return SklearnModel(model_name_arg, dataset_arg, scorer_arg, data_root=path_arg) 
            
            globals()['_build_test_problem'] = _mock_build_test_problem_for_main
            # Store a way to restore, in case the real one was imported but we overrode for the mock
            if _original_build_test_problem_ref:
                globals()['_actual_build_test_problem_before_mock'] = _original_build_test_problem_ref


    print("\n--- Example Usage of BayesmarkBoTorchWrapper (User's Main Example) ---")
    try:
        # User's example: model_name="ada", dataset="boston", scorer="mse"
        # This will use the "ada" case in the MockBayesmarkProblem
        # api_config for "ada": n_estimators (int, 10-100), learning_rate (real, 1e-4, 1e1) -> dim=2
        problem_user_main = BayesmarkBoTorchWrapper(
            model_name="ada", 
            dataset="wine", 
            scorer="acc", 
        )

        print(f"Initialized BayesmarkBoTorchWrapper for 'ada' model.")
        print(f"Dimension: {problem_user_main.dim}") # Should be 2
        print(f"Internal _bounds (list of tuples): {problem_user_main._bounds}")
        print(f"Parameter names: {problem_user_main.param_names}")
        print(f"BoTorch bounds (tensor): {problem_user_main.bounds}")
        print(f"Negate (for BoTorch maximization): {problem_user_main.negate}") # Should be True for mse

        # Test points for "ada" model: (n_estimators, learning_rate)
        # Ranges: n_estimators (10-100), learning_rate (0.0001-10.0)
        test_X_ada = torch.tensor([
            [0.0001, 10.0], # Min bounds
            [0.1, 55.0,],    # Mid-ish point (55 will be rounded)
            [10, 100]   # Max bounds
        ], dtype=torch.float64)
        print(f"\nEvaluating 'ada' model with X (original scale):\n{test_X_ada}")
        raw_Y_ada = problem_user_main.evaluate_true(test_X_ada)
        print(f"Raw Y for 'ada' model (scores from evaluate_true directly):\n{raw_Y_ada}")
        # Example calculation for [10, 0.0001]: n_est=10, lr=1e-4. log10(1e-4) = -4.
        # score = -((10/50-1)^2 + (-4+1)^2) = -((-0.8)^2 + (-3)^2) = -(0.64 + 9) = -9.64
        # Example calculation for [55, 0.1]: n_est=55, lr=0.1. log10(0.1) = -1.
        # score = -((55/50-1)^2 + (-1+1)^2) = -((1.1-1)^2 + 0^2) = -(0.1^2) = -0.01
        # Example calculation for [100, 10]: n_est=100, lr=10. log10(10) = 1.
        # score = -((100/50-1)^2 + (1+1)^2) = -((2-1)^2 + 2^2) = -(1^2 + 4) = -5.0

    except Exception as e:
        print(f"An error occurred during the user's main example: {e}")
        import traceback
        traceback.print_exc()

    # print("\n--- Example Usage of BayesmarkBoTorchWrapper (Simulating SklearnSurrogate) ---")
    # try:
    #     problem_surr = BayesmarkBoTorchWrapper(
    #         model_name="any_model-surr", # Triggers surrogate path in _mock_build_test_problem
    #         dataset="mock_data_surr",
    #         scorer="rmse", # rmse is typically minimization (greater_is_better=False)
    #     )

    #     print(f"Initialized BayesmarkBoTorchWrapper for surrogate simulation.")
    #     print(f"Dimension: {problem_surr.dim}")
    #     print(f"Internal _bounds (list of tuples): {problem_surr._bounds}")
    #     print(f"Parameter names: {problem_surr.param_names}")
    #     print(f"BoTorch bounds (tensor): {problem_surr.bounds}")
    #     print(f"Negate (for BoTorch maximization): {problem_surr.negate}") # Should be False for rmse

    #     # mock_surrogate_model (api_config for "s_alpha") has one param with range [1.0, 5.0]
    #     test_X_surr = torch.tensor([[1.0], [2.5], [5.0]], dtype=torch.float64)
    #     print(f"\nEvaluating surrogate simulation with X (original scale):\n{test_X_surr}")
    #     raw_Y_surr = problem_surr.evaluate_true(test_X_surr)
    #     print(f"Raw Y for surrogate model (scores from evaluate_true directly):\n{raw_Y_surr}")
    #     # Mock evaluate: -s_alpha^2. For 1.0 -> -1. For 2.5 -> -6.25. For 5.0 -> -25

    # except Exception as e:
    #     print(f"An error occurred during the surrogate example: {e}")
    #     import traceback
    #     traceback.print_exc()
    # finally:
    #     # Restore original _build_test_problem if it was mocked and an original existed
    #     if '_actual_build_test_problem_before_mock' in globals():
    #         globals()['_build_test_problem'] = globals()['_actual_build_test_problem_before_mock']
    #     elif '_original_build_test_problem_ref' in globals() and not hasattr(sys.modules.get('bayesmark.experiment'), '_build_test_problem'):
    #         # If the original was None (meaning it wasn't imported), and we mocked it,
    #         # there's nothing to restore it to other than potentially removing our mock.
    #         # This case is tricky; for simplicity, if it wasn't there to begin with, our mock stays.
    #          pass
