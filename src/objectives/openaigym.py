import numpy as np
try:
    import gymnasium as gym
    IS_GYMNASIUM = True

except ImportError:
    import gym # Older gym versions
    IS_GYMNASIUM = False
    print("Warning: gymnasium not found, falling back to gym. API differences might exist.")
    print("Please install gymnasium for the most up-to-date experience: pip install gymnasium[box2d]")

import torch
from botorch.test_functions.synthetic import SyntheticTestFunction
from typing import Optional

seed = 42  # Global seed for reproducibility

class LunarLanderProblem(SyntheticTestFunction):
    dim = 12  # 12 parameters for the heuristic controller
    _bounds = [(0.0, 1.2)] * dim # Search range for controller parameters
    
    # The true global optimum is unknown for this heuristic tuning problem.
    # We aim to maximize reward, so if negate=True, Trieste will minimize -reward.
    # We can set an optimistic target for `optimal_value` if needed for regret plotting,
    # e.g., -250 (negative of max possible reward), or leave as None.
    optimal_value = None # Represents the value Trieste tries to achieve (e.g. min of -reward)
    _optimizers = None   # Location of the optimum is unknown

    def __init__(
        self,
        negate: bool = True, # True: maximize reward (Trieste minimizes -reward)
        noise_std: Optional[float] = None, # Effective noise of the *averaged* reward
        n_runs: int = 10,
        steps_limit: int = 1000,
        timeout_reward: float = -100.0, # Penalty for timeout
        problem_seed: Optional[int] = seed # Use the global script seed by default
    ):
        super().__init__(noise_std=noise_std, negate=negate) # Pass noise_std for the averaged reward
        self.n_runs = n_runs
        self.steps_limit = steps_limit
        self.timeout_reward = timeout_reward # This is the direct penalty value
        self.problem_seed = problem_seed

        self.env_name = "LunarLander-v3"
        # Attempt to create and close a test environment to check for installation issues
        try:
            test_env = gym.make(self.env_name)
            test_env.close()
        except Exception as e:
            print(f"CRITICAL: Could not initialize gym environment '{self.env_name}' during LunarLanderProblem setup. "
                  f"Ensure 'gymnasium' and Box2D are correctly installed ('pip install gymnasium[box2d]'). Error: {e}")
            raise

    @staticmethod
    def _heuristic_controller(s_state, w_params_torch: torch.Tensor):
        # Convert controller parameters tensor to numpy array for existing logic
        w_params = w_params_torch.cpu().numpy()
        
        angle_targ = s_state[0] * w_params[0] + s_state[2] * w_params[1]
        if angle_targ > w_params[2]: angle_targ = w_params[2]
        if angle_targ < -w_params[2]: angle_targ = -w_params[2]
        hover_targ = w_params[3] * np.abs(s_state[0])

        angle_todo = (angle_targ - s_state[4]) * w_params[4] - (s_state[5]) * w_params[5]
        hover_todo = (hover_targ - s_state[1]) * w_params[6] - (s_state[3]) * w_params[7]

        if s_state[6] or s_state[7]:  # Leg contact
            angle_todo = w_params[8]
            hover_todo = -(s_state[3]) * w_params[9]

        action = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > w_params[10]: action = 2
        elif angle_todo < -w_params[11]: action = 3
        elif angle_todo > +w_params[11]: action = 1
        return action

    def _run_single_simulation(self, env_instance, w_params_torch: torch.Tensor, sim_seed: Optional[int]) -> float:
        total_reward = 0.0
        steps_count = 0
        
        if IS_GYMNASIUM:
            s_state, _ = env_instance.reset(seed=sim_seed)
        else: # older gym
            s_state = env_instance.reset() # Seed might not be directly settable in reset for very old gym
            if sim_seed is not None: env_instance.seed(sim_seed)


        while True:
            if steps_count >= self.steps_limit:
                total_reward += self.timeout_reward # Add penalty directly
                break

            action = self._heuristic_controller(s_state, w_params_torch)
            
            if IS_GYMNASIUM:
                s_state, reward_val, terminated, truncated, _ = env_instance.step(action)
                done_flag = terminated or truncated
            else: # older gym
                s_state, reward_val, done_flag, _ = env_instance.step(action)

            total_reward += reward_val
            steps_count += 1
            if done_flag:
                break
        return total_reward

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        # X is a tensor of shape (batch_size, 12)
        batch_avg_rewards = []
        # It's safer to create the env instance here if evaluate_true can be called from different processes
        # or if the environment state isn't perfectly reset. For simplicity and common BO use cases,
        # creating it once per batch evaluation is a robust approach.
        env_instance = gym.make(self.env_name)

        for i in range(X.shape[0]):  # For each set of controller parameters 'w' in the batch
            w_single_set = X[i]
            rewards_for_this_w = []
            for run_idx in range(self.n_runs):
                # Create a unique seed for each simulation run to ensure diverse trajectories for averaging
                # while maintaining overall reproducibility if problem_seed is set.
                current_sim_seed = (self.problem_seed + i * self.n_runs + run_idx) if self.problem_seed is not None else None
                
                run_reward = self._run_single_simulation(env_instance, w_single_set, sim_seed=current_sim_seed)
                rewards_for_this_w.append(run_reward)
            
            batch_avg_rewards.append(np.mean(rewards_for_this_w))
        
        env_instance.close()
        
        # Return rewards (higher is better). If self.negate is True, BO will minimize -rewards.
        return torch.tensor(batch_avg_rewards, dtype=X.dtype, device=X.device)
