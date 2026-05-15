"""
src/utils/fidelity_metrics.py

Simulation-fidelity metrics for RQ2.

There are two distinct experimental setups with different valid metrics:

  Setup A  (GP-guided query points, true f(x) called as counterfactual)
  -------
  Produced by DRO._simulate_trajectory() + save_fidelity_logs().
  At each rollout step, the query point x is chosen by the GP's acquisition
  function using sim_observed_best.  We then call f(x) at the SAME point
  and compare the GP's imagined value with the true value.

  Valid metrics:
    - per_step_reward_rmse   RMSE between simulated y and true f(x) per step
    - per_step_bias          signed mean error per step (overoptimism direction)
    - rtg_rmse               RMSE between simulated RTG and counterfactual RTG
    - rtg_bias               signed mean RTG error

  NOT valid:
    - Wasserstein distance   (trajectories are not independent — query points
                              were still selected by the GP's view of the world)

  Setup B  (true_observed_best drives the acquisition — independent true BO)
  -------
  Produced by DRO._generate_true_trajectory() + save_true_trajectory_logs().
  Query points diverge from the simulated path because true_observed_best
  (updated by f(x)) reshapes the acquisition landscape at every step.

  Valid metrics:
    - Wasserstein distance   compare sim_final_regrets (Setup A) vs
                             true_final_regrets (Setup B)
    - DT-GT training         trajectories are in DT-compatible format

Expected array shapes (from the saved .npz files):
  Setup A npz:  simulated_ys, true_ys, sim_rewards, true_rewards  → [I, R, T]
                sim_final_regrets, counterfactual_final_regrets    → [I, R]
  Setup B npz:  true_ys, true_rewards                             → [I, R, T]
                true_final_regrets                                  → [I, R]
  (I = BO iterations, R = rollouts per iteration, T = max rollout length)
"""

import numpy as np
from scipy.stats import wasserstein_distance
from typing import Dict, Union


# ---------------------------------------------------------------------------
# Core building block: return-to-go
# ---------------------------------------------------------------------------

def compute_rtg(rewards: np.ndarray) -> np.ndarray:
    """
    Compute return-to-go (RTG) from a 1-D reward sequence.

    RTG[t] = sum of rewards from step t to the last non-NaN step.
    Trailing NaNs (early-stopped padding) are preserved as NaN in the output.

    Parameters
    ----------
    rewards : shape [T], may contain trailing NaNs.

    Returns
    -------
    rtg : shape [T], NaN where rewards is NaN.

    Example
    -------
    >>> compute_rtg(np.array([0.0, 0.5, 0.0, 0.3]))
    array([0.8, 0.8, 0.3, 0.3])
    """
    rtg = np.full_like(rewards, np.nan, dtype=float)
    valid = ~np.isnan(rewards)
    if not valid.any():
        return rtg
    r_filled = np.where(valid, rewards, 0.0)
    rtg_filled = np.cumsum(r_filled[::-1])[::-1]
    rtg[valid] = rtg_filled[valid]
    return rtg


def _rtg_matrix(rewards_matrix: np.ndarray) -> np.ndarray:
    """Apply compute_rtg row-wise.  Input [R, T] → output [R, T]."""
    return np.array([compute_rtg(row) for row in rewards_matrix])


# ---------------------------------------------------------------------------
# Setup A metrics  (same query points — valid for per-step comparison)
# ---------------------------------------------------------------------------

def rtg_rmse(sim_rewards: np.ndarray, true_rewards: np.ndarray) -> float:
    """
    [Setup A] Root-mean-square error between simulated RTG and counterfactual
    RTG (true rewards at the GP's chosen points), averaged over all rollouts
    and steps.

    A large value means the GP's imagined future returns differ substantially
    from the true returns achievable at those same query points.

    Parameters
    ----------
    sim_rewards  : [R, T]  GP-improvement rewards (NaN-padded)
    true_rewards : [R, T]  counterfactual true-improvement rewards (NaN-padded)

    Returns
    -------
    scalar RMSE
    """
    sim_rtg  = _rtg_matrix(sim_rewards)
    true_rtg = _rtg_matrix(true_rewards)
    return float(np.sqrt(np.nanmean((sim_rtg - true_rtg) ** 2)))


def rtg_bias(sim_rewards: np.ndarray, true_rewards: np.ndarray) -> float:
    """
    [Setup A] Signed mean error of simulated RTG relative to counterfactual
    RTG (sim − true), averaged over all rollouts and steps.

    Positive → GP is overoptimistic (inflated simulated returns).
    Negative → GP is pessimistic.

    Parameters
    ----------
    sim_rewards  : [R, T]
    true_rewards : [R, T]

    Returns
    -------
    scalar bias
    """
    sim_rtg  = _rtg_matrix(sim_rewards)
    true_rtg = _rtg_matrix(true_rewards)
    return float(np.nanmean(sim_rtg - true_rtg))


def per_step_reward_rmse(sim_ys: np.ndarray, true_ys: np.ndarray) -> np.ndarray:
    """
    [Setup A] RMSE between raw GP samples and true f(x) at each rollout step,
    averaged across rollouts.

    Plotting this against step index t reveals whether GP prediction error
    accumulates over the course of a rollout.

    Parameters
    ----------
    sim_ys  : [R, T]  raw GP posterior samples
    true_ys : [R, T]  true f(x) at the same query points

    Returns
    -------
    shape [T] — one RMSE per rollout step
    """
    return np.sqrt(np.nanmean((sim_ys - true_ys) ** 2, axis=0))


def per_step_bias(sim_ys: np.ndarray, true_ys: np.ndarray) -> np.ndarray:
    """
    [Setup A] Signed mean error (sim − true) at each rollout step.

    Positive at step t → GP systematically overestimates f(x) at that step.

    Parameters
    ----------
    sim_ys  : [R, T]
    true_ys : [R, T]

    Returns
    -------
    shape [T]
    """
    return np.nanmean(sim_ys - true_ys, axis=0)


# ---------------------------------------------------------------------------
# Setup B metric  (independent true BO — valid for outcome distribution)
# ---------------------------------------------------------------------------

def wasserstein_final_regret(
    sim_final_regrets: np.ndarray,
    true_final_regrets: np.ndarray,
) -> float:
    """
    [Setup B] Wasserstein-1 distance between the distribution of final
    simulated regrets (Setup A) and the distribution of final true regrets
    (Setup B independent true trajectories).

    IMPORTANT: true_final_regrets must come from Setup B logs
    (save_true_trajectory_logs), NOT from Setup A's counterfactual_final_regrets.
    The latter are reached along GP-guided paths and are NOT independent.

    A large Wasserstein value means the GP simulation produces systematically
    different final outcomes than an independent true BO would reach.

    Parameters
    ----------
    sim_final_regrets  : [R]  from Setup A log, key 'sim_final_regrets'
    true_final_regrets : [R]  from Setup B log, key 'true_final_regrets'

    Returns
    -------
    scalar Wasserstein-1 distance
    """
    return float(wasserstein_distance(sim_final_regrets, true_final_regrets))


# ---------------------------------------------------------------------------
# Bundle: all Setup A metrics for a single BO iteration
# ---------------------------------------------------------------------------

def compute_all_metrics_setup_a(
    log: Dict[str, np.ndarray]
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Compute all Setup A fidelity metrics from one BO iteration's log dict.

    Does NOT include Wasserstein (requires Setup B).

    Parameters
    ----------
    log : dict with keys
        'simulated_ys'                [R, T]
        'true_ys'                     [R, T]
        'sim_rewards'                 [R, T]
        'true_rewards'                [R, T]
        'sim_final_regrets'           [R]
        'counterfactual_final_regrets'[R]  (present but not used here)

    Returns
    -------
    dict with keys:
        'rtg_rmse'              float
        'rtg_bias'              float
        'per_step_reward_rmse'  array [T]
        'per_step_bias'         array [T]
    """
    return {
        'rtg_rmse':             rtg_rmse(log['sim_rewards'], log['true_rewards']),
        'rtg_bias':             rtg_bias(log['sim_rewards'], log['true_rewards']),
        'per_step_reward_rmse': per_step_reward_rmse(log['simulated_ys'], log['true_ys']),
        'per_step_bias':        per_step_bias(log['simulated_ys'], log['true_ys']),
    }


# ---------------------------------------------------------------------------
# Aggregate over all BO iterations — Setup A
# ---------------------------------------------------------------------------

def aggregate_metrics_over_iterations(
    setup_a_npz_path: str,
) -> Dict[str, np.ndarray]:
    """
    Load a Setup A fidelity log and compute per-step metrics for every
    BO iteration.

    Parameters
    ----------
    setup_a_npz_path : path to the *_log.npz file from save_fidelity_logs()

    Returns
    -------
    dict with keys:
        'rtg_rmse'             shape [I]
        'rtg_bias'             shape [I]
        'per_step_reward_rmse' shape [I, T]
        'per_step_bias'        shape [I, T]
        'bo_iterations'        shape [I]

    Example
    -------
    >>> from src.utils.fidelity_metrics import aggregate_metrics_over_iterations
    >>> r = aggregate_metrics_over_iterations("res/dro_fidelity_log.npz")
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(r['bo_iterations'], r['rtg_rmse'])
    >>> plt.xlabel("BO iteration"); plt.ylabel("RTG RMSE"); plt.show()
    """
    data = np.load(setup_a_npz_path)

    sim_ys       = data['simulated_ys']   # [I, R, T]
    true_ys      = data['true_ys']        # [I, R, T]
    sim_rewards  = data['sim_rewards']    # [I, R, T]
    true_rewards = data['true_rewards']   # [I, R, T]
    bo_iters     = data['bo_iterations']  # [I]

    I = sim_ys.shape[0]
    T = sim_ys.shape[2]

    results: Dict[str, np.ndarray] = {
        'rtg_rmse':             np.zeros(I),
        'rtg_bias':             np.zeros(I),
        'per_step_reward_rmse': np.zeros((I, T)),
        'per_step_bias':        np.zeros((I, T)),
        'bo_iterations':        bo_iters,
    }

    for i in range(I):
        iter_log = {
            'simulated_ys': sim_ys[i],
            'true_ys':      true_ys[i],
            'sim_rewards':  sim_rewards[i],
            'true_rewards': true_rewards[i],
        }
        m = compute_all_metrics_setup_a(iter_log)
        results['rtg_rmse'][i]             = m['rtg_rmse']
        results['rtg_bias'][i]             = m['rtg_bias']
        results['per_step_reward_rmse'][i] = m['per_step_reward_rmse']
        results['per_step_bias'][i]        = m['per_step_bias']

    return results


# ---------------------------------------------------------------------------
# Cross-log Wasserstein — requires both Setup A and Setup B npz files
# ---------------------------------------------------------------------------

def compute_wasserstein_over_iterations(
    setup_a_npz_path: str,
    setup_b_npz_path: str,
) -> Dict[str, np.ndarray]:
    """
    Compute Wasserstein-1 distance between simulated and true final regret
    distributions for each BO iteration.

    Requires BOTH Setup A and Setup B log files because:
      - sim_final_regrets  come from Setup A (GP-guided trajectories)
      - true_final_regrets come from Setup B (independently true-guided)

    Parameters
    ----------
    setup_a_npz_path : path to Setup A *_log.npz (from save_fidelity_logs)
    setup_b_npz_path : path to Setup B *_log.npz (from save_true_trajectory_logs)

    Returns
    -------
    dict with keys:
        'wasserstein'  shape [I]  — one distance per BO iteration
        'bo_iterations'shape [I]

    Example
    -------
    >>> from src.utils.fidelity_metrics import compute_wasserstein_over_iterations
    >>> w = compute_wasserstein_over_iterations(
    ...     "res/dro_fidelity_log.npz",
    ...     "res/dro_true_traj_log.npz"
    ... )
    >>> plt.plot(w['bo_iterations'], w['wasserstein'])
    >>> plt.xlabel("BO iteration"); plt.ylabel("Wasserstein distance"); plt.show()
    """
    a = np.load(setup_a_npz_path)
    b = np.load(setup_b_npz_path)

    sim_final  = a['sim_final_regrets']   # [I, R]
    true_final = b['true_final_regrets']  # [I, R]
    bo_iters   = a['bo_iterations']       # [I]

    assert sim_final.shape[0] == true_final.shape[0], (
        f"BO iteration count mismatch: Setup A has {sim_final.shape[0]}, "
        f"Setup B has {true_final.shape[0]}. "
        "Ensure both logs come from the same DRO run."
    )

    I = sim_final.shape[0]
    wasserstein_vals = np.zeros(I)

    for i in range(I):
        wasserstein_vals[i] = wasserstein_final_regret(sim_final[i], true_final[i])

    return {
        'wasserstein':  wasserstein_vals,
        'bo_iterations': bo_iters,
    }
