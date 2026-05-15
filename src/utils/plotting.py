# src/utils/plotting.py

import numpy as np
import matplotlib.pyplot as plt
import os
import itertools

def plot_comparison(results: dict, function_name: str, total_iterations: int, save_dir: str):
    """
    Plot comparison of the methods based on aggregated results for a single function.

    Args:
        results: Dictionary with aggregated results {method_key: {stats...}}.
        function_name: Name of the function for plotting titles.
        total_iterations: Total number of function evaluations (initial + BO).
        save_dir: Directory where the plot images will be saved.
    """
    method_keys = list(results.keys())
    num_methods = len(method_keys)
    if num_methods == 0:
        print(f"Warning: No methods found in results for plotting {function_name}.")
        return

    os.makedirs(save_dir, exist_ok=True) # Ensure save directory exists

    # Define a color cycle for plots
    colors = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    plot_colors = {key: next(colors) for key in method_keys} # Store colors for consistency

    # --- 1. Plot optimization traces (Mean ± Std Dev) ---
    plt.figure(figsize=(12, 6))

    for key in method_keys:
        method_name = results[key]["name"]
        color = plot_colors[key]

        avg_trace = results[key].get("avg_trace", np.array([np.nan]))
        std_trace = results[key].get("std_trace", np.array([np.nan]))

        if not np.all(np.isnan(avg_trace)):
            # Ensure trace length matches expected total iterations if possible
            current_len = len(avg_trace)
            iterations = np.arange(current_len) # Plot based on actual evaluations done
            plt.plot(iterations, avg_trace, label=f'{method_name} (Mean)', color=color, linewidth=2)
            # Ensure std_trace has same length before plotting fill
            if len(std_trace) == current_len:
                 plt.fill_between(iterations, avg_trace - std_trace, avg_trace + std_trace, alpha=0.2, color=color)
            else:
                 print(f"Warning: Mismatch trace lengths for {method_name} on {function_name}. Skipping std dev fill.")
        else:
             print(f"Warning: No valid trace data to plot for {method_name} on {function_name}")

    plt.xlabel('Function Evaluation')
    plt.ylabel('Best Objective Value Found So Far')
    plt.title(f'Optimization Progress Comparison on {function_name}')
    plt.legend()
    plt.grid(True)
    plt.xlim(left=0)
    plot_filename = os.path.join(save_dir, f"{function_name}_comparison_trace.png")
    plt.savefig(plot_filename)
    print(f"Saved trace plot: {plot_filename}")
    plt.close()

    # --- 2. Plot final performance comparison (Bar Chart) ---
    plt.figure(figsize=(max(8, num_methods * 2), 6))
    bar_width = 0.35
    index = np.arange(num_methods)
    method_names = [results[key]["name"] for key in method_keys]
    avg_values = [results[key].get("avg_best_value", np.nan) for key in method_keys]
    std_values = [results[key].get("std_best_value", np.nan) for key in method_keys]
    bar_colors = [plot_colors[key] for key in method_keys]

    plt.bar(index, avg_values, bar_width, yerr=std_values,
            color=bar_colors, alpha=0.8, capsize=5) # Removed label=method_names

    plt.ylabel('Average Best Objective Value (Final)')
    plt.title(f'Final Performance Comparison on {function_name}')
    plt.xticks(index, method_names, rotation=15, ha="right")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plot_filename = os.path.join(save_dir, f"{function_name}_comparison_performance.png")
    plt.savefig(plot_filename)
    print(f"Saved performance plot: {plot_filename}")
    plt.close()

    # --- 3. Plot computation time comparison (Bar Chart) ---
    plt.figure(figsize=(max(8, num_methods * 2), 6))
    avg_times = [results[key].get("avg_time", np.nan) for key in method_keys]

    plt.bar(index, avg_times, bar_width, color=bar_colors, alpha=0.8) # Removed label=method_names

    plt.ylabel('Average Computation Time (seconds)')
    plt.title(f'Computation Time Comparison on {function_name}')
    plt.xticks(index, method_names, rotation=15, ha="right")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plot_filename = os.path.join(save_dir, f"{function_name}_comparison_time.png")
    plt.savefig(plot_filename)
    print(f"Saved time plot: {plot_filename}")
    plt.close()


def plot_overall_summary(all_results: dict, methods: dict, save_dir: str):
    """
    Create summary plots comparing methods across all functions.

    Args:
        all_results: Dictionary keyed by function name, containing results.
        methods: Dictionary defining the methods compared (for names/colors).
        save_dir: Directory where the summary plot images will be saved.
    """
    function_names = list(all_results.keys())
    method_keys = list(methods.keys()) # Use methods dict passed in, which contains only active methods
    num_functions = len(function_names)
    num_methods = len(method_keys)

    if num_functions == 0 or num_methods == 0:
        print("Warning: No results to plot in overall summary.")
        return

    os.makedirs(save_dir, exist_ok=True) # Ensure save directory exists

    # Use consistent colors
    colors = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    method_colors = {key: next(colors) for key in method_keys}

    # --- Overall Performance Plot ---
    plt.figure(figsize=(max(10, num_functions * 2.5), 7)) # Adjust size
    bar_width = 0.8 / num_methods # Adjust bar width based on number of methods
    index = np.arange(num_functions)

    for i, key in enumerate(method_keys):
        method_name = methods[key]["name"]
        perf_values = [all_results[func_name].get(key, {}).get("avg_best_value", np.nan) for func_name in function_names]

        plt.bar(index + i * bar_width, perf_values, bar_width,
                label=method_name, color=method_colors[key], alpha=0.8)

    plt.xlabel('Test Function')
    plt.ylabel('Average Best Objective Value (Final)')
    plt.title('Overall Performance Comparison Across Test Functions')
    plt.xticks(index + bar_width * (num_methods - 1) / 2, function_names, rotation=15, ha="right")
    plt.legend(title="Methods", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, axis='y')
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
    plot_filename = os.path.join(save_dir, "overall_comparison_performance.png")
    plt.savefig(plot_filename)
    print(f"Saved overall performance plot: {plot_filename}")
    plt.close()

    # --- Overall Time Comparison Plot ---
    plt.figure(figsize=(max(10, num_functions * 2.5), 7))

    for i, key in enumerate(method_keys):
        method_name = methods[key]["name"]
        time_values = [all_results[func_name].get(key, {}).get("avg_time", np.nan) for func_name in function_names]

        plt.bar(index + i * bar_width, time_values, bar_width,
                label=method_name, color=method_colors[key], alpha=0.8)

    plt.xlabel('Test Function')
    plt.ylabel('Average Computation Time (seconds)')
    plt.title('Overall Computation Time Comparison Across Test Functions')
    plt.xticks(index + bar_width * (num_methods - 1) / 2, function_names, rotation=15, ha="right")
    plt.legend(title="Methods", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, axis='y')
    plt.yscale('log') # Use log scale if times vary greatly
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout
    plot_filename = os.path.join(save_dir, "overall_comparison_time.png")
    plt.savefig(plot_filename)
    print(f"Saved overall time plot: {plot_filename}")
    plt.close()