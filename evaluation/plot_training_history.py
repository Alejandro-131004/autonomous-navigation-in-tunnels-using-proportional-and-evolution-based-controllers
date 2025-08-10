import matplotlib
matplotlib.use('TkAgg')  # For compatibility in some environments

import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Imports to support checkpoint loading
try:
    from optimizer.individualNeural import IndividualNeural
    from optimizer.neuralpopulation import NeuralPopulation
    from optimizer.mlpController import MLPController
except ImportError as e:
    print(f"[Warning] Could not import some project classes: {e}")
    print("This might cause errors if the checkpoint file contains these classes.")


def plot_history(checkpoint_path):
    """
    Loads the training history from a checkpoint file and plots graphs
    to visualize the evolution of fitness, success rates, and velocity.
    """
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] File '{checkpoint_path}' not found.")
        return

    try:
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
        history = data.get('history', [])
        if not history:
            print("Checkpoint file does not contain history data to plot.")
            return
    except (pickle.UnpicklingError, EOFError) as e:
        print(f"[ERROR] Failed to read checkpoint file: {e}")
        return

    # Extract history
    generations = [d['generation'] for d in history]
    fitness_min = [d['fitness_min'] for d in history]
    fitness_avg = [d['fitness_avg'] for d in history]
    fitness_max = [d['fitness_max'] for d in history]

    # Handle None values in success rates
    success_rate_prev = [100 * (val if val is not None else 0)
                         for val in (d.get('success_rate_prev') for d in history)]
    success_rate_curr = [100 * (val if val is not None else 0)
                         for val in (d.get('success_rate_curr') for d in history)]

    # Handle None values in velocity
    vel_avg = [val if val is not None else 0
               for val in (d.get('avg_linear_vel') for d in history)]

    stages = [d['stage'] for d in history]

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 11), sharex=True)

    # 1) Fitness
    ax1.plot(generations, fitness_max, label='Max Fitness', color='green', marker='.', linestyle='-')
    ax1.plot(generations, fitness_avg, label='Average Fitness', color='orange', marker='.', linestyle='-')
    ax1.plot(generations, fitness_min, label='Min Fitness', color='red', marker='.', linestyle='--')
    ax1.fill_between(generations, fitness_min, fitness_max, color='green', alpha=0.1)
    ax1.set_ylabel('Fitness', fontsize=12)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.legend()

    # 2) Success Rates
    ax2.plot(generations, success_rate_curr, label='Current Stage', color='dodgerblue',
             marker='o', markersize=4, linestyle='-')
    ax2.plot(generations, success_rate_prev, label='Previous Stages', color='lightcoral',
             marker='x', markersize=4, linestyle='--')
    ax2.set_ylabel('Success Rate (%)', fontsize=12)
    ax2.set_xlabel('Generation', fontsize=12)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.legend()

    # Stage markers on both axes
    # Add S1 explicitly at the start
    if stages[0] == 1:
        ax1.axvline(x=generations[0], color='grey', linestyle='--', linewidth=1)
        ax2.axvline(x=generations[0], color='grey', linestyle='--', linewidth=1)
        ax1.text(generations[0], ax1.get_ylim()[0] - 300, 'S1',
                 color='blue', ha='center', va='top', fontsize=10, fontweight='bold')
        ax2.text(generations[0], -5, 'S1',
                 color='blue', ha='center', va='top', fontsize=10, fontweight='bold')

    last_stage = stages[0]
    for i, stage in enumerate(stages[1:], 1):  # Start from the second element
        if stage != last_stage:
            gen = generations[i]
            # Vertical line on both plots
            ax1.axvline(x=gen, color='grey', linestyle='--', linewidth=1)
            ax2.axvline(x=gen, color='grey', linestyle='--', linewidth=1)
            # Label on fitness plot (top)
            ax1.text(gen, ax1.get_ylim()[0] - 300, f'S{stage}',
                     color='blue', ha='center', va='top', fontsize=10, fontweight='bold')
            # Label on success rate plot (bottom)
            ax2.text(gen, -5, f'S{stage}',
                     color='blue', ha='center', va='top', fontsize=10, fontweight='bold')
            last_stage = stage

    # Adjust y-limits to accommodate labels
    ax1.set_ylim(ax1.get_ylim()[0] - 1000, ax1.get_ylim()[1])
    ax2.set_ylim(-10, 105)

    # Small lateral padding
    for ax in (ax1, ax2):
        ax.margins(x=0.02)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def main():
    try:
        checkpoint_path = input("Enter path to the checkpoint (.pkl) file: > ")
        plot_history(checkpoint_path)
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")


if __name__ == '__main__':
    main()
