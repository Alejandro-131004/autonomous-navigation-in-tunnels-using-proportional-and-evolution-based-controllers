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
    success_rate_prev = [d.get('success_rate_prev', 0) * 100 for d in history]
    success_rate_curr = [d.get('success_rate_curr', 0) * 100 for d in history]
    vel_avg = [d.get('avg_linear_vel', 0) for d in history]
    stages = [d['stage'] for d in history]

    # --- Plotting ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 16), sharex=True)
    fig.suptitle(f'Training History - {os.path.basename(checkpoint_path)}', fontsize=16)

    # Plot 1: Fitness
    ax1.plot(generations, fitness_max, label='Max Fitness', color='green', marker='.', linestyle='-')
    ax1.plot(generations, fitness_avg, label='Average Fitness', color='orange', marker='.', linestyle='-')
    ax1.plot(generations, fitness_min, label='Min Fitness', color='red', marker='.', linestyle='--')
    ax1.fill_between(generations, fitness_min, fitness_max, color='green', alpha=0.1)
    ax1.set_ylabel('Fitness', fontsize=12)
    ax1.set_title('Fitness Evolution per Generation', fontsize=14)
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Stage markers on fitness plot
    if stages:
        last_stage = stages[0]
        for i, stage in enumerate(stages):
            if stage != last_stage:
                ax1.axvline(x=generations[i], color='grey', linestyle='--', linewidth=1)
                ax1.text(generations[i], np.max(fitness_max), f' Stage {stage} ',
                         color='blue', rotation=90, verticalalignment='top')
                last_stage = stage

    # Plot 2: Success Rates
    ax2.plot(generations, success_rate_curr, label='Success Rate (Current Stage)', color='dodgerblue',
             marker='o', markersize=4, linestyle='-')
    ax2.plot(generations, success_rate_prev, label='Success Rate (Previous Stages)', color='lightcoral',
             marker='x', markersize=4, linestyle='--')
    ax2.set_ylabel('Success Rate (%)', fontsize=12)
    ax2.set_title('Success Rate per Generation', fontsize=14)
    ax2.set_ylim(0, 105)
    ax2.legend()
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Plot 3: Average Linear Velocity
    ax3.plot(generations, vel_avg, label='Average Linear Velocity', color='purple',
             marker='s', linestyle='-')
    ax3.set_ylabel('Linear Velocity (m/s)', fontsize=12)
    ax3.set_xlabel('Generation', fontsize=12)
    ax3.set_title('Average Linear Speed per Generation', fontsize=14)
    ax3.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax3.legend()

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
