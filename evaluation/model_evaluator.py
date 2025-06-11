"""
Universal module to evaluate and compare multiple types of controllers.
"""
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict

from controller import Supervisor
from environment.simulation_manager import SimulationManager
from optimizer.individualNeural import IndividualNeural


def _metric_factory():
    return {'fitness': [], 'success': []}


def _difficulty_factory():
    return defaultdict(_metric_factory)


def evaluate_controllers(
        supervisor: Supervisor,
        controllers_to_test: list,
        map_files: list,
        results_output_dir: str = "evaluation/results"
):
    """
    Evaluates a list of controllers (functions, parameter files, or neural networks).
    """
    os.makedirs(results_output_dir, exist_ok=True)
    sim_mgr = SimulationManager(supervisor)
    all_results = defaultdict(_difficulty_factory)

    maps_by_difficulty = defaultdict(list)
    for f_path in map_files:
        with open(f_path, 'rb') as f:
            maps_by_difficulty[pickle.load(f)['difficulty_level']].append(f_path)

    for controller_info in controllers_to_test:
        controller_name = controller_info['name']
        print(f"\n===== EVALUATING CONTROLLER: {controller_name} =====")

        controller_callable = None
        if controller_info['type'] == 'function':
            controller_callable = controller_info['callable']

        elif controller_info['type'] == 'neural_network':
            try:
                with open(controller_info['path'], 'rb') as f:
                    model = pickle.load(f)
                if isinstance(model, IndividualNeural):
                    controller_callable = model.act
                else:
                    print(f"[ERROR] File '{controller_info['path']}' is not an IndividualNeural instance. Skipping.")
                    continue
            except Exception as e:
                print(f"[ERROR] Failed to load model from '{controller_info['path']}': {e}. Skipping.")
                continue

        elif controller_info['type'] == 'ga_params':
            try:
                with open(controller_info['path'], 'rb') as f:
                    params = pickle.load(f)
                distP, angleP = params['distP'], params['angleP']
                # Create a function that calls sim_mgr controller with loaded parameters
                controller_callable = lambda scan: sim_mgr._process_lidar_for_ga(scan, distP, angleP)
            except Exception as e:
                print(f"[ERROR] Failed to load parameters from '{controller_info['path']}': {e}. Skipping.")
                continue

        if not controller_callable:
            continue

        for difficulty, maps in sorted(maps_by_difficulty.items()):
            print(f"  --- Testing Difficulty {difficulty} ({len(maps)} maps) ---")
            for map_filepath in maps:
                results = sim_mgr._run_single_episode(
                    controller_callable=controller_callable,
                    stage=difficulty,
                    total_stages=len(maps_by_difficulty)
                )
                all_results[controller_name][difficulty]['fitness'].append(results['fitness'])
                all_results[controller_name][difficulty]['success'].append(1 if results['success'] else 0)

    _generate_comparison_report(all_results, results_output_dir)


def _generate_comparison_report(all_results, output_dir):
    """Generates comparative plots for all evaluated controllers."""
    print("\n--- Generating Comparative Evaluation Report ---")

    plt.style.use('seaborn-v0_8-whitegrid')

    # Fitness Plot
    plt.figure(figsize=(12, 8))
    for name, results_by_diff in all_results.items():
        difficulties = sorted(results_by_diff.keys())
        avg_fitness = [np.mean(results_by_diff[d]['fitness']) for d in difficulties]
        plt.plot(difficulties, avg_fitness, marker='o', linestyle='-', label=name)

    plt.title('Average Fitness vs Difficulty Level', fontsize=16)
    plt.xlabel('Difficulty Level', fontsize=12)
    plt.ylabel('Average Fitness', fontsize=12)
    plt.xticks(difficulties)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fitness_comparison.png"))
    plt.close()
    print("Fitness comparison plot saved.")

    # Success Rate Plot
    plt.figure(figsize=(12, 8))
    for name, results_by_diff in all_results.items():
        difficulties = sorted(results_by_diff.keys())
        success_rate = [np.mean(results_by_diff[d]['success']) * 100 for d in difficulties]
        plt.plot(difficulties, success_rate, marker='o', linestyle='-', label=name)

    plt.title('Success Rate vs Difficulty Level', fontsize=16)
    plt.xlabel('Difficulty Level', fontsize=12)
    plt.ylabel('Success Rate (%)', fontsize=12)
    plt.xticks(difficulties)
    plt.ylim(0, 105)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "success_rate_comparison.png"))
    plt.close()
    print("Success rate comparison plot saved.")

    with open(os.path.join(output_dir, "full_evaluation_results.pkl"), 'wb') as f:
        pickle.dump(dict(all_results), f)
    print("Detailed evaluation results saved.")
