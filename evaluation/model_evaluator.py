# model_evaluator.py
"""
Universal module to evaluate and compare multiple types of controllers.
Handles loading models directly or extracting them from checkpoint files.
"""
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict

from controller import Supervisor
from environment.simulation_manager import SimulationManager
from optimizer.individualNeural import IndividualNeural
from optimizer.individual import Individual  # Import the class for the classic GA


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
                    loaded_object = pickle.load(f)

                model_to_test = None
                # Check 1: Is it a dictionary (like a checkpoint file)?
                if isinstance(loaded_object, dict) and 'best_individual' in loaded_object:
                    model_to_test = loaded_object['best_individual']
                    print(f"  [INFO] Checkpoint file detected. Extracted 'best_individual'.")
                # Check 2: Is it a direct IndividualNeural instance?
                elif isinstance(loaded_object, IndividualNeural):
                    model_to_test = loaded_object

                # If we found a valid model, get its callable 'act' method
                if isinstance(model_to_test, IndividualNeural):
                    controller_callable = model_to_test.act
                else:
                    print(
                        f"[ERROR] File '{controller_info['path']}' does not contain a valid IndividualNeural instance. Skipping.")
                    continue
            except Exception as e:
                print(f"[ERROR] Failed to load model from '{controller_info['path']}': {e}. Skipping.")
                continue

        elif controller_info['type'] == 'ga_params':
            try:
                with open(controller_info['path'], 'rb') as f:
                    # The loaded object is an 'Individual' instance from the classic GA
                    individual_obj = pickle.load(f)

                # Check if it's the correct type before accessing attributes
                if isinstance(individual_obj, Individual):
                    distP = individual_obj.distP
                    angleP = individual_obj.angleP
                    # Create a function that calls the sim_mgr controller with the loaded parameters
                    controller_callable = lambda scan: sim_mgr._process_lidar_for_ga(scan, distP, angleP)
                else:
                    print(
                        f"[ERROR] File '{controller_info['path']}' is not a valid 'Individual' instance for GA params. Skipping.")
                    continue
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
    difficulties_for_plot = sorted(list(all_results.values())[0].keys()) if all_results else []

    # Fitness Plot
    plt.figure(figsize=(12, 8))
    for name, results_by_diff in all_results.items():
        difficulties = sorted(results_by_diff.keys())
        avg_fitness = [np.mean(results_by_diff[d]['fitness']) for d in difficulties]
        plt.plot(difficulties, avg_fitness, marker='o', linestyle='-', label=name)

    plt.title('Average Fitness vs. Difficulty Comparison', fontsize=16)
    plt.xlabel('Difficulty Level', fontsize=12)
    plt.ylabel('Average Fitness', fontsize=12)
    if difficulties_for_plot:
        plt.xticks(difficulties_for_plot)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_fitness.png"))
    plt.close()
    print("Fitness comparison plot saved.")

    # Success Rate Plot
    plt.figure(figsize=(12, 8))
    for name, results_by_diff in all_results.items():
        difficulties = sorted(results_by_diff.keys())
        success_rate = [np.mean(results_by_diff[d]['success']) * 100 for d in difficulties]
        plt.plot(difficulties, success_rate, marker='o', linestyle='-', label=name)

    plt.title('Success Rate vs. Difficulty Comparison', fontsize=16)
    plt.xlabel('Difficulty Level', fontsize=12)
    plt.ylabel('Success Rate (%)', fontsize=12)
    if difficulties_for_plot:
        plt.xticks(difficulties_for_plot)
    plt.ylim(0, 105)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_success_rate.png"))
    plt.close()
    print("Success rate comparison plot saved.")

    with open(os.path.join(output_dir, "full_evaluation_results.pkl"), 'wb') as f:
        pickle.dump(dict(all_results), f)
    print("Detailed evaluation results saved.")
