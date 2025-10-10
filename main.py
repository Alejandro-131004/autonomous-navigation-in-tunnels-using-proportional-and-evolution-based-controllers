"""
Main script to run the training pipeline.
"""
import os
import sys
import numpy as np  # Required for np.nan_to_num in reactive_controller_logic
import random  # Required for random.sample

# --- Path Fix for Imports ---
# sys.path adjustments (already present)
sys.path.insert(0, '/Applications/Webots.app/Contents/lib/controller/python')  # Mila
from controller import Supervisor

from curriculum import run_unified_curriculum, _load_and_organize_maps  # Imports _load_and_organize_maps
from environment.configuration import STAGE_DEFINITIONS, get_stage_parameters, generate_intermediate_stage, \
    MAX_DIFFICULTY_STAGE
from environment.simulation_manager import SimulationManager  # Imports SimulationManager
from evaluation.map_generator import generate_maps  # Imports generate_maps
from controllers.reactive_controller import reactive_controller_logic
from controllers.utils import cmd_vel  # Ensures cmd_vel is available for reactive mode

# Define checkpoint file paths
NE_CHECKPOINT_FILE = "saved_models/ne_checkpoint.pkl"
GA_CHECKPOINT_FILE = "saved_models/ga_checkpoint.pkl"


def main():
    """
    Main function to execute the training pipeline.
    """
    # Base configuration and specific configurations for each mode
    base_config = {
        'elitism': 2,
        'threshold_prev': 0.7,
        'threshold_curr': 0.7,
    }

    ne_config = {
        'mode': 'NE',
        'pop_size': 30,
        'hidden_size': 16,
        'mutation_rate': 0.15,
        'checkpoint_file': NE_CHECKPOINT_FILE,
    }

    ga_config = {
        'mode': 'GA',
        'pop_size': 30,
        'mutation_rate': 0.15,
        'checkpoint_file': GA_CHECKPOINT_FILE,
    }

    final_config = {}

    # --- MODE SELECTION AND CHECKPOINT LOGIC ---

    # 1. User selects the training mode
    while True:
        mode_choice = input(
            "Select training mode:\n"
            "  1: Neuroevolution (Neural Network)\n"
            "  2: Genetic Algorithm (Reactive Parameters)\n"
            "  3: Reactive Controller (Rule-Based)\n"
            "Enter your choice (1, 2, or 3): "
        ).strip()
        if mode_choice == '1':
            final_config.update(base_config)
            final_config.update(ne_config)
            selected_mode = 'NE'
            break
        elif mode_choice == '2':
            final_config.update(base_config)
            final_config.update(ga_config)
            selected_mode = 'GA'
            break
        elif mode_choice == '3':
            selected_mode = 'REACTIVE'
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

    # 2. User selects debug mode
    while True:
        debug_choice = input(
            "\nSelect debug mode:\n"
            "  1: Normal (generation summary)\n"
            "  2: Debug (detailed information and warnings)\n"
            "Enter your choice (1 or 2): "
        ).strip()
        if debug_choice == '1':
            os.environ['ROBOT_DEBUG_MODE'] = '0'
            break
        elif debug_choice == '2':
            os.environ['ROBOT_DEBUG_MODE'] = '1'
            print("\n--- DEBUG MODE ENABLED ---")
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")

    if selected_mode == 'REACTIVE':
        # --- REACTIVE CONTROLLER EXECUTION ---
        print("\n--- Starting Reactive Controller Evaluation ---")

        # Select FOV mode
        while True:
            fov_choice = input(
                "Select FOV mode for the Reactive Controller:\n"
                "  1: Full FOV\n"
                "  2: Left FOV\n"
                "  3: Right FOV\n"
                "Enter your choice (1, 2, or 3): "
            ).strip()
            if fov_choice == '1':
                selected_fov = 'full'
                fov_name = 'Full FOV'
                break
            elif fov_choice == '2':
                selected_fov = 'left'
                fov_name = 'Left FOV'
                break
            elif fov_choice == '3':
                selected_fov = 'right'
                fov_name = 'Right FOV'
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")

        print(f"Reactive Controller selected with: {fov_name}")

        # Pipeline settings for Reactive Evaluation
        MAPS_OUTPUT_DIR = "evaluation/maps"
        NUM_MAPS_PER_DIFFICULTY = 3  # Set to 3 maps per difficulty level
        TOTAL_DIFFICULTY_STAGES = 15  # Set to 15 difficulty stages

        # Ensure maps exist
        print(
            f"Generating {NUM_MAPS_PER_DIFFICULTY} maps for each of the {TOTAL_DIFFICULTY_STAGES + 1} stages, if not present...")
        generate_maps(
            maps_output_dir=MAPS_OUTPUT_DIR,
            num_maps_per_difficulty=NUM_MAPS_PER_DIFFICULTY,
            total_difficulty_stages=TOTAL_DIFFICULTY_STAGES
        )
        map_pool = _load_and_organize_maps(maps_dir=MAPS_OUTPUT_DIR, num_maps_per_diff=NUM_MAPS_PER_DIFFICULTY)

        sup = Supervisor()
        sim_mgr = SimulationManager(supervisor=sup)

        print(f"\n--- Running Reactive Controller ({fov_name}) on {TOTAL_DIFFICULTY_STAGES + 1} stages ---")

        total_runs = 0
        total_successes = 0

        for difficulty_level in range(TOTAL_DIFFICULTY_STAGES + 1):
            if difficulty_level not in map_pool or not map_pool[difficulty_level]:
                print(f"  [INFO] No map found for Difficulty Stage {difficulty_level}. Skipping.")
                continue

            maps_for_stage = random.sample(map_pool[difficulty_level],
                                           min(NUM_MAPS_PER_DIFFICULTY, len(map_pool[difficulty_level])))
            print(f"\n  Evaluating Difficulty Stage {difficulty_level} with {len(maps_for_stage)} maps...")

            stage_successes = 0
            for i, map_params in enumerate(maps_for_stage):
                print(f"    Running Map {i + 1}/{len(maps_for_stage)} (Difficulty {difficulty_level})...")
                # controller_callable now includes the fov_mode
                results = sim_mgr._run_single_episode(
                    controller_callable=lambda scan: reactive_controller_logic(scan, fov_mode=selected_fov),
                    stage=difficulty_level  # Pass difficulty level directly
                )

                if results['success']:
                    stage_successes += 1
                    total_successes += 1
                    print(
                        f"      Map {i + 1} SUCCESS! Fitness: {results['fitness']:.2f}, Time: {results['elapsed_time']:.2f}s")
                else:
                    print(
                        f"      Map {i + 1} FAILED. Fitness: {results['fitness']:.2f}, Collision: {results['collided']}, Timeout: {results['timeout']}, No Movement: {results['no_movement_timeout']}")
                total_runs += 1

            print(f"  Stage {difficulty_level} Summary: {stage_successes}/{len(maps_for_stage)} maps successful.")

        print(f"\n--- Reactive Controller Evaluation Completed ---")
        print(f"Total Maps Run: {total_runs}")
        print(f"Total Successes: {total_successes}")
        print(f"Overall Success Rate: {total_successes / total_runs:.2%}" if total_runs > 0 else "N/A")
        return  # Exit main after reactive mode

    # Execute these lines ONLY if not in reactive mode
    training_mode = final_config['mode']
    checkpoint_file = final_config['checkpoint_file']

    # 3. Checkpoint handling (unchanged)
    resume_training = False
    if os.path.exists(checkpoint_file):
        while True:
            resume_choice = input(
                f"\nA checkpoint for '{training_mode}' was found.\n"
                f"Do you want to resume training (y) or start from scratch (n)? [y/n]: "
            ).lower().strip()
            if resume_choice == 'y':
                resume_training = True
                print(f"Resuming previous training for {training_mode}...")
                break
            elif resume_choice == 'n':
                try:
                    os.remove(checkpoint_file)
                    print("Checkpoint removed. Starting a new session.")
                except OSError as e:
                    print(f"Error removing checkpoint: {e}")
                resume_training = False
                break
            else:
                print("Invalid option. Please enter 'y' or 'n'.")
    else:
        print(f"No checkpoint found for '{training_mode}'. Starting a new session.")

    final_config['resume_training'] = resume_training

    # --- END OF MODE SELECTION AND CHECKPOINT LOGIC ---
    print(f"Thresholds: previous = {final_config['threshold_prev']}, current = {final_config['threshold_curr']}")

    # Initialize supervisor and run unified curriculum
    sup = Supervisor()
    print(f"\n--- Starting {training_mode} Training ---")

    best_model = run_unified_curriculum(
        supervisor=sup,
        config=final_config
    )

    # Final message
    if best_model:
        print(f"\nTraining completed. Best {training_mode} model saved.")
    else:
        print("\nTraining completed. No final model identified.")


if __name__ == "__main__":
    main()
