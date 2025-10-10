import os
import pickle
import numpy as np
import random
from collections import defaultdict

from environment.configuration import get_stage_parameters
from environment.simulation_manager import SimulationManager
from evaluation.map_generator import generate_maps

# Required imports so that pickle can load objects from the checkpoint
from optimizer.neuralpopulation import NeuralPopulation
from optimizer.population import Population
from optimizer.individualNeural import IndividualNeural
from optimizer.individual import Individual
from optimizer.mlpController import MLPController


def _load_and_organize_maps(maps_dir="evaluation/maps", num_maps_per_diff=100):
    """
    Generates maps if the directory does not exist and organizes them by difficulty.
    """
    from environment.configuration import MAX_DIFFICULTY_STAGE as total_stages_for_gen
    if not os.path.exists(maps_dir) or not os.listdir(maps_dir):
        print(
            f"Map directory '{maps_dir}' not found or empty. Generating {num_maps_per_diff * (total_stages_for_gen + 1)} new maps..."
        )
        generate_maps(maps_output_dir=maps_dir,
                      num_maps_per_difficulty=num_maps_per_diff,
                      total_difficulty_stages=total_stages_for_gen)

    map_pool = defaultdict(list)
    for filename in os.listdir(maps_dir):
        if filename.endswith(".pkl"):
            filepath = os.path.join(maps_dir, filename)
            try:
                with open(filepath, 'rb') as f:
                    map_params = pickle.load(f)
                    difficulty = map_params['difficulty_level']
                    map_pool[difficulty].append(map_params)
            except Exception as e:
                print(f"[WARNING] Could not load map {filepath}: {e}")

    print(f"Maps loaded. {sum(len(v) for v in map_pool.values())} maps across {len(map_pool)} difficulty levels.")
    return dict(map_pool)


def _re_evaluate_past_stages(population, sim_mgr, map_pool, up_to_stage, threshold):
    """
    Re-evaluates the current population's performance across all previous stages.
    Returns a list of stages that did not meet the success threshold.
    """
    print("\n" + "=" * 20 + " STARTING RE-EVALUATION OF PREVIOUS STAGES " + "=" * 20)
    retraining_queue = []

    for stage in range(up_to_stage):
        if stage not in map_pool:
            continue

        print(f"Re-evaluating Stage {stage}...")
        num_maps_to_run = 10  # Use a reasonable number of maps for re-evaluation
        maps_to_run = random.sample(map_pool[stage], min(num_maps_to_run, len(map_pool[stage])))

        if not maps_to_run:
            continue

        avg_success = population.evaluate(sim_mgr, maps_to_run)
        success_rate = avg_success

        result = "OK"
        if success_rate < threshold:
            result = "FAILED"
            retraining_queue.append(stage)

        print(f"--> Stage {stage} Result: Success Rate = {success_rate:.2%} ({result})")

    if not retraining_queue:
        print("\nRe-evaluation completed. All previous stages meet the required thresholds.")
    else:
        print(f"\nRe-evaluation completed. Retraining queue: {retraining_queue}")

    return retraining_queue


def run_unified_curriculum(supervisor, config: dict):
    from environment.configuration import STAGE_DEFINITIONS, generate_intermediate_stage, MAX_DIFFICULTY_STAGE

    mode = config['mode']
    sim_mgr = SimulationManager(supervisor)
    map_pool = _load_and_organize_maps(num_maps_per_diff=100)
    checkpoint_file = config['checkpoint_file']

    retraining_queue = []

    def _save_checkpoint(data):
        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(data, f)

    def _load_checkpoint():
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'rb') as f:
                    return pickle.load(f)
            except (pickle.UnpicklingError, EOFError) as e:
                print(f"[ERROR] Failed to read checkpoint file: {e}. Starting new training session.")
        return None

    population = None
    best_overall_individual = None
    start_stage = 0
    history = []

    if config.get('resume_training', False):
        data = _load_checkpoint()
        if data:
            population = data.get('population')
            best_overall_individual = data.get('best_individual')
            saved_stage = data.get('stage', 0)
            history = data.get('history', [])
            print(f"\nCheckpoint loaded. Last session ended at Stage {saved_stage}.")

            # --- NEW REFRESH/RETRAIN LOGIC ---
            while True:
                refresh_choice = input(
                    "Do you want to perform a 'Refresh Training' to re-evaluate previous stages? [y/n]: "
                ).lower().strip()
                if refresh_choice in ['y', 'n']:
                    break
                print("Invalid option.")

            if refresh_choice == 'y':
                # Re-evaluate performance and build retraining queue
                retraining_queue = _re_evaluate_past_stages(population, sim_mgr, map_pool, up_to_stage=saved_stage,
                                                            threshold=config['threshold_prev'])

            # Ask about continuation stage
            while True:
                override_choice = input(
                    f"Press 'c' to continue from Stage {saved_stage}, or 's' to select a different start stage: [c/s] "
                ).lower().strip()
                if override_choice == 'c':
                    start_stage = saved_stage
                    break
                elif override_choice == 's':
                    # Select a different start stage
                    while True:
                        try:
                            new_stage_input = input(
                                f"Enter the stage number to restart from (0-{MAX_DIFFICULTY_STAGE}): ")
                            new_stage = int(new_stage_input)
                            if 0 <= new_stage <= MAX_DIFFICULTY_STAGE:
                                start_stage = new_stage
                                history = [entry for entry in history if entry['stage'] < start_stage]
                                retraining_queue = [stage for stage in retraining_queue if stage < start_stage]
                                print(f"Updated retraining queue: {retraining_queue}")
                                break
                            else:
                                print(f"Invalid stage. Please enter a number between 0 and {MAX_DIFFICULTY_STAGE}.")
                        except ValueError:
                            print("Invalid input. Please enter a number.")
                    break
                else:
                    print("Invalid option. Please enter 'c' or 's'.")
        else:
            config['resume_training'] = False

    if population is None:
        print("Starting a new population...")
        if mode == 'NE':
            input_size = len(np.nan_to_num(sim_mgr.lidar.getRangeImage(), nan=0.0))
            population = NeuralPopulation(pop_size=config['pop_size'], input_size=input_size,
                                          hidden_size=config['hidden_size'], output_size=2,
                                          mutation_rate=config['mutation_rate'], elitism=config['elitism'])
        else:
            population = Population(pop_size=config['pop_size'], mutation_rate=config['mutation_rate'],
                                    elitism=config['elitism'])
        start_stage = 0
        history = []

    current_stage = start_stage
    threshold_prev = config.get('threshold_prev', 0.7)
    threshold_curr = config.get('threshold_curr', 0.7)

    try:
        while True:
            # --- MAIN TRAINING LOOP (MODIFIED) ---
            is_retraining = False
            if retraining_queue:
                stage_to_train = retraining_queue[0]
                is_retraining = True
                print(
                    f"\n\n{'=' * 20} STARTING RETRAINING OF STAGE {stage_to_train} ({len(retraining_queue)} in queue) {'=' * 20}")
            elif current_stage <= MAX_DIFFICULTY_STAGE:
                stage_to_train = current_stage
                print(f"\n\n{'=' * 20} STARTING DIFFICULTY STAGE {stage_to_train} {'=' * 20}")
            else:
                print("\nTraining and retraining successfully completed!")
                break

            sub_index = 0
            attempts_without_progress = 0

            # Training loop for the selected stage (normal or retraining)
            while True:
                generation_id = len(history) + 1
                print(f"\n--- Generation {generation_id} (Training on Stage {stage_to_train}) ---")

                # Map selection
                runs_prev = 5
                runs_curr = 5
                available_prev_stages = [s for s in map_pool if s < stage_to_train]
                maps_prev = [random.choice(map_pool[stage]) for stage in random.sample(available_prev_stages,
                                                                                       min(runs_prev,
                                                                                           len(available_prev_stages)))] if available_prev_stages else []
                maps_curr = random.sample(map_pool.get(stage_to_train, []),
                                          k=min(runs_curr, len(map_pool.get(stage_to_train, []))))

                if not maps_curr:
                    print(f"[WARNING] No maps found for Stage {stage_to_train}. Skipping this stage.")
                    if is_retraining:
                        retraining_queue.pop(0)
                    else:
                        current_stage += 1
                    break

                # Evaluation
                avg_succ_prev = population.evaluate(sim_mgr, maps_prev) if maps_prev else None
                avg_succ_curr = population.evaluate(sim_mgr, maps_curr)
                rate_prev = avg_succ_prev
                rate_curr = avg_succ_curr

                # Save history and stats
                fitness_values = [ind.fitness for ind in population.individuals if ind.fitness is not None]
                generation_stats = {
                    'stage': stage_to_train, 'generation': generation_id,
                    'fitness_min': min(fitness_values) if fitness_values else 0,
                    'fitness_avg': np.mean(fitness_values) if fitness_values else 0,
                    'fitness_max': max(fitness_values) if fitness_values else 0,
                    'success_rate_prev': rate_prev, 'success_rate_curr': rate_curr
                }
                history.append(generation_stats)

                # Print stats
                print("-" * 50)
                print(
                    f"  FITNESS -> Min: {generation_stats['fitness_min']:.2f} | Avg: {generation_stats['fitness_avg']:.2f} | Max: {generation_stats['fitness_max']:.2f}")
                prev_rate_str = f"{rate_prev:.2%}" if rate_prev is not None else "N/A"
                print(f"  SUCCESS -> Prev: {prev_rate_str} | Curr: {rate_curr:.2%}")
                print("-" * 50)

                gen_best = population.get_best_individual()
                if gen_best and (best_overall_individual is None or gen_best.fitness > best_overall_individual.fitness):
                    best_overall_individual = gen_best
                _save_checkpoint(
                    {'population': population, 'best_individual': best_overall_individual, 'stage': current_stage,
                     'history': history})

                # Check if the current stage (normal or retraining) is complete
                if (rate_prev is None or rate_prev >= threshold_prev) and rate_curr >= threshold_curr:
                    if is_retraining:
                        print(f"[PROGRESS] Retraining of Stage {stage_to_train} successfully completed.")
                        retraining_queue.pop(0)
                    else:
                        print(f"[PROGRESS] Thresholds reached. Moving to next stage.")
                        current_stage += 1
                    break
                else:
                    # Handling stagnation (create intermediate sub-stages)
                    attempts_without_progress += 1
                    if attempts_without_progress >= 50:
                        sub_index += 1
                        base_params = STAGE_DEFINITIONS[stage_to_train]
                        custom_stage_params = generate_intermediate_stage(base_params, sub_index=sub_index)
                        print(f"\nAfter 50 attempts, creating intermediate sub-stage...")
                        # Evaluation now uses sub-stage parameters
                    population.create_next_generation()

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Saving last state...")
    finally:
        _save_checkpoint({'population': population, 'best_individual': best_overall_individual, 'stage': current_stage,
                          'history': history})
        print("Training session finished.")

    return best_overall_individual
