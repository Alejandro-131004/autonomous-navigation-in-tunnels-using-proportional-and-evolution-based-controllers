import os
import pickle
import numpy as np
import random
from collections import defaultdict

from environment.configuration import get_stage_parameters
from environment.simulation_manager import SimulationManager
from evaluation.map_generator import generate_maps

from optimizer.neuralpopulation import NeuralPopulation
from optimizer.population import Population


def _load_and_organize_maps(maps_dir="evaluation/maps", num_maps_per_diff=100):
    """
    Generates 2000 maps (100 for each of the 20 base stages) if they do not exist and loads them.
    """
    from environment.configuration import MAX_DIFFICULTY_STAGE as total_stages_for_gen
    if not os.path.exists(maps_dir) or not os.listdir(maps_dir):
        print(
            f"Map directory '{maps_dir}' not found or empty. Generating {num_maps_per_diff * total_stages_for_gen} new maps..."
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

def run_unified_curriculum(supervisor, config: dict):
    from environment.configuration import STAGE_DEFINITIONS, get_stage_parameters, generate_intermediate_stage

    mode = config['mode']
    sim_mgr = SimulationManager(supervisor)
    map_pool = _load_and_organize_maps(num_maps_per_diff=100)
    checkpoint_file = config['checkpoint_file']

    def _save_checkpoint(data):
        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(data, f)

    def _load_checkpoint():
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'rb') as f:
                return pickle.load(f)
        return None

    # Initialization
    population = None
    best_overall_individual = None
    start_stage = 1
    history = []

    if config['resume_training']:
        data = _load_checkpoint()
        if data:
            population = data.get('population')
            best_overall_individual = data.get('best_individual')
            start_stage = data.get('stage', 1)
            history = data.get('history', [])

    if population is None:
        print("Initializing new population...")
        if mode == 'NE':
            input_size = len(np.nan_to_num(sim_mgr.lidar.getRangeImage(), nan=0.0))
            output_size = 2
            population = NeuralPopulation(
                pop_size=config['pop_size'],
                input_size=input_size,
                hidden_size=config['hidden_size'],
                output_size=output_size,
                mutation_rate=config['mutation_rate'],
                elitism=config['elitism']
            )
        else:
            population = Population(
                pop_size=config['pop_size'],
                mutation_rate=config['mutation_rate'],
                elitism=config['elitism']
            )

    current_stage = start_stage
    threshold_prev = config.get('threshold_prev', 0.7)
    threshold_curr = config.get('threshold_curr', 0.7)

    sub_index = 0
    attempts_without_progress = 0
    custom_stage_params = None

    try:
        while True:
            print(f"\n\n{'=' * 20} STARTING DIFFICULTY STAGE {current_stage} {'=' * 20}")
            attempts_in_stage = 0

            while True:
                attempts_in_stage += 1
                generation_id = len(history) + 1
                print(f"\n--- Generation {generation_id} (Stage {current_stage}, Attempt {attempts_in_stage}) ---")

                runs_prev = 5
                runs_curr = 5

                # PREVIOUS STAGE MAPS
                available_prev_stages = [s for s in map_pool if s < current_stage]
                maps_prev = []
                if available_prev_stages:
                    for stage in random.sample(available_prev_stages, min(runs_prev, len(available_prev_stages))):
                        maps_prev.append(random.choice(map_pool[stage]))

                # CURRENT STAGE MAPS
                maps_curr = random.sample(
                    map_pool.get(current_stage, map_pool.get(20, [])),
                    k=min(runs_curr, len(map_pool.get(current_stage, map_pool.get(20, []))))
                )

                if os.environ.get('ROBOT_DEBUG_MODE') == '1':
                    print(f"[DEBUG] Maps used: Previous: {[m['difficulty_level'] for m in maps_prev]}, Current: {[m['difficulty_level'] for m in maps_curr]}")

                # --- EVALUATION ---
                avg_succ_prev = population.evaluate(sim_mgr, maps_prev) if maps_prev else 0
                avg_succ_curr = population.evaluate(sim_mgr, maps_curr) if maps_curr else 0
                total_runs_prev = len(maps_prev)
                total_runs_curr = len(maps_curr)

                rate_prev_real = avg_succ_prev / total_runs_prev if total_runs_prev > 0 else 0
                rate_curr = avg_succ_curr / total_runs_curr if total_runs_curr > 0 else 0

                # Lógica especial para Stage 1
                if current_stage == 0 and generation_id > 1:
                    rate_prev = history[-1]['success_rate_curr']
                elif current_stage == 1:
                    rate_prev = None
                else:
                    rate_prev = rate_prev_real

                fitness_values = [ind.fitness for ind in population.individuals]
                fitness_min = min(fitness_values)
                fitness_max = max(fitness_values)
                fitness_avg = np.mean(fitness_values)

                generation_stats = {
                    'stage': current_stage, 'generation': generation_id,
                    'fitness_min': fitness_min, 'fitness_avg': fitness_avg, 'fitness_max': fitness_max,
                    'success_rate_prev': rate_prev if rate_prev is not None else 0,
                    'success_rate_curr': rate_curr
                }
                history.append(generation_stats)

                # Impressão de resultados
                print("-" * 50)
                print(f"  FITNESS -> Min: {fitness_min:.2f} | Avg: {fitness_avg:.2f} | Max: {fitness_max:.2f}")
                if rate_prev is not None:
                    print(f"  SUCCESS -> Prev: {rate_prev:.2%} | Curr: {rate_curr:.2%}")
                else:
                    print(f"  SUCCESS -> Curr: {rate_curr:.2%}")
                print("-" * 50)

                # Best individual
                gen_best = population.get_best_individual()
                if gen_best and (best_overall_individual is None or gen_best.fitness > best_overall_individual.fitness):
                    best_overall_individual = gen_best
                    prefix = "ne_best" if mode == 'NE' else "ga_best"
                    sim_mgr.save_model(gen_best, filename=f"{prefix}_stage_{current_stage}_gen_{generation_id}.pkl")

                _save_checkpoint({
                    'population': population,
                    'best_individual': best_overall_individual,
                    'stage': current_stage,
                    'history': history
                })

                if (rate_prev is None or rate_prev >= threshold_prev) and rate_curr >= threshold_curr:
                    print(f"[PROGRESS] Thresholds passed: prev={rate_prev if rate_prev is not None else 'N/A'}, curr={rate_curr:.2%}")
                    current_stage += 1
                    sub_index = 0
                    attempts_without_progress = 0
                    custom_stage_params = None
                    break
                else:
                    attempts_without_progress += 1
                    print(f"[REPEAT] Thresholds not reached. Attempt {attempts_without_progress}...")

                    if attempts_without_progress >= 50: 
                        sub_index += 1
                        base_params = STAGE_DEFINITIONS[current_stage]
                        custom_stage_params = generate_intermediate_stage(base_params, sub_index=sub_index)
                        print(f" Creating intermediate sub-stage {sub_index} -> {custom_stage_params}")
                        print(f"\n 50 attempts without reaching thresholds. Switching to intermediate sub-stage {sub_index}...")
                        print(f"🔧 Adjusted parameters: {custom_stage_params}")
                        attempts_without_progress = 0

                    population.create_next_generation()

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Saving last state...")
    finally:
        _save_checkpoint({
            'population': population,
            'best_individual': best_overall_individual,
            'stage': current_stage,
            'history': history
        })
        print("Training session ended.")

    return best_overall_individual
