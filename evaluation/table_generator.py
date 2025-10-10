import os
import pickle
import sys
import numpy as np
import random
import pandas as pd
from controller import Supervisor
import time

# Configuration
SPECIFIC_STAGES = [0, 4, 8, 12, 13]
NUM_EPISODES_PER_INDIVIDUAL = 1
DEBUG_MODE = False

# Environment setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Essential imports
try:
    from environment.simulation_manager import SimulationManager
    from environment.tunnel import TunnelBuilder
    from controllers.reactive_controller import reactive_controller_logic
    from curriculum import _load_and_organize_maps
except ImportError as e:
    print(f"[ERROR] Failed to import modules: {e}")
    sys.exit(1)


class PopulationEvaluator:
    def __init__(self, supervisor):
        self.supervisor = supervisor
        self.sim_mgr = SimulationManager(supervisor)
        self.map_pool = _load_and_organize_maps()

    def evaluate_population(self, population, model_name, model_type):
        """Evaluates all individuals of a population (NE or GA) using 5 maps per stage."""
        results = {stage: {'success': [], 'progress': [], 'velocity': []}
                   for stage in SPECIFIC_STAGES}

        total_individuals = len(population.individuals)
        start_time = time.time()

        for idx, individual in enumerate(population.individuals):
            print(f"\n=== Evaluating {model_name} [{idx + 1}/{total_individuals}] ===")

            for stage in SPECIFIC_STAGES:
                if stage not in self.map_pool or not self.map_pool[stage]:
                    print(f"[WARNING] No maps found for Stage {stage}")
                    continue

                # 5 distinct maps per stage
                maps_to_run = random.sample(self.map_pool[stage],
                                            min(5, len(self.map_pool[stage])))

                for map_params in maps_to_run:
                    # Reset simulation
                    self.supervisor.simulationReset()
                    self.supervisor.step(self.sim_mgr.timestep)
                    self.sim_mgr.tunnel_builder = TunnelBuilder(self.supervisor)

                    # Run episode
                    try:
                        if model_type == 'NE':
                            controller_callable = individual.act
                        else:  # GA
                            distP, angleP = individual.get_genes()
                            controller_callable = lambda scan: self.sim_mgr._process_lidar_for_ga(scan, distP, angleP)

                        episode_result = self.sim_mgr._run_single_episode(controller_callable, stage)
                    except Exception as e:
                        print(f"Error during episode: {e}")
                        continue

                    # Compute metrics
                    success = 1 if episode_result['success'] else 0

                    start_pos = np.array(episode_result.get('start_pos', [0, 0, 0])[:2])
                    end_pos = np.array(episode_result.get('end_pos', [0, 0, 0])[:2])
                    final_pos = np.array(episode_result.get('final_pos', [0, 0, 0])[:2])

                    initial_dist = np.linalg.norm(end_pos - start_pos)
                    final_dist = np.linalg.norm(end_pos - final_pos)

                    if episode_result['success']:
                        progress = 1.0
                    else:
                        progress = max(0, (initial_dist - final_dist) / initial_dist) if initial_dist > 0 else 0

                    total_dist = episode_result.get('total_dist', 0)
                    elapsed_time = episode_result.get('elapsed_time', 1)
                    velocity = total_dist / elapsed_time if elapsed_time > 0 else 0

                    # Store results
                    results[stage]['success'].append(success)
                    results[stage]['progress'].append(progress)
                    results[stage]['velocity'].append(velocity)

                    if DEBUG_MODE:
                        print(f"  Stage {stage}: {'Success' if success else 'Failure'}, "
                              f"Progress: {progress:.1%}, Velocity: {velocity:.3f} m/s")

        elapsed = time.time() - start_time
        print(f"Total time for {total_individuals} individuals: {elapsed:.1f} seconds")
        return {'model': model_name, 'type': model_type, 'results': results}

    def evaluate_ne_population(self, checkpoint_path):
        """Evaluates the entire population of a Neuroevolution model."""
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)

        population = data.get('population')
        if not population or not population.individuals:
            raise ValueError("Population not found in checkpoint")

        return self.evaluate_population(
            population=population,
            model_name=os.path.basename(checkpoint_path),
            model_type='NE'
        )

    def evaluate_ga_population(self, checkpoint_path):
        """Evaluates the entire population of a Genetic Algorithm model."""
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)

        population = data.get('population')
        if not population or not population.individuals:
            raise ValueError("Population not found in checkpoint")

        return self.evaluate_population(
            population=population,
            model_name=os.path.basename(checkpoint_path),
            model_type='GA'
        )

    def evaluate_reactive_model(self, fov_mode):
        """Evaluates a reactive controller (1 run per map, 5 maps per stage)."""
        results = {stage: {'success': [], 'progress': [], 'velocity': []}
                   for stage in SPECIFIC_STAGES}

        def reactive_controller(scan, mode=fov_mode):
            return reactive_controller_logic(scan, fov_mode=mode)

        print(f"\n=== Evaluating REACTIVE_{fov_mode.upper()} ===")
        start_time = time.time()

        for stage in SPECIFIC_STAGES:
            if stage not in self.map_pool or len(self.map_pool[stage]) == 0:
                print(f"[WARNING] No maps found for Stage {stage}")
                continue

            # 5 different maps, 1 run each
            maps_to_run = random.sample(self.map_pool[stage],
                                        min(10, len(self.map_pool[stage])))

            for map_params in maps_to_run:
                self.supervisor.simulationReset()
                self.supervisor.step(self.sim_mgr.timestep)
                self.sim_mgr.tunnel_builder = TunnelBuilder(self.supervisor)

                try:
                    episode_result = self.sim_mgr._run_single_episode(reactive_controller, stage)
                except Exception as e:
                    print(f"Error during episode: {e}")
                    continue

                success = 1 if episode_result['success'] else 0

                start_pos = np.array(episode_result.get('start_pos', [0, 0, 0])[:2])
                end_pos = np.array(episode_result.get('end_pos', [0, 0, 0])[:2])
                final_pos = np.array(episode_result.get('final_pos', [0, 0, 0])[:2])

                initial_dist = np.linalg.norm(end_pos - start_pos)
                final_dist = np.linalg.norm(end_pos - final_pos)

                progress = 1.0 if episode_result['success'] else \
                    max(0, (initial_dist - final_dist) / initial_dist) if initial_dist > 0 else 0

                total_dist = episode_result.get('total_dist', 0)
                elapsed_time = episode_result.get('elapsed_time', 1)
                velocity = total_dist / elapsed_time if elapsed_time > 0 else 0

                results[stage]['success'].append(success)
                results[stage]['progress'].append(progress)
                results[stage]['velocity'].append(velocity)

                if DEBUG_MODE:
                    print(f"  Stage {stage}: {'Success' if success else 'Failure'}, "
                          f"Progress: {progress:.1%}, Velocity: {velocity:.3f} m/s")

        elapsed = time.time() - start_time
        print(f"Total time for REACTIVE_{fov_mode.upper()}: {elapsed:.1f} seconds")
        return {'model': f'REACTIVE_{fov_mode.upper()}_FOV',
                'type': 'REACTIVE',
                'results': results}

    def evaluate_reactive_models(self):
        """Evaluates all reactive controllers."""
        results = []
        for fov_mode in ['full', 'left', 'right']:
            try:
                results.append(self.evaluate_reactive_model(fov_mode))
            except Exception as e:
                print(f"Error evaluating reactive model {fov_mode}: {e}")
        return results


def generate_table_ii(all_results):
    """Generates Table II with consolidated results (averages per stage)."""
    table_data = []

    for result in all_results:
        model_type = result['type']
        model_name = result['model']

        for stage in SPECIFIC_STAGES:
            stage_results = result['results'].get(stage)
            if not stage_results or not stage_results['success']:
                continue

            # Compute statistics across all individuals/runs
            success_rates = stage_results['success']
            success_rate = np.mean(success_rates) * 100

            progresses = stage_results['progress']
            progress_worst = np.min(progresses) * 100
            progress_avg = np.mean(progresses) * 100
            progress_best = np.max(progresses) * 100
            progress_std = np.std(progresses) * 100

            velocities = stage_results['velocity']
            velocity_worst = np.min(velocities)
            velocity_avg = np.mean(velocities)
            velocity_best = np.max(velocities)
            velocity_std = np.std(velocities)

            # Add row to table
            table_data.append({
                'Model': model_type,
                'Stage': stage,
                'Success Rate (%)': f"{success_rate:.1f}",
                'Worst Progress (%)': f"{progress_worst:.1f}",
                'Avg Progress (%)': f"{progress_avg:.1f}",
                'Best Progress (%)': f"{progress_best:.1f}",
                'Std Progress (%)': f"{progress_std:.1f}",
                'Worst Velocity (m/s)': f"{velocity_worst:.3f}",
                'Avg Velocity (m/s)': f"{velocity_avg:.3f}",
                'Best Velocity (m/s)': f"{velocity_best:.3f}",
                'Std Velocity (m/s)': f"{velocity_std:.3f}"
            })

    return pd.DataFrame(table_data)


def main():
    supervisor = Supervisor()
    evaluator = PopulationEvaluator(supervisor)
    all_results = []

    # Evaluate NE population
    ne_path = input("Path to NE checkpoint (.pkl): ").strip()
    if os.path.exists(ne_path):
        try:
            all_results.append(evaluator.evaluate_ne_population(ne_path))
        except Exception as e:
            print(f"Error evaluating NE population: {e}")

    # Evaluate GA population
    ga_path = input("Path to GA checkpoint (.pkl): ").strip()
    if os.path.exists(ga_path):
        try:
            all_results.append(evaluator.evaluate_ga_population(ga_path))
        except Exception as e:
            print(f"Error evaluating GA population: {e}")

    # Evaluate reactive models
    try:
        all_results.extend(evaluator.evaluate_reactive_models())
    except Exception as e:
        print(f"Error evaluating reactive models: {e}")

    # Generate and save table
    if all_results:
        table_ii = generate_table_ii(all_results)
        output_file = "Table_II_Results.csv"
        table_ii.to_csv(output_file, index=False)

        print("\n" + "=" * 80)
        print("TABLE II - CONSOLIDATED RESULTS".center(80))
        print("=" * 80)
        print(table_ii.to_string(index=False))
        print(f"\nTable saved to: {output_file}")
    else:
        print("\nNo results generated. Check errors above.")


if __name__ == '__main__':
    main()
