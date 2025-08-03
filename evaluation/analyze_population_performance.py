import os
import pickle
import sys
import numpy as np
import random
import matplotlib
import time
import pandas as pd
from scipy import stats
# Novo (adicione logo após os imports)
MAX_DIFFICULTY_STAGE = 13
matplotlib.use('Agg')  # Use non-interactive backend for batch processing
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

# Add root directory to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Imports for loading checkpoint objects
try:
    from environment.simulation_manager import SimulationManager
    from environment.tunnel import TunnelBuilder
    from optimizer.individualNeural import IndividualNeural
    from optimizer.individual import Individual
    from optimizer.neuralpopulation import NeuralPopulation
    from optimizer.population import Population
    from optimizer.mlpController import MLPController
    from curriculum import _load_and_organize_maps
    from controller import Supervisor
except ImportError as e:
    print(f"[ERROR] Failed to import required modules: {e}")
    sys.exit(1)


def evaluate_population_performance(supervisor, checkpoint_paths, num_individuals_to_test=20, num_maps_per_stage=3):
    """Evaluates average fitness per stage for multiple checkpoints."""
    all_results = {}

    # Load maps once for all evaluations
    print("Loading maps...")
    map_pool = _load_and_organize_maps()

    # Create a single SimulationManager instance
    print("Creating SimulationManager...")
    sim_mgr = SimulationManager(supervisor)

    for checkpoint_path in checkpoint_paths:
        print(f"\n{'=' * 40}")
        print(f"Processing: {checkpoint_path}")
        print(f"{'=' * 40}")

        try:
            print("Loading checkpoint...")
            with open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)
            population = data.get('population')
            max_trained_stage = data.get('stage', 0)

            if not population:
                print(f"[ERROR] No population found in {checkpoint_path}")
                continue

            all_individuals = population.individuals
            individuals = all_individuals[:min(num_individuals_to_test, len(all_individuals))]

            mode = 'NE' if isinstance(population, NeuralPopulation) else 'GA'
            print(f"Loaded population of {len(all_individuals)} individuals ({mode})")
            print(f"-> Evaluating {len(individuals)} individuals with {num_maps_per_stage} maps per stage")

            # Reset simulation for clean state
            print("Resetting simulation...")
            supervisor.simulationReset()
            supervisor.step(sim_mgr.timestep)  # Stabilize

            # Reset tunnel builder
            sim_mgr.tunnel_builder = TunnelBuilder(supervisor)

            results = defaultdict(list)
            all_available_stages = sorted(map_pool.keys())
            stages_to_evaluate = [s for s in all_available_stages if s <= max_trained_stage]

            for stage in tqdm(stages_to_evaluate, desc="Evaluating Stages"):
                if not map_pool.get(stage):
                    continue
                maps_to_run = random.sample(map_pool[stage], min(num_maps_per_stage, len(map_pool[stage])))

                for ind in individuals:
                    fitness_scores = []
                    for map_params in maps_to_run:
                        # Reset simulation for each map
                        supervisor.simulationReset()
                        supervisor.step(sim_mgr.timestep)

                        # Reset tunnel builder
                        sim_mgr.tunnel_builder = TunnelBuilder(supervisor)

                        if mode == 'NE':
                            fitness, _ = sim_mgr.run_experiment_with_network(ind, stage)
                        else:
                            fitness, _ = sim_mgr.run_experiment_with_params(
                                ind.distP, ind.angleP, stage)
                        fitness_scores.append(fitness)

                    avg_fitness = np.mean(fitness_scores) if fitness_scores else 0.0
                    results[stage].append(avg_fitness)

            print("\nEvaluation completed.")

            # Generate unique filename based on checkpoint
            checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
            results_filepath = f"evaluation/{checkpoint_name}_fitness_analysis.pkl"
            plot_filename = f"evaluation/{checkpoint_name}_fitness_heatmap.png"

            # Save results
            plot_data = {
                'results': results,
                'stages': stages_to_evaluate,
                'num_individuals': len(individuals),
                'mode': mode,
                'fitness_matrix': None
            }

            # Create fitness matrix for statistical analysis
            fitness_matrix = np.zeros((len(individuals), len(stages_to_evaluate)))
            for i, stage in enumerate(stages_to_evaluate):
                if stage in results:
                    fitness_matrix[:, i] = results[stage]

            plot_data['fitness_matrix'] = fitness_matrix

            with open(results_filepath, "wb") as f:
                pickle.dump(plot_data, f)
            print(f"Fitness results saved to '{results_filepath}'")

            # Generate and save fitness heatmap
            plot_fitness_heatmap(results, stages_to_evaluate, len(individuals), save_path=plot_filename)

            # Store for final report
            all_results[checkpoint_name] = plot_data

        except Exception as e:
            print(f"Error processing {checkpoint_path}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Ensure reset even after error
            supervisor.simulationReset()
            supervisor.step(sim_mgr.timestep)

    return all_results


def plot_fitness_heatmap(results, stages, num_individuals, save_path=None):
    """Generates heatmap with average fitness per stage and individual."""
    fitness_matrix = np.zeros((num_individuals, len(stages)))
    for i, stage in enumerate(stages):
        if results.get(stage):
            fitness_matrix[:, i] = results[stage]

    fig_width = max(15, len(stages) * 1.2)
    fig_height = max(12, num_individuals * 0.6)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Use inverted colormap (red = bad, green = good)
    cax = ax.matshow(fitness_matrix, cmap='RdYlGn')
    fig.colorbar(cax, label='Average Fitness')

    ax.set_xticks(np.arange(len(stages)))
    ax.set_yticks(np.arange(num_individuals))
    ax.set_xticklabels([f'Stage {s}' for s in stages])
    ax.set_yticklabels([f'Ind {i}' for i in range(num_individuals)])

    plt.xticks(rotation=45, ha="left", rotation_mode="anchor")
    ax.set_xlabel("Difficulty Stages")
    ax.set_ylabel("Individuals")
    ax.set_title("Average Fitness per Stage", pad=20)

    for i in range(num_individuals):
        for j in range(len(stages)):
            ax.text(j, i, f"{fitness_matrix[i, j]:.0f}",
                    ha="center", va="center", color="black", fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Fitness heatmap saved to '{save_path}'")
    else:
        plt.show()


def evaluate_reactive_controllers(supervisor, num_maps_per_stage=3):
    """Evaluates reactive controllers with detailed analysis."""
    from controllers.reactive_controller import reactive_controller_logic

    reactive_results = {}
    sim_mgr = SimulationManager(supervisor)
    map_pool = _load_and_organize_maps()

    for fov_mode in ['full', 'left', 'right']:
        model_name = f'REACTIVE_{fov_mode.upper()}'
        print(f"\nEvaluating reactive controller: {model_name}")

        results = {
            'fitness_scores': [],
            'stages': [],
            'success_rates': []
        }

        stages = sorted(map_pool.keys())

        for stage in stages:
            if not map_pool.get(stage):
                continue

            stage_fitness = []
            stage_successes = 0
            maps_to_run = random.sample(map_pool[stage], min(num_maps_per_stage, len(map_pool[stage])))

            for map_params in maps_to_run:
                # Reset simulation
                supervisor.simulationReset()
                supervisor.step(sim_mgr.timestep)

                # Execute reactive controller
                def controller(scan):
                    return reactive_controller_logic(scan, fov_mode=fov_mode)

                # Capture execution details
                run_result = sim_mgr._run_single_episode(controller, stage)
                fitness = run_result['fitness']
                success = run_result['success']

                stage_fitness.append(fitness)
                if success:
                    stage_successes += 1

            avg_fitness = np.mean(stage_fitness)
            success_rate = stage_successes / len(maps_to_run)

            results['fitness_scores'].append(avg_fitness)
            results['success_rates'].append(success_rate)
            results['stages'].append(stage)
            print(f"  Stage {stage}: Avg fitness = {avg_fitness:.0f}, Success rate = {success_rate:.0%}")

        reactive_results[model_name] = {
            'mode': 'REACTIVE',
            'fov_mode': fov_mode,
            'reactive_results': results
        }

    return reactive_results


def calculate_cohens_d(data1, data2):
    """Calculates Cohen's d effect size between two datasets."""
    n1, n2 = len(data1), len(data2)
    s1, s2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    return abs(np.mean(data1) - np.mean(data2)) / pooled_std


def generate_statistical_summary(all_results):
    """Generates statistical report comparing all models."""
    if not all_results:
        return None

    # Prepare data for analysis
    summary_data = []
    all_individual_avgs = []  # Stores per-individual averages for each model
    model_names = []

    for name, data in all_results.items():
        # For population models (NE/GA)
        if 'fitness_matrix' in data:
            # Calculate average per individual (rows of the matrix)
            individual_avgs = np.mean(data['fitness_matrix'], axis=1)
            all_individual_avgs.append(individual_avgs)

            # Sort individuals from best to worst
            sorted_avgs = np.sort(individual_avgs)[::-1]
            num_individuals = len(individual_avgs)

            # Calculate performance metrics
            summary_data.append({
                'Model': name,
                'Type': data['mode'],
                'Individuals': num_individuals,
                'Stages': len(data['stages']),
                'Best Individual': np.max(individual_avgs),
                'Worst Individual': np.min(individual_avgs),
                'Population Average': np.mean(individual_avgs),
                'Top 25% Average': np.mean(sorted_avgs[:int(num_individuals * 0.25)]),
                'Consistency (SD)': np.std(individual_avgs),
                'Best-Worst Gap': np.max(individual_avgs) - np.min(individual_avgs)
            })
            model_names.append(name)

        # For reactive models
        elif 'reactive_results' in data:
            # Reactive models treated as single individual
            fitness_scores = data['reactive_results']['fitness_scores']
            individual_avgs = np.array(fitness_scores)

            summary_data.append({
                'Model': name,
                'Type': 'REACTIVE',
                'Individuals': 1,
                'Stages': len(fitness_scores),
                'Best Individual': np.max(fitness_scores),
                'Worst Individual': np.min(fitness_scores),
                'Population Average': np.mean(fitness_scores),
                'Top 25% Average': np.mean(fitness_scores),
                'Consistency (SD)': 0,  # No variation
                'Best-Worst Gap': 0  # No variation
            })
            all_individual_avgs.append(individual_avgs)
            model_names.append(name)

    # Create DataFrame with descriptive statistics
    df_summary = pd.DataFrame(summary_data)

    # Perform statistical tests between models
    statistical_tests = []
    if len(all_results) > 1:
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                # Independent samples t-test
                try:
                    t_stat, p_value = stats.ttest_ind(
                        all_individual_avgs[i], all_individual_avgs[j],
                        equal_var=False
                    )
                except Exception as e:
                    print(f"t-test error between {model_names[i]} and {model_names[j]}: {e}")
                    continue

                # Effect size
                d_value = calculate_cohens_d(
                    all_individual_avgs[i], all_individual_avgs[j]
                )

                # Effect size interpretation
                effect_size = "Small" if d_value < 0.5 else "Medium" if d_value < 0.8 else "Large"

                # Significance interpretation
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "NS"

                # Determine superior model
                mean_i = np.mean(all_individual_avgs[i])
                mean_j = np.mean(all_individual_avgs[j])
                better_model = model_names[i] if mean_i > mean_j else model_names[j]
                diff = abs(mean_i - mean_j)
                advantage_pct = (diff / max(mean_i, mean_j)) * 100

                statistical_tests.append({
                    'Model A': model_names[i],
                    'Model B': model_names[j],
                    'Difference': diff,
                    'Superior Model': better_model,
                    't-statistic': t_stat,
                    'p-value': p_value,
                    'Significance': significance,
                    "Cohen's d": d_value,
                    "Effect Size": effect_size,
                    'Advantage': f"{advantage_pct:.1f}%"
                })

    return df_summary, pd.DataFrame(statistical_tests) if statistical_tests else None


def print_statistical_summary(df_summary, df_tests):
    """Prints formatted statistical report."""
    print("\n" + "=" * 80)
    print("FITNESS STATISTICAL SUMMARY".center(80))
    print("=" * 80)

    # Descriptive statistics
    print("\n📊 DESCRIPTIVE STATISTICS PER MODEL:")
    print(df_summary.to_string(index=False, float_format="%.0f"))

    # Statistical tests
    if df_tests is not None:
        print("\n\n🔬 STATISTICAL COMPARISONS BETWEEN MODELS:")
        print(df_tests.to_string(index=False, float_format=lambda x: "%.2f" % x if abs(x) > 1 else "%.3f" % x))

        print("\nLegend:")
        print("*** p < 0.001 | ** p < 0.01 | * p < 0.05 | NS: Not Significant")
        print("Effect size: d < 0.5 (Small) | 0.5 ≤ d < 0.8 (Medium) | d ≥ 0.8 (Large)")
        print("Superior Model: Model with higher average fitness")
        print("Advantage: Percentage difference relative to superior model")


def main():
    # Create global supervisor
    supervisor = Supervisor()

    results_dir = "evaluation"
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print("POPULATION FITNESS ANALYSIS".center(60))
    print("=" * 60)

    checkpoint_paths = []
    while True:
        path = input("\nEnter checkpoint path (.pkl), directory, or 'done' to finish: ").strip()
        if path.lower() in ['done', '']:
            break

        if not os.path.exists(path):
            print(f"❌ Path not found: {path}")
            continue

        if os.path.isfile(path) and path.endswith('.pkl'):
            checkpoint_paths.append(path)
            print(f"✅ File added: {path}")

        elif os.path.isdir(path):
            files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.pkl')]
            if not files:
                print(f"❌ No .pkl files found in {path}")
            else:
                checkpoint_paths.extend(files)
                print(f"✅ Added {len(files)} files from {path}")

    if not checkpoint_paths:
        print("\n❌ No valid checkpoints provided. Exiting.")
        return

    print(f"\n▶️ Processing {len(checkpoint_paths)} checkpoint(s)...")
    start_time = time.time()

    # Pass supervisor to functions
    all_results = evaluate_population_performance(supervisor, checkpoint_paths)

    # Evaluate reactive controllers
    print("\n▶️ Evaluating reactive controllers...")
    reactive_results = evaluate_reactive_controllers(supervisor)
    all_results.update(reactive_results)

    # Generate statistical summary
    df_summary, df_tests = generate_statistical_summary(all_results)

    print("\n" + "=" * 60)
    print("FINAL REPORT".center(60))
    print("=" * 60)

    # Print basic info about each model
    for name, data in all_results.items():
        if 'stages' in data:
            stages = data['stages']
            print(f"\n🔹 {name} ({data.get('mode', 'REACTIVE')}):")
            print(f"   Individuals: {data.get('num_individuals', 1)}, Stages: {len(stages)} (0-{max(stages)})")
        elif 'reactive_results' in data:
            print(f"\n🔹 {name} (REACTIVE):")
            print(f"   Stages: {len(data['reactive_results']['stages'])}")

        if 'reactive_results' in data:
            print(f"   Average fitness: {np.mean(data['reactive_results']['fitness_scores']):.0f}")

    # Print full statistical summary
    if df_summary is not None:
        print_statistical_summary(df_summary, df_tests)

    print(f"\n✅ Analysis completed in {time.time() - start_time:.1f} seconds")

    # Save results to CSV
    if df_summary is not None:
        summary_csv = os.path.join(results_dir, "fitness_summary.csv")
        df_summary.to_csv(summary_csv, index=False)
        print(f"\n📝 Fitness summary saved to: {summary_csv}")

        if df_tests is not None:
            tests_csv = os.path.join(results_dir, "fitness_statistical_tests.csv")
            df_tests.to_csv(tests_csv, index=False)
            print(f"📝 Fitness statistical tests saved to: {tests_csv}")


if __name__ == '__main__':
    main()