import os
import sys
import numpy as np

from controller import Supervisor
from environment.simulation_manager import SimulationManager
from optimizer.individual import Individual

# Clean up sys.path from wrong controller entries
sys.path = [p for p in sys.path if 'controller' not in p]
sys.path.insert(0, '/Applications/Webots.app/Contents/lib/controller/python')  # Adjust path if needed

if __name__ == "__main__":
    # Experiment parameters
    population_sizes = [10, 20, 30]
    thresholds = [800, 960, 1120, 1280]
    generations_per_stage = 5
    max_stage = 10
    hidden_size = 16
    output_size = 2
    mutation_rate = 0.1
    elitism = 1

    print("Initializing supervisor and simulation manager...")
    supervisor = Supervisor()
    simulator = SimulationManager(supervisor)

    # Read LIDAR input size
    lidar_data = np.nan_to_num(simulator.lidar.getRangeImage(), nan=0.0)
    input_size = len(lidar_data)

    # Loop over parameter combinations
    for pop_size in population_sizes:
        for initial_threshold in thresholds:
            print(f"\nStarting experiment: population_size={pop_size}, threshold={initial_threshold:.2f}")
            current_stage = 0
            performance_threshold = initial_threshold
            fitness_history = []
            best_individual = None

            # Create result directory for this combination
            run_name = f"pop{pop_size}_thr{int(performance_threshold * 100)}"
            os.makedirs(f"results/{run_name}", exist_ok=True)

            while current_stage <= max_stage:
                print(f"\nSTAGE {current_stage} - Threshold: {performance_threshold:.2f}")
                print(f"[MAIN] Running run_neuroevolution() with current_stage = {current_stage}")

                # Redefine evaluation function for the current stage
                simulator.evaluate = lambda individual: simulator.run_experiment_with_network(
                    individual, stage=current_stage
                )

                # Run neuroevolution for this stage
                best_individual_stage, history_stage = simulator.run_neuroevolution(
                    generations=generations_per_stage,
                    pop_size=pop_size,
                    input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=output_size,
                    mutation_rate=mutation_rate,
                    elitism=elitism,
                    current_stage=current_stage
                )

                avg_fitness = np.mean(history_stage)
                print(f"Average fitness at stage {current_stage}: {avg_fitness:.2f}")
                fitness_history.extend(history_stage)

                # Update best global individual
                if best_individual is None or best_individual_stage.fitness > best_individual.fitness:
                    best_individual = best_individual_stage

                # Check if stage should be increased
                if avg_fitness >= performance_threshold:
                    print(f"Threshold reached ({avg_fitness:.2f} ≥ {performance_threshold:.2f})")
                    current_stage += 1
                else:
                    print("Performance not sufficient to advance to the next stage.")

            # Save fitness history and best genome
            np.savetxt(f"results/{run_name}/fitness_log.txt", fitness_history)
            np.save(f"results/{run_name}/best_genome.npy", best_individual.get_genome())

            print(f"Results saved in results/{run_name}/")
            print(f"Best final fitness: {best_individual.fitness:.2f}")

            # Run best individual in final simulation
            print(f"\nRunning best individual on final simulation (Stage {current_stage - 1})...")
            simulator.run_experiment_with_network(best_individual, stage=current_stage - 1)
