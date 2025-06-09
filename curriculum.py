# curriculum.py

import os
import numpy as np
import math  # Import math for degrees conversion in print statements

from environment.simulation_manager import SimulationManager
from optimizer.neuralpopulation import NeuralPopulation
from environment.configuration import MAX_DIFFICULTY_STAGE, \
    get_stage_parameters  # Import get_stage_parameters for internal use


def run_curriculum(
        supervisor,
        pop_size: int = 30,
        generations_per_difficulty_check: int = 5,
        success_rate_threshold: float = 0.75,
        hidden_size: int = 16,
        mutation_rate: float = 0.1,
        elitism: int = 1,
        initial_difficulty_level: int = 1,
        max_difficulty_level: int = 10
):
    """
    Drives a Neuroevolutionary curriculum across multiple difficulty stages.

    Args:
        supervisor: Webots Supervisor instance.
        pop_size: Number of individuals per generation.
        generations_per_difficulty_check: Number of generations to run at each difficulty level
                                          before checking for advancement.
        success_rate_threshold: Fraction of successful runs required to unlock the next level.
        hidden_size: Number of hidden neurons in the MLP.
        mutation_rate: GA mutation probability.
        elitism: Number of elites to carry over each generation.
        initial_difficulty_level: The starting difficulty level for the curriculum (e.g., 1).
        max_difficulty_level: The maximum difficulty level in the curriculum (e.g., 10).

    Returns:
        best_individual_overall: The individual that performed best across the entire curriculum.
    """
    sim_mgr = SimulationManager(supervisor)

    # Infer input size from Lidar (assuming it's constant for the robot)
    # This needs to be done once to initialize the neural network population
    lidar_data_sample = np.nan_to_num(sim_mgr.lidar.getRangeImage(), nan=0.0)
    input_size = len(lidar_data_sample)
    output_size = 2  # Assuming 2 outputs (linear and angular velocity)

    # Initialize the Neural Population
    population = NeuralPopulation(pop_size, input_size, hidden_size, output_size,
                                  mutation_rate=mutation_rate, elitism=elitism)

    best_individual_overall = None  # To track the best individual found throughout the entire curriculum
    current_difficulty_level = initial_difficulty_level  # Start curriculum at the initial level

    # Main Curriculum Learning Loop
    while current_difficulty_level <= max_difficulty_level:
        print(f"\n--- STARTING DIFFICULTY LEVEL {current_difficulty_level}/{max_difficulty_level} ---")

        # Determine the number of maps each individual will be tested on for this difficulty level
        # Based on your requirement: Level 1 -> 1 map, Level 10 -> 10 maps.
        num_maps_per_individual = int(current_difficulty_level)  # Ensure it's an integer

        # Create the list of difficulty stages that each individual will attempt for evaluation.
        # Each map will be generated with parameters corresponding to `current_difficulty_level`.
        # The diversity *within* this level is handled by `get_stage_parameters` in environment/configuration.py.
        difficulty_levels_to_test_in_stage = [float(current_difficulty_level) for _ in range(num_maps_per_individual)]

        print(
            f"Each individual will be evaluated on {num_maps_per_individual} map(s) at Difficulty Level {current_difficulty_level}.")

        # Run generations for the current difficulty level
        for gen_in_level in range(1, generations_per_difficulty_check + 1):
            print(
                f"\nDifficulty Level {current_difficulty_level} - Generation {gen_in_level}/{generations_per_difficulty_check}")

            # Evaluate the entire population on the defined set of maps for the current difficulty level
            population.evaluate(sim_mgr, difficulty_levels_to_test_in_stage)

            # Collect success data and fitnesses from the evaluated population
            total_successful_goals_in_generation = 0
            total_evaluations_in_generation = 0
            all_fitnesses_in_generation = []

            for ind in population.individuals:
                # `individualNeural` now stores these attributes (`successful_goals`, `total_evaluations_for_stage`)
                total_successful_goals_in_generation += getattr(ind, 'successful_goals', 0)
                total_evaluations_in_generation += getattr(ind, 'total_evaluations_for_stage', 0)
                if ind.fitness is not None:
                    all_fitnesses_in_generation.append(ind.fitness)

            avg_fitness_generation = np.mean(all_fitnesses_in_generation) if all_fitnesses_in_generation else 0.0

            # Get the best individual of the current generation
            best_individual_in_gen = None
            try:
                best_individual_in_gen = population.get_best_individual()
                best_fitness_generation = best_individual_in_gen.fitness
            except ValueError:
                best_fitness_generation = -float('inf')  # If no valid individuals, best fitness is very low

            print(
                f"Average Fitness (Level {current_difficulty_level}, Gen {gen_in_level}): {avg_fitness_generation:.2f}")
            print(f"Best Fitness (Level {current_difficulty_level}, Gen {gen_in_level}): {best_fitness_generation:.2f}")

            # Calculate the population's success rate for this generation
            success_rate_population = 0.0
            if total_evaluations_in_generation > 0:
                success_rate_population = total_successful_goals_in_generation / total_evaluations_in_generation

            print(
                f"Population Success Rate (Level {current_difficulty_level}, Gen {gen_in_level}): {success_rate_population:.2f} ({total_successful_goals_in_generation}/{total_evaluations_in_generation} successful maps)")

            # Update the best individual found across the entire curriculum
            if best_individual_overall is None or (
                    best_individual_in_gen and best_individual_in_gen.fitness > best_individual_overall.fitness):
                best_individual_overall = best_individual_in_gen

            # Curriculum Advancement Check
            # Check for advancement based on population success rate
            if success_rate_population >= success_rate_threshold:
                print(
                    f"[ADVANCE] Population success rate ({success_rate_population:.2f}) met/exceeded threshold ({success_rate_threshold:.2f}). Advancing to next difficulty level!")
                current_difficulty_level += 1  # Increment difficulty
                break  # Break out of the inner generations loop, proceed to next difficulty in while loop
            else:
                # If threshold not met, evolve population for the next generation at the SAME difficulty level
                print(
                    f"[RETRY] Success rate insufficient. Evolving population for another generation at Level {current_difficulty_level}.")
                population.create_next_generation()  # Evolve for next generation

        # This part of the loop is reached when `generations_per_difficulty_check` are exhausted
        # OR when the `success_rate_threshold` is met and the inner loop `break`s.

        # If the curriculum finished (current_difficulty_level exceeded max_difficulty_level)
        if current_difficulty_level > max_difficulty_level:
            print("\n🎉 CURRICULUM COMPLETED! The algorithm successfully reached the maximum difficulty level.")
            break  # Exit the main curriculum loop (outer while loop)

    # --- End of Curriculum Learning ---
    # Save final results (outside the main curriculum loop)
    run_name = f"cl_pop{pop_size}_sr{int(success_rate_threshold * 100)}_max_diff{int(max_difficulty_level)}"
    results_dir = f"results/{run_name}"
    os.makedirs(results_dir, exist_ok=True)

    # Save best genome
    if best_individual_overall:
        np.save(f"{results_dir}/best_genome_final.npy", best_individual_overall.get_genome())
        print(f"\nFinal Best Individual Genome saved to: {results_dir}/best_genome_final.npy")
        print(f"Final Best Individual Fitness (Overall): {best_individual_overall.fitness:.2f}")
    else:
        print("\nNo best individual found during the curriculum.")

    # Optional: Run the best overall individual in a final demonstration simulation
    if best_individual_overall:
        print(f"\nRunning final demonstration with the best overall individual (Level {max_difficulty_level})...")
        # Ensure the simulator runs the final demo at the hardest level.
        final_demo_fitness, final_demo_success = sim_mgr.run_experiment_with_network(best_individual_overall,
                                                                                     float(max_difficulty_level),
                                                                                     MAX_DIFFICULTY_STAGE)
        print(f"Final Demo Run Fitness: {final_demo_fitness:.2f} | Success: {final_demo_success}")

    print("\nNeuroevolution experiment concluded.")

    return best_individual_overall  # Return the best individual for further use if desired
