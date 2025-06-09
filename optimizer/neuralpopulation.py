from optimizer.individualNeural import IndividualNeural
import random
import numpy as np
from environment.configuration import MAX_DIFFICULTY_STAGE
from typing import List  # Import List for type hinting


class NeuralPopulation:
    def __init__(self, pop_size, input_size, hidden_size, output_size, mutation_rate=0.1, elitism=1):
        self.pop_size = pop_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.mutation_rate = mutation_rate
        self.elitism = elitism

        self.individuals = [ \
            IndividualNeural(input_size, hidden_size, output_size, id=i) \
            for i in range(pop_size) \
            ]

    def evaluate(self, simulator, difficulty_levels_to_test: List[float]):
        """
        Evaluates the fitness of each individual in the population by running them through
        multiple simulation stages specified in `difficulty_levels_to_test`.

        Args:
            simulator: An instance of SimulationManager.
            difficulty_levels_to_test (List[float]): A list of floats, where each float represents
                                                      a difficulty stage for one simulation episode.
        """
        print(f"[EVAL] Evaluating {len(self.individuals)} individuals on {len(difficulty_levels_to_test)} maps each.")

        for individual in self.individuals:
            total_fitness_for_individual = 0.0
            successful_runs_count = 0
            total_maps_attempted = len(difficulty_levels_to_test)

            # Reset successful_goals and total_evaluations_for_stage for the current evaluation round
            individual.successful_goals = 0
            individual.total_evaluations_for_stage = 0

            print(f"\n[EVAL] -> Individual {getattr(individual, 'id', '?')} starting multi-map evaluation...")

            for map_idx, difficulty_stage in enumerate(difficulty_levels_to_test):
                try:
                    print(
                        f"[EVAL] ...Running map {map_idx + 1}/{total_maps_attempted} (Difficulty {difficulty_stage:.1f})")
                    # simulator.run_experiment_with_network now returns fitness AND success status
                    fitness, success_status = simulator.run_experiment_with_network(
                        individual,
                        stage=difficulty_stage,
                        total_stages=MAX_DIFFICULTY_STAGE  # Pass MAX_DIFFICULTY_STAGE for get_stage_parameters
                    )
                    total_fitness_for_individual += fitness
                    if success_status:
                        successful_runs_count += 1
                except Exception as e:
                    print(
                        f"[ERROR] Simulation failed for Individual {individual.id} on Difficulty {difficulty_stage:.1f}: {e}")
                    total_fitness_for_individual += -10000.0  # Severe penalty for simulation errors
                    # Do not increment successful_runs_count for failed simulations

                individual.total_evaluations_for_stage += 1  # Increment regardless of success/failure

            # Calculate individual's average fitness over all tested maps
            individual.fitness = total_fitness_for_individual / total_maps_attempted if total_maps_attempted > 0 else -1e6

            # Store success count for population-level success rate calculation
            individual.successful_goals = successful_runs_count  # Update with the count for this evaluation round

            print(
                f"[RESULT] Individual {getattr(individual, 'id', '?')} -> Avg Fitness: {individual.fitness:.2f} | Successful Maps: {successful_runs_count}/{total_maps_attempted}")

    def select_parents(self, tournament_size=3):
        """
        Selects two parents using tournament selection.
        """
        # Ensure there are enough individuals for tournament selection
        if len(self.individuals) < tournament_size:
            # Fallback: simple random selection if population is too small for tournament
            print(
                f"[WARNING] Population size ({len(self.individuals)}) is less than tournament_size ({tournament_size}). Falling back to random parent selection.")
            parent1 = random.choice(self.individuals)
            parent2 = random.choice(self.individuals)
            return parent1, parent2

        competitors = random.sample(self.individuals, tournament_size)
        competitors.sort(key=lambda ind: ind.fitness, reverse=True)
        return competitors[0], competitors[1]

    def create_next_generation(self):
        """
        Generates the next generation via elitism, crossover and mutation.
        """
        self.individuals.sort(key=lambda ind: ind.fitness, reverse=True)
        next_generation = []

        # Elitism: preserve top individuals
        for i in range(self.elitism):
            if i < len(self.individuals):  # Ensure we don't go out of bounds if elitism > pop_size
                elite = self.individuals[i]
                # Create a deep copy of the elite individual to prevent direct modification
                copied_elite = IndividualNeural(
                    elite.input_size,
                    elite.hidden_size,
                    elite.output_size,
                    elite.get_genome(),  # Pass the genome directly to ensure it's copied
                    id=i  # Assign ID for the next generation
                )
                next_generation.append(copied_elite)
            else:
                # If elitism value is too high for current pop_size, just break
                break

        # Generate remaining children with assigned IDs
        while len(next_generation) < self.pop_size:
            parent1, parent2 = self.select_parents()
            child_id = len(next_generation)  # Assign a unique ID for the new child

            # Perform crossover
            child = parent1.crossover(parent2, id=child_id)

            # Perform mutation
            child.mutate(mutation_rate=self.mutation_rate)

            next_generation.append(child)

        self.individuals = next_generation  # Replace old population with the new generation

    def get_best_individual(self):
        """
        Returns the best individual with a valid fitness from the current population.
        If no individuals have valid fitness, raises an error.
        """
        valid_individuals = [ind for ind in self.individuals if ind.fitness is not None and not np.isnan(ind.fitness)]
        if not valid_individuals:
            raise ValueError(
                "[FATAL] No valid individuals found in the population to determine the best. Check fitness assignment.")

        # Sort individuals by fitness in descending order
        best_individual = max(valid_individuals, key=lambda ind: ind.fitness)
        return best_individual
