import random
import numpy as np
import os
from optimizer.individualNeural import IndividualNeural


class NeuralPopulation:
    def __init__(self, pop_size, input_size, hidden_size, output_size, mutation_rate=0.1, elitism=1):
        self.pop_size = pop_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.individuals = [IndividualNeural(input_size, hidden_size, output_size, id=i) for i in range(pop_size)]

    def evaluate(self, sim_manager, maps_to_run):
        """
        Evaluates the entire population using a preselected list of maps.
        """
        total_successes_population = 0
        debug_mode = os.environ.get('ROBOT_DEBUG_MODE') == '1'

        if not maps_to_run:
            if debug_mode:
                print("[WARNING] No maps provided for evaluation. Skipping this generation's evaluation.")
            return 0.0

        num_maps = len(maps_to_run)
        print(f"Evaluating each individual on {num_maps} preselected maps...")

        # Header for compact view in normal mode
        if not debug_mode:
            print("  Successes per Individual:", end="")

        for ind in self.individuals:
            individual_fitness_scores = []
            individual_success_count = 0

            for map_params in maps_to_run:
                stage = map_params['difficulty_level']

                fitness, succeeded = sim_manager.run_experiment_with_network(
                    ind, stage=stage
                )

                individual_fitness_scores.append(fitness)
                if succeeded:
                    individual_success_count += 1

            ind.fitness = np.mean(individual_fitness_scores) if individual_fitness_scores else 0.0
            ind.total_successes = individual_success_count
            total_successes_population += ind.total_successes

            # --- DEBUG/PRINT LOGIC ---
            if debug_mode:
                print(
                    f"    [DEBUG | NE Individual #{ind.id:02d}] Fitness: {ind.fitness:8.2f} | Successes: {ind.total_successes}/{num_maps}")
            else:
                print(f" {ind.total_successes}/{num_maps}", end="")
            # --- END PRINT LOGIC ---

        if not debug_mode:
            print()  # Newline after compact summary

        if self.pop_size == 0:
            return 0.0
        return total_successes_population / (self.pop_size * num_maps)


    def select_parents(self, tournament_size=3):
        """Selects two parents using tournament selection."""
        pool = self.individuals
        if len(pool) < 2:
            return pool[0], pool[0]

        contenders = random.sample(pool, min(tournament_size, len(pool)))
        contenders.sort(key=lambda ind: ind.fitness, reverse=True)
        return contenders[0], contenders[1] if len(contenders) > 1 else (contenders[0], contenders[0])

    def create_next_generation(self):
        """Creates the next generation by applying elitism, crossover, and mutation."""
        self.individuals.sort(key=lambda ind: ind.fitness, reverse=True)
        next_gen = []

        # Elitism
        for i in range(min(self.elitism, len(self.individuals))):
            elite = self.individuals[i]
            copy = IndividualNeural(
                elite.input_size, elite.hidden_size, elite.output_size, elite.get_genome(), id=i
            )
            copy.fitness = elite.fitness
            next_gen.append(copy)

        # Fill the rest of the population with offspring
        while len(next_gen) < self.pop_size:
            p1, p2 = self.select_parents()
            child = p1.crossover(p2, id=len(next_gen))
            child.mutate(mutation_rate=self.mutation_rate, mutation_strength=0.1)
            next_gen.append(child)

        self.individuals = next_gen

    def get_best_individual(self):
        """Returns the individual with the highest fitness."""
        if not self.individuals:
            return None
        return max(self.individuals, key=lambda ind: ind.fitness)