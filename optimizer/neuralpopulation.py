from optimizer.individualNeural import IndividualNeural
import random
import numpy as np
from environment.configuration import MAX_DIFFICULTY_STAGE


class NeuralPopulation:
    def __init__(self, pop_size, input_size, hidden_size, output_size, mutation_rate=0.1, elitism=1):
        self.pop_size = pop_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.mutation_rate = mutation_rate
        self.elitism = elitism

        self.individuals = [
            IndividualNeural(input_size, hidden_size, output_size, id=i)
            for i in range(pop_size)
        ]

    '''def evaluate(self, simulator, current_difficulty=1, total_stages=MAX_DIFFICULTY_STAGE):
        """
        Evaluate each individual on `current_difficulty` maps, all at the same difficulty level.
        Passes both the stage index and total number of stages to the simulator so that
        get_stage_parameters(stage_index, total_stages) can normalize correctly.
        """
        print(
            f"[EVALUATE] {len(self.individuals)} individuals × "
            f"{current_difficulty} maps each (Difficulty {current_difficulty}/{total_stages})"
        )
        self.success_count = 0

        for individual in self.individuals:
            fitnesses = []
            individual_successes = 0
            print(f"\n[EVAL] → Individual {getattr(individual, 'id', '?')}")

            # For each map (same difficulty), prepare environment, run simulation, record results
            for map_idx in range(current_difficulty):
                try:
                    # Clear old walls and set up the map parameters
                    simulator.prepare_environment(current_difficulty, map_idx)

                    # Run the simulation, passing both the stage and the total number of stages
                    fitness, success = simulator.run_experiment_with_network(
                        individual,
                        stage=current_difficulty,
                        total_stages=total_stages
                    )

                    fitnesses.append(fitness)
                    if success:
                        individual_successes += 1
                        self.success_count += 1

                    print(
                        f"[MAP {map_idx + 1}/{current_difficulty}] "
                        f"Fitness: {fitness:.2f} | Success: {success}"
                    )

                except Exception as e:
                    print(f"[ERROR] Failed on map {map_idx + 1}: {e}")

            # If at least one run succeeded, average the fitness; otherwise assign heavy penalty
            if fitnesses:
                individual.fitness = np.mean(fitnesses)
            else:
                individual.fitness = -1e6

            # Also keep avg_fitness for tracking
            individual.avg_fitness = individual.fitness
            individual.successes = individual_successes

            print(
                f"[RESULT] Individual {getattr(individual, 'id', '?')} → "
                f"Avg Fitness: {individual.fitness:.2f} | "
                f"Successes: {individual_successes}/{current_difficulty}"
            )'''

    
    import numpy as np

    def evaluate(
        self,
        simulator,
        difficulty_levels: list[int],
        total_stages: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate each individual on exactly one random map at
        each difficulty in `difficulty_levels`, level by level.

        Returns:
        fitness_matrix: shape (N, L) of floats
        success_matrix: shape (N, L) of 0/1 ints

        Where N = number of individuals, L = len(difficulty_levels).
        """
        N = len(self.individuals)
        L = len(difficulty_levels)

        # allocate storage
        fitness_matrix = np.zeros((N, L), dtype=float)
        success_matrix = np.zeros((N, L), dtype=int)

        # loop levels × individuals
        for j, lvl in enumerate(difficulty_levels):
            for i, ind in enumerate(self.individuals):
                # returns (fitness, success_bool)
                f, succeeded = simulator.run_experiment_with_network(
                    ind,
                    stage=lvl,
                    total_stages=total_stages
                )
                fitness_matrix[i, j] = f
                success_matrix[i, j] = 1 if succeeded else 0

        # update each individual's averaged stats (for selection, logging…)
        for i, ind in enumerate(self.individuals):
            ind.fitness = fitness_matrix[i].mean()
            ind.avg_fitness = ind.fitness
            ind.successes = int(success_matrix[i].sum())

        return fitness_matrix, success_matrix

    
    def select_parents(self, tournament_size=3):
        competitors = random.sample(self.individuals, tournament_size)
        competitors.sort(key=lambda ind: ind.fitness, reverse=True)
        return competitors[0], competitors[1]

    def create_next_generation(self):
        self.individuals.sort(key=lambda ind: ind.fitness, reverse=True)
        next_generation = []

        # Elitism: preserve top individuals with correct IDs
        for i in range(self.elitism):
            elite = self.individuals[i]
            copied = IndividualNeural(
                elite.input_size,
                elite.hidden_size,
                elite.output_size,
                elite.get_genome(),
                id=i
            )
            next_generation.append(copied)

        # Generate remaining children with assigned IDs
        while len(next_generation) < self.pop_size:
            parent1, parent2 = self.select_parents()
            child_id = len(next_generation)
            child = parent1.crossover(parent2, id=child_id)
            child.mutate(mutation_rate=self.mutation_rate)
            next_generation.append(child)

        self.individuals = next_generation

    def get_best_individual(self):
        valid_individuals = [ind for ind in self.individuals if ind.fitness is not None]
        if not valid_individuals:
            raise ValueError("[FATAL] No individual with valid fitness.")
        return max(valid_individuals, key=lambda ind: ind.fitness)
