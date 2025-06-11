from optimizer.individualNeural import IndividualNeural
import random
import numpy as np


class NeuralPopulation:
    def __init__(self, pop_size, input_size, hidden_size, output_size, mutation_rate=0.1, elitism=1):
        self.pop_size = pop_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.individuals = [IndividualNeural(input_size, hidden_size, output_size, id=i) for i in range(pop_size)]

    def evaluate(self, simulator, difficulty_levels: list, total_stages: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluates each individual on a random map for each provided difficulty level.

        Returns:
            fitness_matrix (np.ndarray): Matrix of fitness scores (individual x difficulty).
            success_matrix (np.ndarray): Matrix of success flags (individual x difficulty).
        """
        N = len(self.individuals)
        L = len(difficulty_levels)
        fitness_matrix = np.zeros((N, L), dtype=float)
        success_matrix = np.zeros((N, L), dtype=int)

        for j, level in enumerate(difficulty_levels):
            # print(f"  -> Evaluating population at difficulty level: {level}")
            for i, ind in enumerate(self.individuals):
                fitness, succeeded = simulator.run_experiment_with_network(ind, stage=level, total_stages=total_stages)
                fitness_matrix[i, j] = fitness
                success_matrix[i, j] = 1 if succeeded else 0

        for i, ind in enumerate(self.individuals):
            ind.fitness = np.mean(fitness_matrix[i])
            ind.avg_fitness = ind.fitness
            ind.successes = int(np.sum(success_matrix[i]))

        return fitness_matrix, success_matrix

    def select_parents(self, parent_pool, tournament_size=3):
        """
        Selects two parents from a given parent_pool using tournament selection.

        If the pool is too small, selects randomly from available parents.
        """
        if len(parent_pool) < tournament_size:
            # If pool is too small, select from those available
            return random.sample(parent_pool, 2) if len(parent_pool) >= 2 else (parent_pool[0], parent_pool[0])

        competitors = random.sample(parent_pool, tournament_size)
        competitors.sort(key=lambda ind: ind.fitness, reverse=True)
        return competitors[0], competitors[1]

    def create_next_generation(self, parent_pool=None):
        """
        Creates the next generation.
        If `parent_pool` is provided, uses it for reproduction.
        Otherwise, uses the internal population.
        """
        if parent_pool is None:
            parent_pool = self.individuals

        # Ensure parent_pool is sorted by fitness descending
        parent_pool.sort(key=lambda ind: ind.fitness, reverse=True)

        next_generation = []

        # Elitism: best from parent_pool pass directly
        num_elites = min(self.elitism, len(parent_pool))
        for i in range(num_elites):
            elite = parent_pool[i]
            copied_elite = IndividualNeural(
                elite.input_size, elite.hidden_size, elite.output_size,
                elite.get_genome(), id=i
            )
            next_generation.append(copied_elite)

        # Crossover and Mutation: generate rest of population from parent_pool
        while len(next_generation) < self.pop_size:
            parent1, parent2 = self.select_parents(parent_pool)
            child_id = len(next_generation)
            child = parent1.crossover(parent2, id=child_id)
            child.mutate(mutation_rate=self.mutation_rate, mutation_strength=0.1)
            next_generation.append(child)

        self.individuals = next_generation

    def get_best_individual(self):
        """Returns the best individual in the current population."""
        valid_individuals = [ind for ind in self.individuals if ind.fitness is not None]
        if not valid_individuals:
            return random.choice(self.individuals)
        return max(valid_individuals, key=lambda ind: ind.fitness)
