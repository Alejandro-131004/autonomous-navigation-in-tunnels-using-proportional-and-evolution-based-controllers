import random
from individual import Individual

class Population:
    def __init__(self, size, mutation_rate=0.1, elitism=1):
        self.size = size
        self.mutation_rate = mutation_rate
        self.elitism = elitism  # Number of top individuals to copy directly (elitism)
        self.individuals = [Individual() for _ in range(size)]

    def evaluate(self, simulator):
        """
        Evaluates the fitness all individuals in the population using the provided simulator.
        """
        for individual in self.individuals:
            distP, angleP = individual.get_genes()
            individual.fitness = simulator.run_with_parameters(distP, angleP)

    def select_parents(self):
        """
        Tournament selection: chooses the 2 individuals with the highest fitness.
        """
        tournament_size = 3
        parents = random.sample(self.individuals, tournament_size)
        parents.sort(key=lambda ind: ind.fitness, reverse=True)  # Higher fitness is better
        return parents[0], parents[1]

    def create_next_generation(self):
        """
        Creates the next generation using elitism + crossover + mutation.
        The top individuals are preserved, and the rest are generated through genetic operations.
        """
        # Sort the current population by fitness
        self.individuals.sort(key=lambda ind: ind.fitness, reverse=True)

        next_generation = []

        # Elitism: copy the best individuals
        for i in range(self.elitism):
            next_generation.append(self.individuals[i])

        # Generate the rest of the new generation
        while len(next_generation) < self.size:
            parent1, parent2 = self.select_parents()
            child = parent1.crossover(parent2)
            child.mutate(mutation_rate=self.mutation_rate)
            next_generation.append(child)

        self.individuals = next_generation

    def get_best_individual(self):
        """
        Returns the best individual in the population.
        """
        return max(self.individuals, key=lambda ind: ind.fitness)