from optimizer.individualNeural import IndividualNeural
import random
import numpy as np

class NeuralPopulation:
    def __init__(self, pop_size, input_size, hidden_size, output_size, mutation_rate=0.1, elitism=1):
        """
        Manages a population of neural individuals for neuroevolution.
        """
        self.pop_size = pop_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.mutation_rate = mutation_rate
        self.elitism = elitism

        self.individuals = [
            IndividualNeural(input_size, hidden_size, output_size)
            for _ in range(pop_size)
        ]

    def evaluate(self, simulator):
        """
        Evaluates the fitness of each individual using the provided simulator.
        """
        for individual in self.individuals:
            individual.fitness = simulator.evaluate(individual)

    def select_parents(self, tournament_size=3):
        """
        Selects two parents using tournament selection.
        """
        competitors = random.sample(self.individuals, tournament_size)
        competitors.sort(key=lambda ind: ind.fitness, reverse=True)
        return competitors[0], competitors[1]

    def create_next_generation(self):
        """
        Generates the next generation via elitism, crossover and mutation.
        """
        self.individuals.sort(key=lambda ind: ind.fitness, reverse=True)
        next_generation = self.individuals[:self.elitism]  # Elitism

        while len(next_generation) < self.pop_size:
            parent1, parent2 = self.select_parents()
            child = parent1.crossover(parent2)
            child.mutate(mutation_rate=self.mutation_rate)
            next_generation.append(child)

        self.individuals = next_generation

    def get_best_individual(self):
        """
        Returns the best individual with a valid fitness.
        """
        valid_individuals = [ind for ind in self.individuals if ind.fitness is not None]
        if not valid_individuals:
            raise ValueError("[FATAL] Nenhum indivíduo com fitness válido.")
        return max(valid_individuals, key=lambda ind: ind.fitness)

