from optimizer.individual import Individual
import random

class Population:
    def __init__(self, pop_size, mutation_rate=0.1, elitism=1, tournament_size=3):
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.tournament_size = tournament_size
        self.individuals = [Individual(id=i) for i in range(pop_size)]

    def evaluate(self, sim_manager, stage):
        """Evaluates all individuals using the simulation manager at a given stage."""
        for ind in self.individuals:
            distP, angleP = ind.get_genes()
            fitness, succeeded = sim_manager.run_experiment_with_params(distP, angleP, stage)
            ind.fitness = fitness
            # Accumulate number of successful runs
            ind.successes = getattr(ind, 'successes', 0) + (1 if succeeded else 0)

    def select_parents(self):
        """Selects two parents using simple tournament selection."""
        pool = sorted(self.individuals, key=lambda ind: ind.fitness, reverse=True)
        contenders = random.sample(pool, self.tournament_size)
        contenders.sort(key=lambda ind: ind.fitness, reverse=True)
        if len(contenders) > 1:
            return contenders[0], contenders[1]
        else:
            return contenders[0], contenders[0]

    def create_next_generation(self):
        """Creates the next generation applying elitism and crossover + mutation."""
        self.individuals.sort(key=lambda ind: ind.fitness, reverse=True)
        next_gen = []

        # Elitism: directly copy the best individuals
        for i in range(min(self.elitism, len(self.individuals))):
            elite = self.individuals[i]
            copy = Individual(elite.distP, elite.angleP, id=i)
            next_gen.append(copy)

        # Fill the rest of the population with offspring
        while len(next_gen) < self.pop_size:
            p1, p2 = self.select_parents()
            child = p1.crossover(p2, id=len(next_gen))
            child.mutate(self.mutation_rate)
            next_gen.append(child)

        self.individuals = next_gen

    def get_best_individual(self):
        """Returns the individual with the highest fitness."""
        return max(self.individuals, key=lambda ind: ind.fitness)
