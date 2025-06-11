# optimizer/population.py
from optimizer.individual import Individual
import random

class Population:
    def _init_(self, pop_size, mutation_rate=0.1, elitism=1, tournament_size=3):
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.tournament_size = tournament_size
        self.individuals = [Individual(id=i) for i in range(pop_size)]

    def evaluate(self, sim_manager, stage):
        for ind in self.individuals:
            distP, angleP = ind.get_genes()
            fitness, succeeded = sim_manager.run_experiment_with_params(distP, angleP, stage)
            ind.fitness = fitness
            # acumula quantas vezes passou com sucesso
            ind.successes = getattr(ind, 'successes', 0) + (1 if succeeded else 0)


    def select_parents(self):
        pool = sorted(self.individuals, key=lambda ind: ind.fitness, reverse=True)
        # torneio simples:
        contenders = random.sample(pool, self.tournament_size)
        contenders.sort(key=lambda ind: ind.fitness, reverse=True)
        return contenders[0], contenders[1] if len(contenders)>1 else (contenders[0], contenders[0])

    def create_next_generation(self):
        self.individuals.sort(key=lambda ind: ind.fitness, reverse=True)
        next_gen = []
        # elitismo
        for i in range(min(self.elitism, len(self.individuals))):
            elite = self.individuals[i]
            copy = Individual(elite.distP, elite.angleP, id=i)
            next_gen.append(copy)
        # resto via crossover+mutação
        while len(next_gen) < self.pop_size:
            p1, p2 = self.select_parents()
            child = p1.crossover(p2, id=len(next_gen))
            child.mutate(self.mutation_rate)
            next_gen.append(child)
        self.individuals = next_gen

    def get_best_individual(self):
        return max(self.individuals, key=lambda ind: ind.fitness)