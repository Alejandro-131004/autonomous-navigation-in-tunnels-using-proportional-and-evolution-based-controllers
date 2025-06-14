import random
import numpy as np
import os
from optimizer.individual import Individual


class Population:
    def __init__(self, pop_size, mutation_rate=0.1, elitism=1, tournament_size=3):
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.tournament_size = tournament_size
        self.individuals = [Individual(id=i) for i in range(pop_size)]

    def evaluate(self, sim_manager, maps_to_run):
        """
        Avalia toda a população usando uma lista pré-selecionada de mapas.
        """
        total_successes_population = 0
        debug_mode = os.environ.get('ROBOT_DEBUG_MODE') == '1'

        if not maps_to_run:
            if debug_mode:
                print("[AVISO] Nenhum mapa fornecido para avaliação. A saltar a avaliação desta geração.")
            return 0.0

        num_maps = len(maps_to_run)
        print(f"A avaliar cada indivíduo em {num_maps} mapas pré-selecionados...")

        # Cabeçalho para o modo normal, para dar contexto à linha de resultados
        if not debug_mode:
            print("  Sucessos por Indivíduo:", end="")

        for ind in self.individuals:
            individual_fitness_scores = []
            individual_success_count = 0

            for map_params in maps_to_run:
                stage = map_params['difficulty_level']
                distP, angleP = ind.get_genes()

                fitness, succeeded = sim_manager.run_experiment_with_params(
                    distP, angleP, stage=stage
                )

                individual_fitness_scores.append(fitness)
                if succeeded:
                    individual_success_count += 1

            ind.fitness = np.mean(individual_fitness_scores) if individual_fitness_scores else 0.0
            ind.total_successes = individual_success_count
            total_successes_population += ind.total_successes

            # --- LÓGICA DE PRINT ATUALIZADA ---
            if debug_mode:
                print(
                    f"    [DEBUG | Indivíduo GA #{ind.id:02d}] Fitness: {ind.fitness:8.2f} | Sucessos: {ind.total_successes}/{num_maps}")
            else:
                # Imprime de forma compacta, sem nova linha
                print(f" {ind.total_successes}/{num_maps}", end="")
            # --- FIM DA ALTERAÇÃO ---

        # Adiciona uma nova linha no final da lista compacta do modo normal
        if not debug_mode:
            print()

        if self.pop_size == 0:
            return 0.0
        return total_successes_population / self.pop_size

    def select_parents(self):
        """Seleciona dois pais usando seleção por torneio."""
        pool = self.individuals
        if len(pool) < 2:
            return pool[0], pool[0]

        contenders = random.sample(pool, min(self.tournament_size, len(pool)))
        contenders.sort(key=lambda ind: ind.fitness, reverse=True)

        return contenders[0], contenders[1] if len(contenders) > 1 else (contenders[0], contenders[0])

    def create_next_generation(self):
        """Cria a próxima geração aplicando elitismo, crossover e mutação."""
        self.individuals.sort(key=lambda ind: ind.fitness, reverse=True)
        next_gen = []

        for i in range(min(self.elitism, len(self.individuals))):
            elite = self.individuals[i]
            copy = Individual(elite.distP, elite.angleP, id=i)
            copy.fitness = elite.fitness
            next_gen.append(copy)

        while len(next_gen) < self.pop_size:
            p1, p2 = self.select_parents()
            child = p1.crossover(p2, id=len(next_gen))
            child.mutate(self.mutation_rate)
            next_gen.append(child)

        self.individuals = next_gen

    def get_best_individual(self):
        """Retorna o indivíduo com a maior fitness na população atual."""
        if not self.individuals:
            return None
        return max(self.individuals, key=lambda ind: ind.fitness)
