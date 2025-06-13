import random
import numpy as np
from optimizer.individual import Individual


class Population:
    def __init__(self, pop_size, mutation_rate=0.1, elitism=1, tournament_size=3):
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.tournament_size = tournament_size
        self.individuals = [Individual(id=i) for i in range(pop_size)]

    def evaluate(self, sim_manager, current_stage, map_pool, runs_per_eval=10):
        """
        Avalia toda a população, garantindo que são sempre usados `runs_per_eval` mapas.
        """
        total_successes_population = 0

        # --- LÓGICA DE SELEÇÃO DE MAPAS CORRIGIDA ---
        maps_to_run = []

        # 1. Tentar obter mapas de fases anteriores
        num_previous_maps_desired = runs_per_eval // 2
        if current_stage > 1:
            all_previous_maps = [m for key, maps in map_pool.items() if key < current_stage for m in maps]
            if all_previous_maps:
                num_to_sample = min(num_previous_maps_desired, len(all_previous_maps))
                maps_to_run.extend(random.sample(all_previous_maps, num_to_sample))

        # 2. Calcular quantos mapas faltam e obtê-los da fase atual
        num_current_maps_needed = runs_per_eval - len(maps_to_run)
        current_stage_maps = map_pool.get(current_stage, [])
        if current_stage_maps:
            num_to_sample = min(num_current_maps_needed, len(current_stage_maps))
            maps_to_run.extend(random.sample(current_stage_maps, num_to_sample))
        # --- FIM DA CORREÇÃO ---

        if not maps_to_run:
            print("[AVISO] Nenhum mapa encontrado para avaliação. A saltar a avaliação desta geração.")
            return 0.0

        print(f"A avaliar cada indivíduo em {len(maps_to_run)} mapas...")

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
