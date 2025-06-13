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
        Avalia toda a população de acordo com o novo protocolo.
        Cada indivíduo é testado em `runs_per_eval` mapas.

        Args:
            sim_manager: A instância do SimulationManager.
            current_stage (int): O nível de dificuldade atual.
            map_pool (dict): Um dicionário com os mapas organizados por dificuldade.
            runs_per_eval (int): O número total de mapas para testar (ex: 10).

        Returns:
            float: A média de sucessos de toda a população.
        """
        total_successes_population = 0

        # Define o número de mapas a usar de cada categoria
        num_current_stage_maps = runs_per_eval // 2
        num_previous_stage_maps = runs_per_eval - num_current_stage_maps

        # Selecionar mapas para a fase atual
        current_maps = random.sample(map_pool.get(current_stage, []), num_current_stage_maps)

        # Selecionar mapas de fases anteriores
        previous_maps = []
        if current_stage > 1:
            previous_stage_keys = [key for key in map_pool.keys() if key < current_stage]
            if previous_stage_keys:
                # Cria uma lista de todos os mapas de fases anteriores
                all_previous_maps = [m for key in previous_stage_keys for m in map_pool[key]]
                if len(all_previous_maps) > num_previous_stage_maps:
                    previous_maps = random.sample(all_previous_maps, num_previous_stage_maps)
                else:
                    previous_maps = all_previous_maps  # Usa todos se não houver suficientes

        maps_to_run = current_maps + previous_maps

        if not maps_to_run:
            print("[AVISO] Nenhum mapa encontrado para avaliação. A saltar a avaliação desta geração.")
            return 0.0

        print(
            f"A avaliar cada indivíduo em {len(maps_to_run)} mapas ({len(current_maps)} da fase {current_stage}, {len(previous_maps)} de fases anteriores)...")

        for ind in self.individuals:
            individual_fitness_scores = []
            individual_success_count = 0

            for map_params in maps_to_run:
                stage = map_params['difficulty_level']
                distP, angleP = ind.get_genes()

                # A função run_experiment_with_params retorna (fitness, success_bool)
                fitness, succeeded = sim_manager.run_experiment_with_params(
                    distP, angleP, stage=stage
                )

                individual_fitness_scores.append(fitness)
                if succeeded:
                    individual_success_count += 1

            # A fitness do indivíduo é a média das pontuações de todos os mapas
            ind.fitness = np.mean(individual_fitness_scores) if individual_fitness_scores else 0.0
            ind.total_successes = individual_success_count
            total_successes_population += ind.total_successes

        # Retorna a média de sucessos por indivíduo na população
        return total_successes_population / self.pop_size

    def select_parents(self):
        """Seleciona dois pais usando seleção por torneio."""
        # A seleção por torneio já favorece os indivíduos com maior fitness
        pool = self.individuals

        # Garante que temos pelo menos dois indivíduos para o torneio
        if len(pool) < 2:
            return pool[0], pool[0]

        # Seleciona `tournament_size` indivíduos aleatórios para competir
        contenders = random.sample(pool, min(self.tournament_size, len(pool)))
        contenders.sort(key=lambda ind: ind.fitness, reverse=True)

        return contenders[0], contenders[1] if len(contenders) > 1 else (contenders[0], contenders[0])

    def create_next_generation(self):
        """Cria a próxima geração aplicando elitismo, crossover e mutação."""
        self.individuals.sort(key=lambda ind: ind.fitness, reverse=True)
        next_gen = []

        # Elitismo: os melhores indivíduos passam diretamente para a próxima geração
        for i in range(min(self.elitism, len(self.individuals))):
            elite = self.individuals[i]
            # Cria uma cópia para evitar problemas de referência
            copy = Individual(elite.distP, elite.angleP, id=i)
            copy.fitness = elite.fitness  # Mantém a fitness para referência
            next_gen.append(copy)

        # Preenche o resto da população com descendentes
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

