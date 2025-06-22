import os
import pickle
import sys
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

# Adiciona o diretório raiz ao path para permitir imports de outros módulos
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Imports essenciais para carregar os objetos do checkpoint e simulação
try:
    from environment.simulation_manager import SimulationManager
    from optimizer.individualNeural import IndividualNeural
    from optimizer.individual import Individual
    from optimizer.neuralpopulation import NeuralPopulation
    from optimizer.population import Population
    from optimizer.mlpController import MLPController
    from curriculum import _load_and_organize_maps
    # Importar Supervisor aqui para evitar problemas de path
    from controller import Supervisor
except ImportError as e:
    print(f"[ERRO] Falha ao importar módulos necessários: {e}")
    sys.exit(1)


def evaluate_population_performance(checkpoint_path, num_maps_per_stage=5):
    """
    Carrega a população final de um checkpoint e avalia o fitness de cada indivíduo
    em todas as fases de dificuldade disponíveis até à fase máxima treinada.
    """
    # --- CORREÇÃO: Valida se o ficheiro fornecido é um ficheiro .pkl ---
    if not checkpoint_path.lower().endswith('.pkl'):
        print(f"\n[ERRO] Tipo de ficheiro inválido: '{os.path.basename(checkpoint_path)}'")
        print(
            "O caminho fornecido deve apontar para um ficheiro de checkpoint com a extensão .pkl (ex: 'saved_models/ne_checkpoint.pkl').")
        return

    if not os.path.exists(checkpoint_path):
        print(f"[ERRO] Ficheiro de checkpoint não encontrado: '{checkpoint_path}'")
        return

    print("A carregar checkpoint e a inicializar o ambiente de simulação...")
    try:
        # Inicializa o Supervisor do Webots
        supervisor = Supervisor()
        sim_mgr = SimulationManager(supervisor)
        map_pool = _load_and_organize_maps()

        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
        population = data.get('population')

        max_trained_stage = data.get('stage', 0)

        if not population:
            print("[ERRO] Não foi encontrada uma população no ficheiro de checkpoint.")
            return

    except Exception as e:
        print(f"[ERRO] Falha ao inicializar a simulação ou ao carregar o checkpoint: {e}")
        return

    # Extrai indivíduos e determina o modo de avaliação
    individuals = population.individuals
    mode = 'NE' if isinstance(population, NeuralPopulation) else 'GA'
    print(f"População de {len(individuals)} indivíduos ({mode}) carregada. A iniciar avaliação por fase...")

    # Estrutura para guardar os resultados
    results = defaultdict(list)

    # Avalia apenas as fases até ao máximo treinado
    all_available_stages = sorted(map_pool.keys())
    stages_to_evaluate = [s for s in all_available_stages if s <= max_trained_stage]

    expected_stages = set(range(max_trained_stage + 1))
    actual_stages_available = set(stages_to_evaluate)
    missing_stages = sorted(list(expected_stages - actual_stages_available))

    if missing_stages:
        print(
            f"\n[AVISO] O checkpoint indica progresso até à Fase {max_trained_stage}, mas não foram encontrados mapas para as seguintes fases: {missing_stages}.")
        print("Para corrigir, pode apagar a pasta 'evaluation/maps' para forçar a sua regeneração completa.\n")

    if not stages_to_evaluate:
        print("Não foram encontradas fases para avaliar com base no progresso do checkpoint e nos mapas disponíveis.")
        return

    print(f"A avaliar performance nas fases disponíveis: {stages_to_evaluate}")

    # Itera por cada fase e avalia todos os indivíduos
    for stage in tqdm(stages_to_evaluate, desc="A avaliar fases"):
        if not map_pool.get(stage):
            continue
        maps_to_run = random.sample(map_pool[stage], min(num_maps_per_stage, len(map_pool[stage])))

        # Itera por cada indivíduo
        for ind in individuals:
            fitness_scores = []
            for map_params in maps_to_run:
                if mode == 'NE':
                    fitness, _ = sim_mgr.run_experiment_with_network(ind, stage)
                else:  # GA
                    fitness, _ = sim_mgr.run_experiment_with_params(ind.distP, ind.angleP, stage)
                fitness_scores.append(fitness)

            # Guarda a média de fitness do indivíduo para esta fase
            avg_fitness = np.mean(fitness_scores) if fitness_scores else 0
            results[stage].append(avg_fitness)

    print("Avaliação concluída. A gerar o gráfico...")
    plot_performance_heatmap(results, stages_to_evaluate, len(individuals))


def plot_performance_heatmap(results, stages, num_individuals):
    """
    Cria um heatmap da performance (fitness) de cada indivíduo por fase.
    """
    # Converte os resultados para uma matriz 2D para o heatmap
    performance_matrix = np.zeros((num_individuals, len(stages)))
    for i, stage in enumerate(stages):
        if results[stage]:
            performance_matrix[:, i] = results[stage]

    fig, ax = plt.subplots(figsize=(max(12, len(stages) * 0.8), max(8, num_individuals * 0.35)))

    # Usa um mapa de cores divergente para distinguir facilmente performance boa (verde), média (amarelo) e má (vermelho)
    cax = ax.matshow(performance_matrix, cmap='RdYlGn')
    fig.colorbar(cax, label='Fitness Score Médio')

    # Configurações do gráfico
    ax.set_xticks(np.arange(len(stages)))
    ax.set_yticks(np.arange(num_individuals))
    ax.set_xticklabels([f'Fase {s}' for s in stages])
    ax.set_yticklabels([f'Indivíduo {i}' for i in range(num_individuals)])

    # Roda os labels do eixo X para melhor leitura se houver muitas fases
    plt.xticks(rotation=45, ha="left", rotation_mode="anchor")

    ax.set_xlabel("Fases de Dificuldade")
    ax.set_ylabel("Indivíduos na População")
    ax.set_title("Heatmap de Performance da População Final por Fase", pad=20)

    # Adiciona os valores de fitness dentro de cada célula do heatmap
    for i in range(num_individuals):
        for j in range(len(stages)):
            ax.text(j, i, f"{performance_matrix[i, j]:.0f}",
                    ha="center", va="center", color="black", fontsize=8)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    try:
        if len(sys.argv) > 1:
            checkpoint_path = sys.argv[1]
        else:
            checkpoint_path = input(
                "Por favor, introduza o caminho para o ficheiro de checkpoint (.pkl) que deseja analisar: > ")

        evaluate_population_performance(checkpoint_path)

    except KeyboardInterrupt:
        print("\nAnálise interrompida pelo utilizador.")
    except Exception as e:
        print(f"\nOcorreu um erro inesperado: {e}")
