import os
import pickle
import sys
import numpy as np
import random
import matplotlib

# Força o uso de um backend de UI compatível (TkAgg)
matplotlib.use('TkAgg')

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
    from controller import Supervisor
except ImportError as e:
    print(f"[ERRO] Falha ao importar módulos necessários: {e}")
    sys.exit(1)


def evaluate_population_performance(checkpoint_path, num_individuals_to_test=20, num_maps_per_stage=3):
    """
    Carrega um subconjunto da população final de um checkpoint, avalia a sua aptidão
    nas fases treinadas, guarda os resultados e, em seguida, desenha-os.
    """
    print("A carregar checkpoint e a inicializar o ambiente de simulação...")
    try:
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

    all_individuals = population.individuals

    # Seleciona um subconjunto da população para testar para uma avaliação mais rápida
    individuals = all_individuals[:min(num_individuals_to_test, len(all_individuals))]

    mode = 'NE' if isinstance(population, NeuralPopulation) else 'GA'
    print(f"População completa de {len(all_individuals)} indivíduos ({mode}) carregada.")
    print(f"-> Para uma análise mais rápida, a testar a performance dos primeiros {len(individuals)} indivíduos.")

    results = defaultdict(list)
    all_available_stages = sorted(map_pool.keys())
    stages_to_evaluate = [s for s in all_available_stages if s <= max_trained_stage]

    print(f"A avaliar performance até à Fase {max_trained_stage} usando {num_maps_per_stage} mapas por fase.")

    for stage in tqdm(stages_to_evaluate, desc="A avaliar Fases"):
        if not map_pool.get(stage):
            continue
        maps_to_run = random.sample(map_pool[stage], min(num_maps_per_stage, len(map_pool[stage])))

        for ind in individuals:
            fitness_scores = []
            for map_params in maps_to_run:
                if mode == 'NE':
                    fitness, _ = sim_mgr.run_experiment_with_network(ind, stage)
                else:  # GA
                    fitness, _ = sim_mgr.run_experiment_with_params(ind.distP, ind.angleP, stage)
                fitness_scores.append(fitness)

            avg_fitness = np.mean(fitness_scores) if fitness_scores else 0
            results[stage].append(avg_fitness)

    print("\nAvaliação concluída.")

    results_filepath = "evaluation/analysis_results.pkl"
    print(f"A guardar os resultados da avaliação para '{results_filepath}'...")
    try:
        plot_data = {
            'results': results,
            'stages': stages_to_evaluate,
            'num_individuals': len(individuals)
        }
        with open(results_filepath, "wb") as f:
            pickle.dump(plot_data, f)
        print("Resultados guardados com sucesso.")
    except Exception as e:
        print(f"[ERRO] Não foi possível guardar os resultados da avaliação: {e}")

    plot_performance_heatmap(results, stages_to_evaluate, len(individuals))


def plot_performance_heatmap(results, stages, num_individuals):
    """
    Cria um heatmap da performance (fitness) de cada indivíduo por fase.
    """
    performance_matrix = np.zeros((num_individuals, len(stages)))
    for i, stage in enumerate(stages):
        if results[stage]:
            performance_matrix[:, i] = results[stage]

    # --- ALTERAÇÃO: Aumenta o tamanho do gráfico para melhor legibilidade ---
    # Ajusta o tamanho da figura dinamicamente com base na quantidade de dados.
    fig_width = max(15, len(stages) * 1.2)
    fig_height = max(12, num_individuals * 0.6)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    cax = ax.matshow(performance_matrix, cmap='RdYlGn')
    fig.colorbar(cax, label='Average Fitness Score')

    ax.set_xticks(np.arange(len(stages)))
    ax.set_yticks(np.arange(num_individuals))
    ax.set_xticklabels([f'Stage {s}' for s in stages])
    ax.set_yticklabels([f'Individual {i}' for i in range(num_individuals)])

    plt.xticks(rotation=45, ha="left", rotation_mode="anchor")

    ax.set_xlabel("Difficulty Stages")
    ax.set_ylabel("Individuals in Population")
    ax.set_title("Final Population Performance Heatmap by Stage", pad=20)

    # Adiciona os valores de fitness dentro de cada célula
    for i in range(num_individuals):
        for j in range(len(stages)):
            # --- ALTERAÇÃO: Aumenta ligeiramente o tamanho da fonte do texto ---
            ax.text(j, i, f"{performance_matrix[i, j]:.0f}",
                    ha="center", va="center", color="black", fontsize=10)

    plt.tight_layout()
    plt.show()


def main():
    """
    Função principal. Verifica se existem resultados pré-calculados para evitar reexecutar a avaliação.
    """
    results_filepath = "evaluation/analysis_results.pkl"

    if os.path.exists(results_filepath):
        while True:
            choice = input(
                "Foram encontrados resultados de avaliação existentes. Desenhar o gráfico a partir destes resultados [p] ou executar uma [n]ova avaliação? [p/n]: ").lower().strip()
            if choice == 'p':
                print("A carregar resultados existentes...")
                try:
                    with open(results_filepath, "rb") as f:
                        plot_data = pickle.load(f)
                    print("A gerar o gráfico a partir dos dados guardados...")
                    plot_performance_heatmap(
                        plot_data['results'],
                        plot_data['stages'],
                        plot_data['num_individuals']
                    )
                    return  # Sai após desenhar o gráfico
                except Exception as e:
                    print(f"[ERRO] Não foi possível carregar ou desenhar a partir do ficheiro de resultados: {e}")
                    return
            elif choice == 'n':
                print("A prosseguir com uma nova avaliação...")
                break
            else:
                print("Opção inválida. Por favor, insira 'p' ou 'n'.")

    try:
        if len(sys.argv) > 1:
            checkpoint_path = sys.argv[1]
        else:
            checkpoint_path = input(
                "Por favor, introduza o caminho para o ficheiro de checkpoint (.pkl) que deseja analisar: > ")

        evaluate_population_performance()

    except KeyboardInterrupt:
        print("\nAnálise interrompida pelo utilizador.")
    except Exception as e:
        print(f"\nOcorreu um erro inesperado: {e}")


if __name__ == '__main__':
    main()
