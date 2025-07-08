import os
import pickle
import sys
import numpy as np
import random
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

# Adiciona o diretório raiz ao path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Imports para carregar os objetos dos checkpoints
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
    Avalia tempo médio de conclusão por fase para indivíduos de um checkpoint.
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
    individuals = all_individuals[:min(num_individuals_to_test, len(all_individuals))]

    mode = 'NE' if isinstance(population, NeuralPopulation) else 'GA'
    print(f"População completa de {len(all_individuals)} indivíduos ({mode}) carregada.")
    print(f"-> A avaliar {len(individuals)} indivíduos com {num_maps_per_stage} mapas por fase.")

    results = defaultdict(list)
    all_available_stages = sorted(map_pool.keys())
    stages_to_evaluate = [s for s in all_available_stages if s <= max_trained_stage]

    for stage in tqdm(stages_to_evaluate, desc="A avaliar Fases"):
        if not map_pool.get(stage):
            continue
        maps_to_run = random.sample(map_pool[stage], min(num_maps_per_stage, len(map_pool[stage])))

        for ind in individuals:
            times = []
            for map_params in maps_to_run:
                if mode == 'NE':
                    fitness, success, elapsed = sim_mgr.run_experiment_with_network(ind, stage, return_time=True)
                else:
                    fitness, success, elapsed = sim_mgr.run_experiment_with_params(
                        ind.distP, ind.angleP, stage, return_time=True)

                times.append(elapsed)

            avg_time = np.mean(times) if times else 0.0
            results[stage].append(avg_time)

    print("\nAvaliação concluída.")
    results_filepath = "evaluation/analysis_results.pkl"
    print(f"A guardar os resultados da avaliação em '{results_filepath}'...")

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
        print(f"[ERRO] Não foi possível guardar os resultados: {e}")

    plot_performance_heatmap(results, stages_to_evaluate, len(individuals))


def plot_performance_heatmap(results, stages, num_individuals):
    """
    Gera um heatmap com tempo médio (s) por fase e indivíduo.
    """
    performance_matrix = np.zeros((num_individuals, len(stages)))
    for i, stage in enumerate(stages):
        if results[stage]:
            performance_matrix[:, i] = results[stage]

    fig_width = max(15, len(stages) * 1.2)
    fig_height = max(12, num_individuals * 0.6)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    cax = ax.matshow(performance_matrix, cmap='YlGnBu')
    fig.colorbar(cax, label='Average Completion Time (s)')

    ax.set_xticks(np.arange(len(stages)))
    ax.set_yticks(np.arange(num_individuals))
    ax.set_xticklabels([f'Stage {s}' for s in stages])
    ax.set_yticklabels([f'Ind {i}' for i in range(num_individuals)])

    plt.xticks(rotation=45, ha="left", rotation_mode="anchor")
    ax.set_xlabel("Difficulty Stages")
    ax.set_ylabel("Individuals")
    ax.set_title("Average Completion Time per Stage", pad=20)

    for i in range(num_individuals):
        for j in range(len(stages)):
            ax.text(j, i, f"{performance_matrix[i, j]:.1f}",
                    ha="center", va="center", color="black", fontsize=10)

    plt.tight_layout()
    plt.show()


def main():
    results_filepath = "evaluation/analysis_results.pkl"

    if os.path.exists(results_filepath):
        choice = input("Resultados anteriores encontrados. Mostrar [p] ou nova avaliação [n]? [p/n]: ").lower().strip()
        if choice == 'p':
            try:
                with open(results_filepath, "rb") as f:
                    plot_data = pickle.load(f)
                plot_performance_heatmap(
                    plot_data['results'],
                    plot_data['stages'],
                    plot_data['num_individuals']
                )
                return
            except Exception as e:
                print(f"[ERRO] Não foi possível carregar resultados anteriores: {e}")
                return
        elif choice != 'n':
            print("Opção inválida. Saindo.")
            return

    try:
        checkpoint_path = input("Introduza o caminho para o checkpoint (.pkl): > ")
        evaluate_population_performance(checkpoint_path)
    except KeyboardInterrupt:
        print("\nAnálise interrompida.")
    except Exception as e:
        print(f"\nErro inesperado: {e}")


if __name__ == '__main__':
    main()
