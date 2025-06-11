# model_evaluator.py
"""
Módulo universal para avaliar e comparar o desempenho de múltiplos
controladores (reativos e de redes neuronais) em mapas gerados aleatoriamente.
"""
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict

from controller import Supervisor
from environment.simulation_manager import SimulationManager
from optimizer.individualNeural import IndividualNeural


# --- Funções Auxiliares para defaultdict (Seguras para Pickle) ---
def _metric_factory():
    """Retorna um dicionário padrão para as métricas de um episódio."""
    return {'fitness': [], 'success': []}


def _difficulty_factory():
    """Retorna uma defaultdict para agrupar métricas por dificuldade."""
    return defaultdict(_metric_factory)


# -----------------------------------------------------------------

def evaluate_controllers(
        supervisor: Supervisor,
        controllers_to_test: list,
        map_files: list,
        results_output_dir: str = "evaluation/results"
):
    """
    Avalia uma lista de controladores (funções ou ficheiros .pkl) nos mapas fornecidos.
    """
    os.makedirs(results_output_dir, exist_ok=True)
    sim_mgr = SimulationManager(supervisor)

    # CORREÇÃO: Usa a função _difficulty_factory em vez de lambdas para ser compatível com pickle.
    all_results = defaultdict(_difficulty_factory)

    # Itera sobre cada controlador a ser testado
    for controller_info in controllers_to_test:
        controller_name = controller_info['name']
        print(f"\n===== AVALIANDO CONTROLADOR: {controller_name} =====")

        # Obtém a função "callable" do controlador
        controller_callable = None
        if controller_info['type'] == 'function':
            controller_callable = controller_info['callable']
        elif controller_info['type'] == 'file':
            try:
                with open(controller_info['path'], 'rb') as f:
                    model = pickle.load(f)
                if isinstance(model, IndividualNeural):
                    controller_callable = model.act
                else:
                    print(f"[ERRO] Ficheiro '{controller_info['path']}' não é um IndividualNeural válido. A ignorar.")
                    continue
            except Exception as e:
                print(f"[ERRO] Falha ao carregar o modelo de '{controller_info['path']}': {e}. A ignorar.")
                continue

        if not controller_callable:
            continue

        # Agrupa os mapas por dificuldade
        maps_by_difficulty = defaultdict(list)
        for f_path in map_files:
            with open(f_path, 'rb') as f:
                maps_by_difficulty[pickle.load(f)['difficulty_level']].append(f_path)

        # Avalia o controlador em todas as dificuldades
        for difficulty, maps in sorted(maps_by_difficulty.items()):
            print(f"  --- A testar Dificuldade {difficulty} ({len(maps)} mapas) ---")
            for map_filepath in maps:
                results = sim_mgr._run_single_episode(
                    controller_callable=controller_callable,
                    stage=difficulty,
                    total_stages=len(maps_by_difficulty)
                )
                all_results[controller_name][difficulty]['fitness'].append(results['fitness'])
                all_results[controller_name][difficulty]['success'].append(1 if results['success'] else 0)

    # Gera o relatório comparativo
    _generate_comparison_report(all_results, results_output_dir)


def _generate_comparison_report(all_results, output_dir):
    """Gera gráficos comparativos para todos os controladores avaliados."""
    print("\n--- Gerando Relatório de Avaliação Comparativa ---")

    # Gráfico de Fitness
    plt.figure(figsize=(12, 7))
    difficulties_for_plot = []
    for name, results_by_diff in all_results.items():
        difficulties = sorted(results_by_diff.keys())
        if not difficulties_for_plot:
            difficulties_for_plot = difficulties
        avg_fitness = [np.mean(results_by_diff[d]['fitness']) for d in difficulties]
        plt.plot(difficulties, avg_fitness, marker='o', linestyle='-', label=name)

    plt.title('Comparação de Fitness Médio vs. Dificuldade')
    plt.xlabel('Nível de Dificuldade')
    plt.ylabel('Fitness Médio')
    if difficulties_for_plot:
        plt.xticks(difficulties_for_plot)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "comparacao_fitness.png"))
    plt.close()
    print("Gráfico de comparação de fitness guardado.")

    # Gráfico de Taxa de Sucesso
    plt.figure(figsize=(12, 7))
    for name, results_by_diff in all_results.items():
        difficulties = sorted(results_by_diff.keys())
        success_rate = [np.mean(results_by_diff[d]['success']) * 100 for d in difficulties]
        plt.plot(difficulties, success_rate, marker='o', linestyle='-', label=name)

    plt.title('Comparação de Taxa de Sucesso vs. Dificuldade')
    plt.xlabel('Nível de Dificuldade')
    plt.ylabel('Taxa de Sucesso (%)')
    if difficulties_for_plot:
        plt.xticks(difficulties_for_plot)
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "comparacao_sucesso.png"))
    plt.close()
    print("Gráfico de comparação de sucesso guardado.")

    # Guardar dados brutos
    with open(os.path.join(output_dir, "full_evaluation_results.pkl"), 'wb') as f:
        # Converte a defaultdict para um dict normal antes de guardar
        pickle.dump(dict(all_results), f)
    print("Resultados detalhados da avaliação guardados.")
