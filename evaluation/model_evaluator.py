# model_evaluator.py
"""
Módulo universal para avaliar e comparar múltiplos tipos de controladores.
"""
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict

from controller import Supervisor
from environment.simulation_manager import SimulationManager
from optimizer.individualNeural import IndividualNeural


def _metric_factory():
    return {'fitness': [], 'success': []}


def _difficulty_factory():
    return defaultdict(_metric_factory)


def evaluate_controllers(
        supervisor: Supervisor,
        controllers_to_test: list,
        map_files: list,
        results_output_dir: str = "evaluation/results"
):
    """
    Avalia uma lista de controladores (funções, ficheiros de parâmetros ou redes neuronais).
    """
    os.makedirs(results_output_dir, exist_ok=True)
    sim_mgr = SimulationManager(supervisor)
    all_results = defaultdict(_difficulty_factory)

    maps_by_difficulty = defaultdict(list)
    for f_path in map_files:
        with open(f_path, 'rb') as f:
            maps_by_difficulty[pickle.load(f)['difficulty_level']].append(f_path)

    for controller_info in controllers_to_test:
        controller_name = controller_info['name']
        print(f"\n===== AVALIANDO CONTROLADOR: {controller_name} =====")

        controller_callable = None
        if controller_info['type'] == 'function':
            controller_callable = controller_info['callable']

        elif controller_info['type'] == 'neural_network':
            try:
                with open(controller_info['path'], 'rb') as f:
                    model = pickle.load(f)
                if isinstance(model, IndividualNeural):
                    controller_callable = model.act
                else:
                    print(f"[ERRO] Ficheiro '{controller_info['path']}' não é um IndividualNeural. A ignorar.")
                    continue
            except Exception as e:
                print(f"[ERRO] Falha ao carregar modelo de '{controller_info['path']}': {e}. A ignorar.")
                continue

        elif controller_info['type'] == 'ga_params':
            try:
                with open(controller_info['path'], 'rb') as f:
                    params = pickle.load(f)
                distP, angleP = params['distP'], params['angleP']
                # Cria uma função que chama o controlador do sim_mgr com os parâmetros carregados
                controller_callable = lambda scan: sim_mgr._process_lidar_for_ga(scan, distP, angleP)
            except Exception as e:
                print(f"[ERRO] Falha ao carregar parâmetros de '{controller_info['path']}': {e}. A ignorar.")
                continue

        if not controller_callable:
            continue

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

    _generate_comparison_report(all_results, results_output_dir)


def _generate_comparison_report(all_results, output_dir):
    """Gera gráficos comparativos para todos os controladores avaliados."""
    print("\n--- Gerando Relatório de Avaliação Comparativa ---")

    plt.style.use('seaborn-v0_8-whitegrid')

    # Gráfico de Fitness
    plt.figure(figsize=(12, 8))
    for name, results_by_diff in all_results.items():
        difficulties = sorted(results_by_diff.keys())
        avg_fitness = [np.mean(results_by_diff[d]['fitness']) for d in difficulties]
        plt.plot(difficulties, avg_fitness, marker='o', linestyle='-', label=name)

    plt.title('Comparação de Fitness Médio vs. Dificuldade', fontsize=16)
    plt.xlabel('Nível de Dificuldade', fontsize=12)
    plt.ylabel('Fitness Médio', fontsize=12)
    plt.xticks(difficulties)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparacao_fitness.png"))
    plt.close()
    print("Gráfico de comparação de fitness guardado.")

    # Gráfico de Taxa de Sucesso
    plt.figure(figsize=(12, 8))
    for name, results_by_diff in all_results.items():
        difficulties = sorted(results_by_diff.keys())
        success_rate = [np.mean(results_by_diff[d]['success']) * 100 for d in difficulties]
        plt.plot(difficulties, success_rate, marker='o', linestyle='-', label=name)

    plt.title('Comparação de Taxa de Sucesso vs. Dificuldade', fontsize=16)
    plt.xlabel('Nível de Dificuldade', fontsize=12)
    plt.ylabel('Taxa de Sucesso (%)', fontsize=12)
    plt.xticks(difficulties)
    plt.ylim(0, 105)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparacao_sucesso.png"))
    plt.close()
    print("Gráfico de comparação de sucesso guardado.")

    with open(os.path.join(output_dir, "full_evaluation_results.pkl"), 'wb') as f:
        pickle.dump(dict(all_results), f)
    print("Resultados detalhados da avaliação guardados.")
