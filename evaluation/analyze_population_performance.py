import os
import pickle
import sys
import numpy as np
import random
import matplotlib
import time
import pandas as pd
from scipy import stats

matplotlib.use('Agg')  # Use non-interactive backend for batch processing
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
    from environment.tunnel import TunnelBuilder  # ADDED MISSING IMPORT
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


def evaluate_population_performance(supervisor, checkpoint_paths, num_individuals_to_test=30, num_maps_per_stage=10):
    """Avalia fitness médio por fase para múltiplos checkpoints."""
    all_results = {}

    # Carrega mapas uma vez para todas as avaliações
    print("A carregar mapa...")
    map_pool = _load_and_organize_maps()

    # Cria uma única instância de SimulationManager
    print("Criando SimulationManager...")
    sim_mgr = SimulationManager(supervisor)

    for checkpoint_path in checkpoint_paths:
        print(f"\n{'=' * 40}")
        print(f"Processando: {checkpoint_path}")
        print(f"{'=' * 40}")

        try:
            print("A carregar checkpoint...")
            with open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)
            population = data.get('population')
            max_trained_stage = data.get('stage', 0)

            if not population:
                print(f"[ERRO] Não foi encontrada uma população em {checkpoint_path}")
                continue

            all_individuals = population.individuals
            individuals = all_individuals[:min(num_individuals_to_test, len(all_individuals))]

            mode = 'NE' if isinstance(population, NeuralPopulation) else 'GA'
            print(f"População completa de {len(all_individuals)} indivíduos ({mode}) carregada.")
            print(f"-> A avaliar {len(individuals)} indivíduos com {num_maps_per_stage} mapas por fase.")

            # Reinicia a simulação para garantir um estado limpo
            print("Reiniciando simulação...")
            supervisor.simulationReset()
            supervisor.step(sim_mgr.timestep)  # Executa um passo para estabilizar

            # Reset the tunnel builder
            sim_mgr.tunnel_builder = TunnelBuilder(supervisor)

            results = defaultdict(list)
            all_available_stages = sorted(map_pool.keys())
            stages_to_evaluate = [s for s in all_available_stages if s <= max_trained_stage]

            for stage in tqdm(stages_to_evaluate, desc="Avaliando Fases"):
                if not map_pool.get(stage):
                    continue
                maps_to_run = random.sample(map_pool[stage], min(num_maps_per_stage, len(map_pool[stage])))

                for ind in individuals:
                    fitness_scores = []
                    for map_params in maps_to_run:
                        # Reinicia a simulação para cada mapa
                        supervisor.simulationReset()
                        supervisor.step(sim_mgr.timestep)

                        # Reset the tunnel builder
                        sim_mgr.tunnel_builder = TunnelBuilder(supervisor)

                        if mode == 'NE':
                            fitness, _ = sim_mgr.run_experiment_with_network(ind, stage)
                        else:
                            fitness, _ = sim_mgr.run_experiment_with_params(
                                ind.distP, ind.angleP, stage)
                        fitness_scores.append(fitness)

                    avg_fitness = np.mean(fitness_scores) if fitness_scores else 0.0
                    results[stage].append(avg_fitness)

            print("\nAvaliação concluída.")

            # Gera nome único baseado no checkpoint
            checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
            results_filepath = f"evaluation/{checkpoint_name}_fitness_analysis.pkl"
            plot_filename = f"evaluation/{checkpoint_name}_fitness_heatmap.png"

            # Guarda resultados
            plot_data = {
                'results': results,
                'stages': stages_to_evaluate,
                'num_individuals': len(individuals),
                'mode': mode,
                'fitness_matrix': None
            }

            # Cria matriz de fitness para análise estatística
            fitness_matrix = np.zeros((len(individuals), len(stages_to_evaluate)))
            for i, stage in enumerate(stages_to_evaluate):
                if stage in results:
                    fitness_matrix[:, i] = results[stage]

            plot_data['fitness_matrix'] = fitness_matrix

            with open(results_filepath, "wb") as f:
                pickle.dump(plot_data, f)
            print(f"Resultados de fitness guardados em '{results_filepath}'")

            # Gera e guarda heatmap de fitness
            plot_fitness_heatmap(results, stages_to_evaluate, len(individuals), save_path=plot_filename)

            # Guarda para relatório final
            all_results[checkpoint_name] = plot_data

        except Exception as e:
            print(f"Erro ao processar {checkpoint_path}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Garante reset mesmo após erro
            supervisor.simulationReset()
            supervisor.step(sim_mgr.timestep)

    return all_results


def plot_fitness_heatmap(results, stages, num_individuals, save_path=None):
    """Gera um heatmap com fitness médio por fase e indivíduo."""
    fitness_matrix = np.zeros((num_individuals, len(stages)))
    for i, stage in enumerate(stages):
        if results.get(stage):
            fitness_matrix[:, i] = results[stage]

    fig_width = max(15, len(stages) * 1.2)
    fig_height = max(12, num_individuals * 0.6)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Usamos um colormap invertido (quanto mais vermelho, pior; mais verde, melhor)
    cax = ax.matshow(fitness_matrix, cmap='RdYlGn')
    fig.colorbar(cax, label='Fitness Médio')

    ax.set_xticks(np.arange(len(stages)))
    ax.set_yticks(np.arange(num_individuals))
    ax.set_xticklabels([f'Fase {s}' for s in stages])
    ax.set_yticklabels([f'Ind {i}' for i in range(num_individuals)])

    plt.xticks(rotation=45, ha="left", rotation_mode="anchor")
    ax.set_xlabel("Fases de Dificuldade")
    ax.set_ylabel("Indivíduos")
    ax.set_title("Fitness Médio por Fase", pad=20)

    for i in range(num_individuals):
        for j in range(len(stages)):
            ax.text(j, i, f"{fitness_matrix[i, j]:.0f}",
                    ha="center", va="center", color="black", fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Heatmap de fitness guardado em '{save_path}'")
    else:
        plt.show()


def evaluate_reactive_controllers(supervisor, num_maps_per_stage=3):
    """Avalia controladores reativos com análise detalhada."""
    from controllers.reactive_controller import reactive_controller_logic

    reactive_results = {}
    sim_mgr = SimulationManager(supervisor)
    map_pool = _load_and_organize_maps()

    for fov_mode in ['full', 'left', 'right']:
        model_name = f'REACTIVE_{fov_mode.upper()}'
        print(f"\nAvaliando controlador reativo: {model_name}")

        results = {
            'fitness_scores': [],
            'stages': [],
            'success_rates': []
        }

        stages = sorted(map_pool.keys())

        for stage in stages:
            if not map_pool.get(stage):
                continue

            stage_fitness = []
            stage_successes = 0
            maps_to_run = random.sample(map_pool[stage], min(num_maps_per_stage, len(map_pool[stage])))

            for map_params in maps_to_run:
                # Reinicia simulação
                supervisor.simulationReset()
                supervisor.step(sim_mgr.timestep)

                # Executa controlador reativo
                def controller(scan):
                    return reactive_controller_logic(scan, fov_mode=fov_mode)

                # Modificado para capturar detalhes da execução
                run_result = sim_mgr._run_single_episode(controller, stage)
                fitness = run_result['fitness']
                success = run_result['success']

                stage_fitness.append(fitness)
                if success:
                    stage_successes += 1

            avg_fitness = np.mean(stage_fitness)
            success_rate = stage_successes / len(maps_to_run)

            results['fitness_scores'].append(avg_fitness)
            results['success_rates'].append(success_rate)
            results['stages'].append(stage)
            print(f"  Fase {stage}: Fitness médio = {avg_fitness:.0f}, Taxa de sucesso = {success_rate:.0%}")

        reactive_results[model_name] = {
            'mode': 'REACTIVE',
            'fov_mode': fov_mode,
            'reactive_results': results
        }

    return reactive_results


def calculate_cohens_d(data1, data2):
    """Calcula o tamanho do efeito de Cohen's d entre dois conjuntos de dados."""
    n1, n2 = len(data1), len(data2)
    s1, s2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    return abs(np.mean(data1) - np.mean(data2)) / pooled_std


def generate_statistical_summary(all_results):
    """Gera um relatório estatístico comparando todos os modelos."""
    if not all_results:
        return None

    # Prepara dados para análise
    summary_data = []
    all_individual_avgs = []  # Armazenará as médias por indivíduo para cada modelo
    model_names = []

    for name, data in all_results.items():
        # Para modelos populacionais (NE/GA)
        if 'fitness_matrix' in data:
            # Calcula a média por indivíduo (linhas da matriz)
            individual_avgs = np.mean(data['fitness_matrix'], axis=1)
            all_individual_avgs.append(individual_avgs)

            # Ordena indivíduos do melhor para o pior
            sorted_avgs = np.sort(individual_avgs)[::-1]
            num_individuals = len(individual_avgs)

            # Calcula métricas de desempenho
            summary_data.append({
                'Modelo': name,
                'Tipo': data['mode'],
                'Indivíduos': num_individuals,
                'Fases': len(data['stages']),
                'Melhor Indivíduo': np.max(individual_avgs),
                'Pior Indivíduo': np.min(individual_avgs),
                'Média População': np.mean(individual_avgs),
                'Top 25%': np.mean(sorted_avgs[:int(num_individuals * 0.25)]),
                'Consistência (DP)': np.std(individual_avgs),
                'Gap Melhor-Pior': np.max(individual_avgs) - np.min(individual_avgs)
            })
            model_names.append(name)

        # Para modelos reativos
        elif 'reactive_results' in data:
            # Modelos reativos são tratados como população de 1 indivíduo
            fitness_scores = data['reactive_results']['fitness_scores']
            individual_avgs = np.array(fitness_scores)  # Tratado como um único "indivíduo" com múltiplas fases

            summary_data.append({
                'Modelo': name,
                'Tipo': 'REACTIVE',
                'Indivíduos': 1,
                'Fases': len(fitness_scores),
                'Melhor Indivíduo': np.max(fitness_scores),
                'Pior Indivíduo': np.min(fitness_scores),
                'Média População': np.mean(fitness_scores),
                'Top 25%': np.mean(fitness_scores),  # Único valor
                'Consistência (DP)': 0,  # Não há variação
                'Gap Melhor-Pior': 0  # Não há variação
            })
            all_individual_avgs.append(individual_avgs)
            model_names.append(name)

    # Cria DataFrame com estatísticas descritivas
    df_summary = pd.DataFrame(summary_data)

    # Realiza testes estatísticos entre modelos
    statistical_tests = []
    if len(all_results) > 1:
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                # Teste t para amostras independentes
                try:
                    t_stat, p_value = stats.ttest_ind(
                        all_individual_avgs[i], all_individual_avgs[j],
                        equal_var=False
                    )
                except Exception as e:
                    print(f"Erro no teste t entre {model_names[i]} e {model_names[j]}: {e}")
                    continue

                # Tamanho do efeito
                d_value = calculate_cohens_d(
                    all_individual_avgs[i], all_individual_avgs[j]
                )

                # Interpretação do tamanho do efeito
                effect_size = "Pequeno" if d_value < 0.5 else "Médio" if d_value < 0.8 else "Grande"

                # Interpretação da significância
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "NS"

                # Determina modelo superior
                mean_i = np.mean(all_individual_avgs[i])
                mean_j = np.mean(all_individual_avgs[j])
                better_model = model_names[i] if mean_i > mean_j else model_names[j]
                diff = abs(mean_i - mean_j)
                advantage_pct = (diff / max(mean_i, mean_j)) * 100

                statistical_tests.append({
                    'Modelo A': model_names[i],
                    'Modelo B': model_names[j],
                    'Diferença': diff,
                    'Superior': better_model,
                    't': t_stat,
                    'Valor p': p_value,
                    'Significância': significance,
                    'd Cohen': d_value,
                    'Tamanho Efeito': effect_size,
                    'Vantagem': f"{advantage_pct:.1f}%"
                })

    return df_summary, pd.DataFrame(statistical_tests) if statistical_tests else None


def print_statistical_summary(df_summary, df_tests):
    """Imprime um relatório estatístico formatado."""
    print("\n" + "=" * 80)
    print("RESUMO ESTATÍSTICO DE FITNESS".center(80))
    print("=" * 80)

    # Estatísticas descritivas
    print("\n📊 ESTATÍSTICAS DESCRITIVAS POR MODELO:")
    print(df_summary.to_string(index=False, float_format="%.0f"))

    # Testes estatísticos
    if df_tests is not None:
        print("\n\n🔬 COMPARAÇÕES ESTATÍSTICAS ENTRE MODELOS:")
        print(df_tests.to_string(index=False, float_format=lambda x: "%.2f" % x if abs(x) > 1 else "%.3f" % x))

        print("\nLegenda:")
        print("*** p < 0.001 | ** p < 0.01 | * p < 0.05 | NS: Não Significativo")
        print("Tamanho do efeito: d < 0.5 (Pequeno) | 0.5 ≤ d < 0.8 (Médio) | d ≥ 0.8 (Grande)")
        print("Modelo Superior: Modelo com maior fitness médio")
        print("Vantagem: Diferença percentual em relação ao modelo superior")


def main():
    # Cria supervisor global
    supervisor = Supervisor()

    results_dir = "evaluation"
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print("ANÁLISE DE FITNESS DE POPULAÇÕES".center(60))
    print("=" * 60)

    checkpoint_paths = []
    while True:
        path = input("\nIntroduza caminho para checkpoint (.pkl), diretório ou 'feito' para terminar: ").strip()
        if path.lower() in ['feito', 'done', '']:
            break

        if not os.path.exists(path):
            print(f"❌ Caminho não encontrado: {path}")
            continue

        if os.path.isfile(path) and path.endswith('.pkl'):
            checkpoint_paths.append(path)
            print(f"✅ Ficheiro adicionado: {path}")

        elif os.path.isdir(path):
            files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.pkl')]
            if not files:
                print(f"❌ Nenhum ficheiro .pkl encontrado em {path}")
            else:
                checkpoint_paths.extend(files)
                print(f"✅ Adicionados {len(files)} ficheiros de {path}")

    if not checkpoint_paths:
        print("\n❌ Nenhum checkpoint válido fornecido. A sair.")
        return

    print(f"\n▶️ A processar {len(checkpoint_paths)} checkpoint(s)...")
    start_time = time.time()

    # Passa o supervisor para as funções
    all_results = evaluate_population_performance(supervisor, checkpoint_paths)

    # Avalia controladores reativos
    print("\n▶️ Avaliando controladores reativos...")
    reactive_results = evaluate_reactive_controllers(supervisor)
    all_results.update(reactive_results)

    # Gera resumo estatístico
    df_summary, df_tests = generate_statistical_summary(all_results)

    print("\n" + "=" * 60)
    print("RELATÓRIO FINAL".center(60))
    print("=" * 60)

    # Imprime informações básicas sobre cada modelo
    for name, data in all_results.items():
        if 'stages' in data:
            stages = data['stages']
            print(f"\n🔹 {name} ({data.get('mode', 'REACTIVE')}):")
            print(f"   Indivíduos: {data.get('num_individuals', 1)}, Fases: {len(stages)} (0-{max(stages)})")
        elif 'reactive_results' in data:
            print(f"\n🔹 {name} (REACTIVE):")
            print(f"   Fases: {len(data['reactive_results']['stages'])}")

        if 'reactive_results' in data:
            print(f"   Fitness médio: {np.mean(data['reactive_results']['fitness_scores']):.0f}")

    # Imprime resumo estatístico completo
    if df_summary is not None:
        print_statistical_summary(df_summary, df_tests)

    print(f"\n✅ Análise completa em {time.time() - start_time:.1f} segundos")

    # Guarda resultados em CSV
    if df_summary is not None:
        summary_csv = os.path.join(results_dir, "fitness_summary.csv")
        df_summary.to_csv(summary_csv, index=False)
        print(f"\n📝 Resumo de fitness guardado em: {summary_csv}")

        if df_tests is not None:
            tests_csv = os.path.join(results_dir, "fitness_statistical_tests.csv")
            df_tests.to_csv(tests_csv, index=False)
            print(f"📝 Testes estatísticos de fitness guardados em: {tests_csv}")


if __name__ == '__main__':
    main()