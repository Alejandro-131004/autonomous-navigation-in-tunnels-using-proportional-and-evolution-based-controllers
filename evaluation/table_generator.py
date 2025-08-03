import os
import pickle
import sys
import numpy as np
import random
import pandas as pd
from controller import Supervisor
import time

# Configurações
SPECIFIC_STAGES = [0, 4, 8, 12, 13]
NUM_EPISODES_PER_INDIVIDUAL = 1
DEBUG_MODE = False

# Setup do ambiente
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Imports essenciais
try:
    from environment.simulation_manager import SimulationManager
    from environment.tunnel import TunnelBuilder
    from controllers.reactive_controller import reactive_controller_logic
    from curriculum import _load_and_organize_maps
except ImportError as e:
    print(f"[ERRO] Falha ao importar módulos: {e}")
    sys.exit(1)


class PopulationEvaluator:
    def __init__(self, supervisor):
        self.supervisor = supervisor
        self.sim_mgr = SimulationManager(supervisor)
        self.map_pool = _load_and_organize_maps()

    def evaluate_population(self, population, model_name, model_type):
        """Avalia todos os indivíduos de uma população (NE ou GA) usando 5 mapas por estágio"""
        results = {stage: {'success': [], 'progress': [], 'velocity': []}
                   for stage in SPECIFIC_STAGES}

        total_individuals = len(population.individuals)
        start_time = time.time()

        for idx, individual in enumerate(population.individuals):
            print(f"\n=== Avaliando {model_name} [{idx + 1}/{total_individuals}] ===")

            for stage in SPECIFIC_STAGES:
                if stage not in self.map_pool or not self.map_pool[stage]:
                    print(f"[AVISO] Nenhum mapa encontrado para o Estágio {stage}")
                    continue

                # 5 mapas distintos por estágio
                maps_to_run = random.sample(self.map_pool[stage],
                                            min(5, len(self.map_pool[stage])))

                for map_params in maps_to_run:
                    # Reiniciar simulação
                    self.supervisor.simulationReset()
                    self.supervisor.step(self.sim_mgr.timestep)
                    self.sim_mgr.tunnel_builder = TunnelBuilder(self.supervisor)

                    # Executar episódio
                    try:
                        if model_type == 'NE':
                            controller_callable = individual.act
                        else:  # GA
                            distP, angleP = individual.get_genes()
                            controller_callable = lambda scan: self.sim_mgr._process_lidar_for_ga(scan, distP, angleP)

                        episode_result = self.sim_mgr._run_single_episode(controller_callable, stage)
                    except Exception as e:
                        print(f"Erro no episódio: {e}")
                        continue

                    # Calcular métricas
                    success = 1 if episode_result['success'] else 0

                    start_pos = np.array(episode_result.get('start_pos', [0, 0, 0])[:2])
                    end_pos = np.array(episode_result.get('end_pos', [0, 0, 0])[:2])
                    final_pos = np.array(episode_result.get('final_pos', [0, 0, 0])[:2])

                    initial_dist = np.linalg.norm(end_pos - start_pos)
                    final_dist = np.linalg.norm(end_pos - final_pos)

                    if episode_result['success']:
                        progress = 1.0
                    else:
                        progress = max(0, (initial_dist - final_dist) / initial_dist) if initial_dist > 0 else 0

                    total_dist = episode_result.get('total_dist', 0)
                    elapsed_time = episode_result.get('elapsed_time', 1)
                    velocity = total_dist / elapsed_time if elapsed_time > 0 else 0

                    # Armazenar resultados
                    results[stage]['success'].append(success)
                    results[stage]['progress'].append(progress)
                    results[stage]['velocity'].append(velocity)

                    if DEBUG_MODE:
                        print(f"  Estágio {stage}: {'Sucesso' if success else 'Falha'}, "
                              f"Progresso: {progress:.1%}, Velocidade: {velocity:.3f} m/s")

        elapsed = time.time() - start_time
        print(f"Tempo total para {total_individuals} indivíduos: {elapsed:.1f} segundos")
        return {'model': model_name, 'type': model_type, 'results': results}

    def evaluate_ne_population(self, checkpoint_path):
        """Avalia toda a população de um modelo de Neuroevolution"""
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)

        population = data.get('population')
        if not population or not population.individuals:
            raise ValueError("População não encontrada no checkpoint")

        return self.evaluate_population(
            population=population,
            model_name=os.path.basename(checkpoint_path),
            model_type='NE'
        )

    def evaluate_ga_population(self, checkpoint_path):
        """Avalia toda a população de um modelo de Algoritmo Genético"""
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)

        population = data.get('population')
        if not population or not population.individuals:
            raise ValueError("População não encontrada no checkpoint")

        return self.evaluate_population(
            population=population,
            model_name=os.path.basename(checkpoint_path),
            model_type='GA'
        )

    def evaluate_reactive_model(self, fov_mode):
        """Avalia um controlador reativo (1 run por mapa, 5 mapas distintos por estágio)"""
        results = {stage: {'success': [], 'progress': [], 'velocity': []}
                   for stage in SPECIFIC_STAGES}

        def reactive_controller(scan, mode=fov_mode):
            return reactive_controller_logic(scan, fov_mode=mode)

        print(f"\n=== Avaliando REACTIVE_{fov_mode.upper()} ===")
        start_time = time.time()

        for stage in SPECIFIC_STAGES:
            if stage not in self.map_pool or len(self.map_pool[stage]) == 0:
                print(f"[AVISO] Nenhum mapa encontrado para o Estágio {stage}")
                continue

            # 5 mapas diferentes, 1 execução cada
            maps_to_run = random.sample(self.map_pool[stage],
                                        min(1, len(self.map_pool[stage])))

            for map_params in maps_to_run:
                self.supervisor.simulationReset()
                self.supervisor.step(self.sim_mgr.timestep)
                self.sim_mgr.tunnel_builder = TunnelBuilder(self.supervisor)

                try:
                    episode_result = self.sim_mgr._run_single_episode(reactive_controller, stage)
                except Exception as e:
                    print(f"Erro no episódio: {e}")
                    continue

                success = 1 if episode_result['success'] else 0

                start_pos = np.array(episode_result.get('start_pos', [0, 0, 0])[:2])
                end_pos = np.array(episode_result.get('end_pos', [0, 0, 0])[:2])
                final_pos = np.array(episode_result.get('final_pos', [0, 0, 0])[:2])

                initial_dist = np.linalg.norm(end_pos - start_pos)
                final_dist = np.linalg.norm(end_pos - final_pos)

                progress = 1.0 if episode_result['success'] else \
                    max(0, (initial_dist - final_dist) / initial_dist) if initial_dist > 0 else 0

                total_dist = episode_result.get('total_dist', 0)
                elapsed_time = episode_result.get('elapsed_time', 1)
                velocity = total_dist / elapsed_time if elapsed_time > 0 else 0

                results[stage]['success'].append(success)
                results[stage]['progress'].append(progress)
                results[stage]['velocity'].append(velocity)

                if DEBUG_MODE:
                    print(f"  Estágio {stage}: {'Sucesso' if success else 'Falha'}, "
                          f"Progresso: {progress:.1%}, Velocidade: {velocity:.3f} m/s")

        elapsed = time.time() - start_time
        print(f"Tempo total para REACTIVE_{fov_mode.upper()}: {elapsed:.1f} segundos")
        return {'model': f'REACTIVE_{fov_mode.upper()}_FOV',
                'type': 'REACTIVE',
                'results': results}

    def evaluate_reactive_models(self):
        """Avalia todos os modelos reativos"""
        results = []
        for fov_mode in ['full', 'left', 'right']:
            try:
                results.append(self.evaluate_reactive_model(fov_mode))
            except Exception as e:
                print(f"Erro ao avaliar modelo reativo {fov_mode}: {e}")
        return results


def generate_table_ii(all_results):
    """Gera a Tabela II com os resultados consolidados (médias por estágio)"""
    table_data = []

    for result in all_results:
        model_type = result['type']
        model_name = result['model']

        for stage in SPECIFIC_STAGES:
            stage_results = result['results'].get(stage)
            if not stage_results or not stage_results['success']:
                continue

            # Calcular estatísticas sobre todos os indivíduos/execuções
            success_rates = stage_results['success']
            success_rate = np.mean(success_rates) * 100

            progresses = stage_results['progress']
            progress_worst = np.min(progresses) * 100
            progress_avg = np.mean(progresses) * 100
            progress_best = np.max(progresses) * 100
            progress_std = np.std(progresses) * 100

            velocities = stage_results['velocity']
            velocity_worst = np.min(velocities)
            velocity_avg = np.mean(velocities)
            velocity_best = np.max(velocities)
            velocity_std = np.std(velocities)

            # Adicionar linha à tabela
            table_data.append({
                'Model': model_type,
                'Stage': stage,
                'Success Rate (%)': f"{success_rate:.1f}",
                'Worst Progress (%)': f"{progress_worst:.1f}",
                'Avg Progress (%)': f"{progress_avg:.1f}",
                'Best Progress (%)': f"{progress_best:.1f}",
                'Std Progress (%)': f"{progress_std:.1f}",
                'Worst Velocity (m/s)': f"{velocity_worst:.3f}",
                'Avg Velocity (m/s)': f"{velocity_avg:.3f}",
                'Best Velocity (m/s)': f"{velocity_best:.3f}",
                'Std Velocity (m/s)': f"{velocity_std:.3f}"
            })

    return pd.DataFrame(table_data)


def main():
    supervisor = Supervisor()
    evaluator = PopulationEvaluator(supervisor)
    all_results = []

    # Avaliar população NE
    ne_path = input("Caminho para checkpoint NE (.pkl): ").strip()
    if os.path.exists(ne_path):
        try:
            all_results.append(evaluator.evaluate_ne_population(ne_path))
        except Exception as e:
            print(f"Erro ao avaliar população NE: {e}")

    # Avaliar população GA
    ga_path = input("Caminho para checkpoint GA (.pkl): ").strip()
    if os.path.exists(ga_path):
        try:
            all_results.append(evaluator.evaluate_ga_population(ga_path))
        except Exception as e:
            print(f"Erro ao avaliar população GA: {e}")

    # Avaliar modelos reativos
    try:
        all_results.extend(evaluator.evaluate_reactive_models())
    except Exception as e:
        print(f"Erro ao avaliar modelos reativos: {e}")

    # Gerar e salvar tabela
    if all_results:
        table_ii = generate_table_ii(all_results)
        output_file = "Tabela_II_Resultados.csv"
        table_ii.to_csv(output_file, index=False)

        print("\n" + "=" * 80)
        print("TABELA II - RESULTADOS CONSOLIDADOS".center(80))
        print("=" * 80)
        print(table_ii.to_string(index=False))
        print(f"\nTabela salva em: {output_file}")
    else:
        print("\nNenhum resultado foi gerado. Verifique os erros acima.")


if __name__ == '__main__':
    main()