import os
import pickle
import numpy as np
import random
from collections import defaultdict

from environment.configuration import get_stage_parameters
from environment.simulation_manager import SimulationManager
from evaluation.map_generator import generate_maps

# Imports necessários para que o pickle consiga carregar os objetos do checkpoint
from optimizer.neuralpopulation import NeuralPopulation
from optimizer.population import Population
from optimizer.individualNeural import IndividualNeural
from optimizer.individual import Individual
from optimizer.mlpController import MLPController


def _load_and_organize_maps(maps_dir="evaluation/maps", num_maps_per_diff=100):
    """
    Gera mapas se o diretório não existir e organiza-os por dificuldade.
    """
    from environment.configuration import MAX_DIFFICULTY_STAGE as total_stages_for_gen
    if not os.path.exists(maps_dir) or not os.listdir(maps_dir):
        print(
            f"Diretório de mapas '{maps_dir}' não encontrado ou vazio. A gerar {num_maps_per_diff * (total_stages_for_gen + 1)} novos mapas..."
        )
        generate_maps(maps_output_dir=maps_dir,
                      num_maps_per_difficulty=num_maps_per_diff,
                      total_difficulty_stages=total_stages_for_gen)

    map_pool = defaultdict(list)
    for filename in os.listdir(maps_dir):
        if filename.endswith(".pkl"):
            filepath = os.path.join(maps_dir, filename)
            try:
                with open(filepath, 'rb') as f:
                    map_params = pickle.load(f)
                    difficulty = map_params['difficulty_level']
                    map_pool[difficulty].append(map_params)
            except Exception as e:
                print(f"[AVISO] Não foi possível carregar o mapa {filepath}: {e}")

    print(f"Mapas carregados. {sum(len(v) for v in map_pool.values())} mapas em {len(map_pool)} níveis de dificuldade.")
    return dict(map_pool)


def run_unified_curriculum(supervisor, config: dict):
    from environment.configuration import STAGE_DEFINITIONS, generate_intermediate_stage, MAX_DIFFICULTY_STAGE

    mode = config['mode']
    sim_mgr = SimulationManager(supervisor)
    map_pool = _load_and_organize_maps(num_maps_per_diff=100)
    checkpoint_file = config['checkpoint_file']

    def _save_checkpoint(data):
        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(data, f)

    def _load_checkpoint():
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'rb') as f:
                    return pickle.load(f)
            except (pickle.UnpicklingError, EOFError) as e:
                print(f"[ERRO] Falha ao ler o ficheiro de checkpoint: {e}. A começar um novo treino.")
        return None

    # --- Lógica de Inicialização e Carregamento de Checkpoint ---
    population = None
    best_overall_individual = None
    start_stage = 0
    history = []

    if config.get('resume_training', False):
        data = _load_checkpoint()
        if data:
            population = data.get('population')
            best_overall_individual = data.get('best_individual')
            saved_stage = data.get('stage', 0)
            history = data.get('history', [])
            print(f"\nCheckpoint carregado. A última sessão terminou na Fase {saved_stage}.")

            while True:
                override_choice = input(
                    f"Pressione 'c' para continuar da Fase {saved_stage}, ou 's' para selecionar uma fase de início diferente: [c/s] ").lower().strip()
                if override_choice == 'c':
                    start_stage = saved_stage
                    break
                elif override_choice == 's':
                    while True:
                        try:
                            new_stage_input = input(
                                f"Insira a fase de onde pretende recomeçar (0-{MAX_DIFFICULTY_STAGE}): ")
                            new_stage = int(new_stage_input)
                            if 0 <= new_stage <= MAX_DIFFICULTY_STAGE:
                                start_stage = new_stage
                                # Filtra o histórico para manter apenas as entradas de fases anteriores à nova fase de início.
                                history_before_restart = [entry for entry in history if entry['stage'] < start_stage]
                                if history_before_restart:
                                    last_gen = history_before_restart[-1]['generation']
                                    print(
                                        f"Histórico anterior mantido. O novo treino começará na geração {last_gen + 1} (Fase {start_stage}).")
                                else:
                                    print(
                                        f"Histórico reiniciado. O novo treino começará na geração 1 (Fase {start_stage}).")
                                history = history_before_restart
                                break
                            else:
                                print(f"Fase inválida. Por favor, insira um número entre 0 e {MAX_DIFFICULTY_STAGE}.")
                        except ValueError:
                            print("Input inválido. Por favor, insira um número.")
                    break
                else:
                    print("Opção inválida. Por favor, insira 'c' ou 's'.")
        else:
            # Se o checkpoint não puder ser carregado, força um novo início.
            config['resume_training'] = False

    if population is None:
        print("A iniciar uma nova população...")
        if mode == 'NE':
            input_size = len(np.nan_to_num(sim_mgr.lidar.getRangeImage(), nan=0.0))
            output_size = 2
            population = NeuralPopulation(
                pop_size=config['pop_size'], input_size=input_size,
                hidden_size=config['hidden_size'], output_size=output_size,
                mutation_rate=config['mutation_rate'], elitism=config['elitism']
            )
        else:
            population = Population(pop_size=config['pop_size'], mutation_rate=config['mutation_rate'],
                                    elitism=config['elitism'])
        start_stage = 0
        history = []

    current_stage = start_stage
    threshold_prev = config.get('threshold_prev', 0.7)
    threshold_curr = config.get('threshold_curr', 0.7)
    sub_index = 0
    attempts_without_progress = 0

    try:
        while True:
            print(f"\n\n{'=' * 20} A INICIAR FASE DE DIFICULDADE {current_stage} {'=' * 20}")
            attempts_in_stage = 0

            while True:
                attempts_in_stage += 1
                # A geração continua a partir do final do histórico (completo ou truncado)
                generation_id = len(history) + 1
                print(f"\n--- Geração {generation_id} (Fase {current_stage}, Tentativa {attempts_in_stage}) ---")

                # Seleção de mapas para avaliação
                runs_prev = 5
                runs_curr = 5
                available_prev_stages = [s for s in map_pool if s < current_stage]
                maps_prev = [random.choice(map_pool[stage]) for stage in
                             random.sample(available_prev_stages, min(runs_prev, len(available_prev_stages)))]
                maps_curr = random.sample(map_pool.get(current_stage, []),
                                          k=min(runs_curr, len(map_pool.get(current_stage, []))))

                if not maps_curr:
                    print(f"[AVISO] Não foram encontrados mapas para a Fase {current_stage}. A terminar o treino.")
                    return best_overall_individual

                # Avaliação da população
                avg_succ_prev = population.evaluate(sim_mgr, maps_prev) if maps_prev else 0
                avg_succ_curr = population.evaluate(sim_mgr, maps_curr)

                rate_prev = avg_succ_prev / runs_prev if runs_prev > 0 and maps_prev else None
                rate_curr = avg_succ_curr / runs_curr if runs_curr > 0 else 0

                fitness_values = [ind.fitness for ind in population.individuals]
                generation_stats = {
                    'stage': current_stage, 'generation': generation_id,
                    'fitness_min': min(fitness_values), 'fitness_avg': np.mean(fitness_values),
                    'fitness_max': max(fitness_values),
                    'success_rate_prev': rate_prev if rate_prev is not None else 0,
                    'success_rate_curr': rate_curr
                }
                history.append(generation_stats)

                print("-" * 50)
                print(
                    f"  FITNESS -> Min: {generation_stats['fitness_min']:.2f} | Avg: {generation_stats['fitness_avg']:.2f} | Max: {generation_stats['fitness_max']:.2f}")
                print(f"  SUCESSO -> Prev: {rate_prev:.2% if rate_prev is not None else 'N/A'} | Curr: {rate_curr:.2%}")
                print("-" * 50)

                gen_best = population.get_best_individual()
                if gen_best and (best_overall_individual is None or gen_best.fitness > best_overall_individual.fitness):
                    best_overall_individual = gen_best
                    prefix = "ne_best" if mode == 'NE' else "ga_best"
                    sim_mgr.save_model(gen_best, filename=f"{prefix}_stage_{current_stage}_gen_{generation_id}.pkl")

                _save_checkpoint(
                    {'population': population, 'best_individual': best_overall_individual, 'stage': current_stage,
                     'history': history})

                if (rate_prev is None or rate_prev >= threshold_prev) and rate_curr >= threshold_curr:
                    print(f"[PROGRESSO] Limiares alcançados. A avançar para a próxima fase.")
                    current_stage += 1
                    sub_index, attempts_without_progress = 0, 0
                    break
                else:
                    attempts_without_progress += 1
                    print(f"[REPETIR] Limiares não alcançados. Tentativa {attempts_without_progress}...")

                    if attempts_without_progress >= 50:
                        sub_index += 1
                        base_params = STAGE_DEFINITIONS[current_stage]
                        custom_stage_params = generate_intermediate_stage(base_params, sub_index=sub_index)
                        print(f"\nApós 50 tentativas, a criar sub-fase intermédia {sub_index}...")
                        print(f"-> Parâmetros ajustados: {custom_stage_params}")
                        attempts_without_progress = 0

                    population.create_next_generation()

    except KeyboardInterrupt:
        print("\n[INTERROMPIDO] A guardar o último estado...")
    finally:
        _save_checkpoint({'population': population, 'best_individual': best_overall_individual, 'stage': current_stage,
                          'history': history})
        print("Sessão de treino terminada.")

    return best_overall_individual
