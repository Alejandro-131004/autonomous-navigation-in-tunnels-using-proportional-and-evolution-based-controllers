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


def _re_evaluate_past_stages(population, sim_mgr, map_pool, up_to_stage, threshold):
    """
    Reavalia a performance da população atual em todas as fases anteriores.
    Retorna uma lista de fases que não cumprem o threshold de sucesso.
    """
    print("\n" + "=" * 20 + " A INICIAR REAVALIAÇÃO DE FASES ANTERIORES " + "=" * 20)
    retraining_queue = []

    for stage in range(up_to_stage):
        if stage not in map_pool:
            continue

        print(f"A reavaliar Fase {stage}...")
        num_maps_to_run = 10  # Usar um número razoável de mapas para a reavaliação
        maps_to_run = random.sample(map_pool[stage], min(num_maps_to_run, len(map_pool[stage])))

        if not maps_to_run:
            continue

        avg_success = population.evaluate(sim_mgr, maps_to_run)
        success_rate = avg_success

        result = "OK"
        if success_rate < threshold:
            result = "FALHOU"
            retraining_queue.append(stage)

        print(f"--> Resultado Fase {stage}: Taxa de Sucesso = {success_rate:.2%} ({result})")

    if not retraining_queue:
        print("\nReavaliação concluída. Todas as fases anteriores cumprem os requisitos.")
    else:
        print(f"\nReavaliação concluída. Fila de retreino: {retraining_queue}")

    return retraining_queue


def run_unified_curriculum(supervisor, config: dict):
    from environment.configuration import STAGE_DEFINITIONS, generate_intermediate_stage, MAX_DIFFICULTY_STAGE

    mode = config['mode']
    sim_mgr = SimulationManager(supervisor)
    map_pool = _load_and_organize_maps(num_maps_per_diff=100)
    checkpoint_file = config['checkpoint_file']

    retraining_queue = []

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

            # --- NOVA LÓGICA DE REFRESH/RETREINO ---
            while True:
                refresh_choice = input(
                    "Deseja fazer um 'Refresh Training' para reavaliar fases anteriores? [s/n]: ").lower().strip()
                if refresh_choice in ['s', 'n']:
                    break
                print("Opção inválida.")

            if refresh_choice == 's':
                # Reavalia o desempenho e cria a fila de retreino
                retraining_queue = _re_evaluate_past_stages(population, sim_mgr, map_pool, up_to_stage=saved_stage,
                                                            threshold=config['threshold_prev'])

            # Pergunta sobre a fase de continuação normal
            while True:
                override_choice = input(
                    f"Pressione 'c' para continuar da Fase {saved_stage}, ou 's' para selecionar uma fase de início diferente: [c/s] ").lower().strip()
                if override_choice == 'c':
                    start_stage = saved_stage
                    break
                elif override_choice == 's':
                    # A lógica para selecionar uma fase diferente permanece a mesma
                    while True:
                        try:
                            new_stage_input = input(
                                f"Insira a fase de onde pretende recomeçar (0-{MAX_DIFFICULTY_STAGE}): ")
                            new_stage = int(new_stage_input)
                            if 0 <= new_stage <= MAX_DIFFICULTY_STAGE:
                                start_stage = new_stage
                                history = [entry for entry in history if entry['stage'] < start_stage]
                                # Se a fila de retreino existir, remove dela as fases que vêm depois do novo início
                                retraining_queue = [stage for stage in retraining_queue if stage < start_stage]
                                print(f"Fila de retreino atualizada: {retraining_queue}")
                                break
                            else:
                                print(f"Fase inválida. Por favor, insira um número entre 0 e {MAX_DIFFICULTY_STAGE}.")
                        except ValueError:
                            print("Input inválido. Por favor, insira um número.")
                    break
                else:
                    print("Opção inválida. Por favor, insira 'c' ou 's'.")
        else:
            config['resume_training'] = False

    if population is None:
        print("A iniciar uma nova população...")
        # Lógica de criação de população permanece a mesma
        if mode == 'NE':
            input_size = len(np.nan_to_num(sim_mgr.lidar.getRangeImage(), nan=0.0))
            population = NeuralPopulation(pop_size=config['pop_size'], input_size=input_size,
                                          hidden_size=config['hidden_size'], output_size=2,
                                          mutation_rate=config['mutation_rate'], elitism=config['elitism'])
        else:
            population = Population(pop_size=config['pop_size'], mutation_rate=config['mutation_rate'],
                                    elitism=config['elitism'])
        start_stage = 0
        history = []

    current_stage = start_stage
    threshold_prev = config.get('threshold_prev', 0.7)
    threshold_curr = config.get('threshold_curr', 0.7)

    try:
        while True:
            # --- LÓGICA DE TREINO PRINCIPAL (MODIFICADA) ---
            is_retraining = False
            if retraining_queue:
                # Se há fases na fila, treina a primeira
                stage_to_train = retraining_queue[0]
                is_retraining = True
                print(
                    f"\n\n{'=' * 20} A INICIAR RETREINO DA FASE {stage_to_train} ({len(retraining_queue)} na fila) {'=' * 20}")
            elif current_stage <= MAX_DIFFICULTY_STAGE:
                # Se não, continua o treino normal
                stage_to_train = current_stage
                print(f"\n\n{'=' * 20} A INICIAR FASE DE DIFICULDADE {stage_to_train} {'=' * 20}")
            else:
                # Se o treino normal e o retreino acabaram, termina
                print("\nTreino e retreino concluídos com sucesso!")
                break

            sub_index = 0
            attempts_without_progress = 0

            # Loop de treino para a fase selecionada (normal ou de retreino)
            while True:
                generation_id = len(history) + 1
                print(f"\n--- Geração {generation_id} (A treinar na Fase {stage_to_train}) ---")

                # Seleção de mapas
                runs_prev = 5
                runs_curr = 5
                available_prev_stages = [s for s in map_pool if s < stage_to_train]
                maps_prev = [random.choice(map_pool[stage]) for stage in random.sample(available_prev_stages,
                                                                                       min(runs_prev,
                                                                                           len(available_prev_stages)))] if available_prev_stages else []
                maps_curr = random.sample(map_pool.get(stage_to_train, []),
                                          k=min(runs_curr, len(map_pool.get(stage_to_train, []))))

                if not maps_curr:
                    print(f"[AVISO] Não foram encontrados mapas para a Fase {stage_to_train}. A saltar esta fase.")
                    if is_retraining:
                        retraining_queue.pop(0)
                    else:
                        current_stage += 1
                    break

                # Avaliação
                avg_succ_prev = population.evaluate(sim_mgr, maps_prev) if maps_prev else 0
                avg_succ_curr = population.evaluate(sim_mgr, maps_curr)
                rate_prev = avg_succ_prev
                rate_curr = avg_succ_curr

                # Guardar histórico e stats
                fitness_values = [ind.fitness for ind in population.individuals if ind.fitness is not None]
                generation_stats = {
                    'stage': stage_to_train, 'generation': generation_id,
                    'fitness_min': min(fitness_values) if fitness_values else 0,
                    'fitness_avg': np.mean(fitness_values) if fitness_values else 0,
                    'fitness_max': max(fitness_values) if fitness_values else 0,
                    'success_rate_prev': rate_prev, 'success_rate_curr': rate_curr
                }
                history.append(generation_stats)

                # ... (impressão de stats e gravação de modelos) ...
                print("-" * 50)
                print(
                    f"  FITNESS -> Min: {generation_stats['fitness_min']:.2f} | Avg: {generation_stats['fitness_avg']:.2f} | Max: {generation_stats['fitness_max']:.2f}")
                prev_rate_str = f"{rate_prev:.2%}" if rate_prev is not None else "N/A"
                print(f"  SUCCESS -> Prev: {prev_rate_str} | Curr: {rate_curr:.2%}")
                print("-" * 50)

                gen_best = population.get_best_individual()
                if gen_best and (best_overall_individual is None or gen_best.fitness > best_overall_individual.fitness):
                    best_overall_individual = gen_best
                _save_checkpoint(
                    {'population': population, 'best_individual': best_overall_individual, 'stage': current_stage,
                     'history': history})

                # Verifica se a fase (normal ou de retreino) foi concluída
                if (rate_prev is None or rate_prev >= threshold_prev) and rate_curr >= threshold_curr:
                    if is_retraining:
                        print(f"[PROGRESSO] Retreino da Fase {stage_to_train} concluído com sucesso.")
                        retraining_queue.pop(0)
                    else:
                        print(f"[PROGRESSO] Limiares alcançados. A avançar para a próxima fase.")
                        current_stage += 1
                    break
                else:
                    # Lógica para ficar preso numa fase (criar sub-fases)
                    attempts_without_progress += 1
                    if attempts_without_progress >= 50:
                        sub_index += 1
                        base_params = STAGE_DEFINITIONS[stage_to_train]
                        custom_stage_params = generate_intermediate_stage(base_params, sub_index=sub_index)
                        print(f"\nApós 50 tentativas, a criar sub-fase intermédia...")
                        # A avaliação agora acontece com os parâmetros da sub-fase
                    population.create_next_generation()

    except KeyboardInterrupt:
        print("\n[INTERROMPIDO] A guardar o último estado...")
    finally:
        _save_checkpoint({'population': population, 'best_individual': best_overall_individual, 'stage': current_stage,
                          'history': history})
        print("Sessão de treino terminada.")

    return best_overall_individual
