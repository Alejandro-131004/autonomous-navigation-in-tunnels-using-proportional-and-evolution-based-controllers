import os
import pickle
import numpy as np
import random
from collections import defaultdict

from environment.configuration import MAX_DIFFICULTY_STAGE
from environment.simulation_manager import SimulationManager
from evaluation.map_generator import generate_maps

from optimizer.neuralpopulation import NeuralPopulation
from optimizer.population import Population


def _load_and_organize_maps(maps_dir="evaluation/maps", num_maps_per_diff=50):
    """
    Gera 1000 mapas (50 por cada uma das 20 fases) se não existirem e carrega-os.
    """
    # A geração de mapas agora usa o MAX_DIFFICULTY_STAGE do ficheiro de configuração (20)
    if not os.path.exists(maps_dir) or not os.listdir(maps_dir):
        print(
            f"Diretório de mapas '{maps_dir}' não encontrado ou vazio. A gerar {num_maps_per_diff * MAX_DIFFICULTY_STAGE} novos mapas...")
        generate_maps(maps_output_dir=maps_dir, num_maps_per_difficulty=num_maps_per_diff)

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
    """
    Executa um currículo de treino unificado para NE e GA com base na avaliação média da população.
    """
    mode = config['mode']
    sim_mgr = SimulationManager(supervisor)

    map_pool = _load_and_organize_maps()

    checkpoint_file = config['checkpoint_file']

    def _save_checkpoint(data):
        try:
            os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"[ERRO] Não foi possível guardar o checkpoint em {checkpoint_file}: {e}")

    def _load_checkpoint():
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'rb') as f:
                    data = pickle.load(f)
                print(f"|--- Checkpoint carregado de {checkpoint_file} ---|")
                return data
            except Exception as e:
                print(f"[ERRO] Não foi possível carregar o checkpoint de {checkpoint_file}: {e}")
        return None

    population = None
    best_overall_individual = None
    start_stage = 1
    history = []
    generations_run_in_stage = 0

    if config['resume_training']:
        checkpoint_data = _load_checkpoint()
        if checkpoint_data:
            population = checkpoint_data.get('population')
            best_overall_individual = checkpoint_data.get('best_individual')
            start_stage = checkpoint_data.get('stage', 1)
            history = checkpoint_data.get('history', [])

            if history:
                last_stage_in_history = history[-1]['stage']
                if last_stage_in_history == start_stage:
                    for entry in reversed(history):
                        if entry['stage'] == start_stage:
                            generations_run_in_stage += 1
                        else:
                            break
            print(f"A retomar da fase {start_stage}. {generations_run_in_stage} gerações já concluídas nesta fase.")

    if population is None:
        print("A inicializar nova população...")
        if mode == 'NE':
            input_size = len(np.nan_to_num(sim_mgr.lidar.getRangeImage(), nan=0.0))
            output_size = 2
            population = NeuralPopulation(
                pop_size=config['pop_size'],
                input_size=input_size,
                hidden_size=config['hidden_size'],
                output_size=output_size,
                mutation_rate=config['mutation_rate'],
                elitism=config['elitism']
            )
        elif mode == 'GA':
            population = Population(
                pop_size=config['pop_size'],
                mutation_rate=config['mutation_rate'],
                elitism=config['elitism']
            )

    current_stage = start_stage

    try:
        while current_stage <= MAX_DIFFICULTY_STAGE:
            print(f"\n\n{'=' * 20} A INICIAR FASE DE DIFICULDADE {current_stage} {'=' * 20}")

            start_gen_for_stage = generations_run_in_stage + 1
            for gen_in_stage in range(start_gen_for_stage, config['max_generations'] + 1):
                global_generation_count = len(history) + 1
                print(
                    f"\n--- Geração Global {global_generation_count} (Fase {current_stage}, Tentativa {gen_in_stage}/{config['max_generations']}) ---")

                # --- LÓGICA DE SELEÇÃO DE MAPAS ESTRUTURADA ---
                maps_for_this_generation = []
                runs_per_eval = 10

                # 1. Selecionar mapas de fases anteriores de forma estruturada
                num_previous_maps_desired = runs_per_eval // 2
                available_previous_stages = [key for key in map_pool.keys() if key < current_stage]

                if available_previous_stages:
                    # Escolhe aleatoriamente as FASES de onde tirar os mapas
                    num_stages_to_sample = min(num_previous_maps_desired, len(available_previous_stages))
                    stages_to_sample_from = random.sample(available_previous_stages, num_stages_to_sample)

                    # Tira um mapa de cada fase selecionada
                    for stage_key in stages_to_sample_from:
                        if map_pool[stage_key]:
                            maps_for_this_generation.append(random.choice(map_pool[stage_key]))

                # 2. Preencher o resto com mapas da fase atual
                num_current_maps_needed = runs_per_eval - len(maps_for_this_generation)
                current_stage_maps = map_pool.get(current_stage, [])
                if current_stage_maps:
                    num_to_sample = min(num_current_maps_needed, len(current_stage_maps))
                    maps_for_this_generation.extend(random.sample(current_stage_maps, num_to_sample))

                print(
                    f"Amostra de mapas para esta geração (Fases): {[m['difficulty_level'] for m in maps_for_this_generation]}")
                # --- FIM DA LÓGICA DE SELEÇÃO ---

                avg_successes = population.evaluate(sim_mgr, maps_for_this_generation)

                if population.individuals:
                    sorted_by_fitness = sorted(population.individuals, key=lambda ind: ind.fitness)

                    fitness_min = sorted_by_fitness[0].fitness
                    fitness_max = sorted_by_fitness[-1].fitness
                    fitness_avg = np.mean([ind.fitness for ind in population.individuals])

                    # Garantir que runs_per_eval > 0 para evitar divisão por zero
                    num_runs = len(maps_for_this_generation) if len(maps_for_this_generation) > 0 else 1

                    success_rates = [ind.total_successes / num_runs for ind in sorted_by_fitness]
                    success_rate_min = success_rates[0]
                    success_rate_median = success_rates[len(success_rates) // 2]
                    success_rate_max = success_rates[-1]
                    success_rate_avg_pop = avg_successes / num_runs if num_runs > 0 else 0

                    generation_stats = {
                        'stage': current_stage, 'generation': global_generation_count,
                        'fitness_min': fitness_min, 'fitness_avg': fitness_avg, 'fitness_max': fitness_max,
                        'success_rate_min': success_rate_min, 'success_rate_median': success_rate_median,
                        'success_rate_max': success_rate_max, 'success_rate_avg_pop': success_rate_avg_pop
                    }
                    history.append(generation_stats)

                    print("-" * 50)
                    print("  ESTATÍSTICAS DA GERAÇÃO:")
                    print(
                        f"    Fitness  -> Mín: {fitness_min:8.2f} | Média: {fitness_avg:8.2f} | Máx: {fitness_max:8.2f}")
                    print(
                        f"    Sucesso  -> Pior: {success_rate_min:7.2%} | Mediano: {success_rate_median:7.2%} | Melhor: {success_rate_max:7.2%}")
                    print(f"    MÉDIA DE SUCESSO DA POPULAÇÃO: {success_rate_avg_pop:.2%}")
                    print("-" * 50)

                gen_best = population.get_best_individual()
                if gen_best and (best_overall_individual is None or (gen_best.fitness is not None and (
                        best_overall_individual.fitness is None or gen_best.fitness > best_overall_individual.fitness))):
                    best_overall_individual = gen_best
                    model_name_prefix = "ne_best" if mode == 'NE' else "ga_best"
                    sim_mgr.save_model(best_overall_individual,
                                       filename=f"{model_name_prefix}_stage_{current_stage}_gen_{global_generation_count}.pkl")

                _save_checkpoint({
                    'population': population, 'best_individual': best_overall_individual,
                    'stage': current_stage, 'history': history
                })

                advancement_threshold = 7.0
                if avg_successes >= advancement_threshold:
                    print(
                        f"[A AVANÇAR] A média de sucessos ({avg_successes:.2f}/{runs_per_eval}) atingiu o limiar. A passar para a próxima fase.")
                    current_stage += 1
                    break

                print("O limiar não foi atingido. A criar a próxima geração...")
                population.create_next_generation()

            generations_run_in_stage = 0

            if gen_in_stage == config['max_generations'] and current_stage < MAX_DIFFICULTY_STAGE:
                print(f"[AVANÇO FORÇADO] Limite de gerações atingido. A passar para a fase {current_stage + 1}.")
                current_stage += 1
            elif current_stage >= MAX_DIFFICULTY_STAGE:
                print("[TREINO COMPLETO] Fase final do currículo concluída.")
                break

    except KeyboardInterrupt:
        print("\nTreino interrompido pelo utilizador. A guardar o checkpoint final...")
    finally:
        if population:
            _save_checkpoint({
                'population': population, 'best_individual': best_overall_individual,
                'stage': current_stage, 'history': history
            })
        print("Sessão de treino terminada.")

    return best_overall_individual
