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
    Gera mapas se não existirem e carrega-os, organizados por dificuldade.
    """
    if not os.path.exists(maps_dir) or not os.listdir(maps_dir):
        print(f"Diretório de mapas '{maps_dir}' não encontrado ou vazio. A gerar novos mapas...")
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

    # --- Gestão de Checkpoints ---
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

    # --- Inicialização ou Carregamento da População ---
    population = None
    best_overall_individual = None
    start_stage = 1

    if config['resume_training']:
        checkpoint_data = _load_checkpoint()
        if checkpoint_data:
            population = checkpoint_data.get('population')
            best_overall_individual = checkpoint_data.get('best_individual')
            start_stage = checkpoint_data.get('stage', 1)

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

    # --- Loop de Treino Principal ---
    try:
        while current_stage <= MAX_DIFFICULTY_STAGE:
            print(f"\n\n{'=' * 20} A INICIAR FASE DE DIFICULDADE {current_stage} {'=' * 20}")

            for gen_in_stage in range(1, config['max_generations'] + 1):
                print(f"\n--- Geração {gen_in_stage}/{config['max_generations']} (Fase {current_stage}) ---")

                # 1. AVALIAÇÃO DA POPULAÇÃO
                avg_successes = population.evaluate(sim_mgr, current_stage, map_pool)

                print(f"[ESTATÍSTICAS DA GERAÇÃO] Média de sucessos da população: {avg_successes:.2f} / 10")

                # --- LÓGICA REORDENADA ---

                # 2. GUARDAR O MELHOR E O CHECKPOINT (da geração atualmente avaliada)
                gen_best = population.get_best_individual()
                if gen_best and (best_overall_individual is None or (gen_best.fitness is not None and (
                        best_overall_individual.fitness is None or gen_best.fitness > best_overall_individual.fitness))):
                    best_overall_individual = gen_best
                    model_name_prefix = "ne_best" if mode == 'NE' else "ga_best"
                    sim_mgr.save_model(best_overall_individual,
                                       filename=f"{model_name_prefix}_stage_{current_stage}_gen_{gen_in_stage}.pkl")

                _save_checkpoint({
                    'population': population,
                    'best_individual': best_overall_individual,
                    'stage': current_stage
                })

                # 3. CRITÉRIO DE AVANÇO
                advancement_threshold = 7.0
                if avg_successes >= advancement_threshold:
                    print(
                        f"[A AVANÇAR] A média de sucessos ({avg_successes:.2f}) atingiu o limiar ({advancement_threshold}). A passar para a próxima fase.")
                    current_stage += 1
                    break  # Sai do ciclo de gerações para iniciar a próxima fase

                # 4. CRIAÇÃO DA PRÓXIMA GERAÇÃO (se não avançou)
                print("O limiar não foi atingido. A criar a próxima geração...")
                population.create_next_generation()

            else:
                if current_stage < MAX_DIFFICULTY_STAGE:
                    print(f"[AVANÇO FORÇADO] Limite de gerações atingido. A passar para a fase {current_stage + 1}.")
                    current_stage += 1
                else:
                    print("[TREINO COMPLETO] Fase final do currículo concluída.")
                    break

    except KeyboardInterrupt:
        print("\nTreino interrompido pelo utilizador. A guardar o checkpoint final...")
    finally:
        if population:
            _save_checkpoint({
                'population': population,
                'best_individual': best_overall_individual,
                'stage': current_stage
            })
        print("Sessão de treino terminada.")

    return best_overall_individual
