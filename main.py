"""
Main script to run the training pipeline.
"""
import os
import sys
import numpy as np  # Necessário para np.nan_to_num em reactive_controller_logic
import random  # Necessário para random.sample

# --- Path Fix for Imports ---
# sys.path adjustments (já presentes)
sys.path.insert(0, '/Applications/Webots.app/Contents/lib/controller/python')  # Mila
from controller import Supervisor

from curriculum import run_unified_curriculum, _load_and_organize_maps  # Importa _load_and_organize_maps
from environment.configuration import STAGE_DEFINITIONS, get_stage_parameters, generate_intermediate_stage, \
    MAX_DIFFICULTY_STAGE
from environment.simulation_manager import SimulationManager  # Importa SimulationManager
from evaluation.map_generator import generate_maps  # Importa generate_maps
from controllers.reactive_controller import reactive_controller_logic
from controllers.utils import cmd_vel  # Garante que cmd_vel está disponível para o modo reativo

# Define os caminhos para os arquivos de checkpoint
NE_CHECKPOINT_FILE = "saved_models/ne_checkpoint.pkl"
GA_CHECKPOINT_FILE = "saved_models/ga_checkpoint.pkl"


def main():
    """
    Função principal para executar o pipeline de treinamento.
    """
    # Configuração base e configurações específicas para cada modo
    base_config = {
        'elitism': 2,
        'threshold_prev': 0.7,
        'threshold_curr': 0.7,
    }

    ne_config = {
        'mode': 'NE',
        'pop_size': 30,
        'hidden_size': 16,
        'mutation_rate': 0.15,
        'checkpoint_file': NE_CHECKPOINT_FILE,
    }

    ga_config = {
        'mode': 'GA',
        'pop_size': 30,
        'mutation_rate': 0.15,
        'checkpoint_file': GA_CHECKPOINT_FILE,
    }

    final_config = {}

    # --- SELEÇÃO DE MODO E LÓGICA DE CHECKPOINT ---

    # 1. O usuário seleciona o modo de treinamento
    while True:
        mode_choice = input(
            "Selecione o modo de treinamento:\n"
            "  1: Neuroevolução (Rede Neural)\n"
            "  2: Algoritmo Genético (Parâmetros Reativos)\n"
            "  3: Controlador Reativo (Baseado em regras clássicas)\n"
            "Digite sua escolha (1, 2 ou 3): "
        ).strip()
        if mode_choice == '1':
            final_config.update(base_config)
            final_config.update(ne_config)
            selected_mode = 'NE'
            break
        elif mode_choice == '2':
            final_config.update(base_config)
            final_config.update(ga_config)
            selected_mode = 'GA'
            break
        elif mode_choice == '3':
            selected_mode = 'REACTIVE'
            break
        else:
            print("Escolha inválida. Por favor, digite 1, 2 ou 3.")

    # 2. O usuário seleciona o modo de depuração
    while True:
        debug_choice = input(
            "\nSelecione o modo de depuração:\n"
            "  1: Normal (resumo da geração)\n"
            "  2: Depuração (informações detalhadas e avisos)\n"
            "Digite sua escolha (1 ou 2): "
        ).strip()
        if debug_choice == '1':
            os.environ['ROBOT_DEBUG_MODE'] = '0'
            break
        elif debug_choice == '2':
            os.environ['ROBOT_DEBUG_MODE'] = '1'
            print("\n--- MODO DE DEPURAÇÃO ATIVADO ---")
            break
        else:
            print("Escolha inválida. Por favor, digite 1 ou 2.")

    if selected_mode == 'REACTIVE':
        # --- EXECUÇÃO DO CONTROLADOR REATIVO ---
        print("\n--- Iniciando Avaliação do Controlador Reativo ---")

        # Seleciona o modo FOV
        while True:
            fov_choice = input(
                "Selecione o modo FOV para o Controlador Reativo:\n"
                "  1: FOV Completo\n"
                "  2: FOV Esquerdo\n"
                "  3: FOV Direito\n"
                "Digite sua escolha (1, 2 ou 3): "
            ).strip()
            if fov_choice == '1':
                selected_fov = 'full'
                fov_name = 'FOV Completo'
                break
            elif fov_choice == '2':
                selected_fov = 'left'
                fov_name = 'FOV Esquerdo'
                break
            elif fov_choice == '3':
                selected_fov = 'right'
                fov_name = 'FOV Direito'
                break
            else:
                print("Escolha inválida. Por favor, digite 1, 2 ou 3.")

        print(f"Controlador Reativo selecionado com: {fov_name}")

        # Configurações do Pipeline para Avaliação Reativa
        MAPS_OUTPUT_DIR = "evaluation/maps"
        NUM_MAPS_PER_DIFFICULTY = 3  # Definido para 3 mapas por dificuldade, conforme solicitado
        TOTAL_DIFFICULTY_STAGES = 15  # Alterado para 15, conforme solicitado

        # Garante que os mapas existem
        print(
            f"Gerando {NUM_MAPS_PER_DIFFICULTY} mapas para cada uma das {TOTAL_DIFFICULTY_STAGES + 1} fases, se não existirem...")
        generate_maps(
            maps_output_dir=MAPS_OUTPUT_DIR,
            num_maps_per_difficulty=NUM_MAPS_PER_DIFFICULTY,
            total_difficulty_stages=TOTAL_DIFFICULTY_STAGES
        )
        map_pool = _load_and_organize_maps(maps_dir=MAPS_OUTPUT_DIR, num_maps_per_diff=NUM_MAPS_PER_DIFFICULTY)

        sup = Supervisor()
        sim_mgr = SimulationManager(supervisor=sup)

        print(f"\n--- Executando Controlador Reativo ({fov_name}) em {TOTAL_DIFFICULTY_STAGES + 1} fases ---")

        total_runs = 0
        total_successes = 0

        for difficulty_level in range(TOTAL_DIFFICULTY_STAGES + 1):
            if difficulty_level not in map_pool or not map_pool[difficulty_level]:
                print(f"  [INFO] Nenhum mapa encontrado para a Fase de Dificuldade {difficulty_level}. Pulando.")
                continue

            maps_for_stage = random.sample(map_pool[difficulty_level],
                                           min(NUM_MAPS_PER_DIFFICULTY, len(map_pool[difficulty_level])))
            print(f"\n  Avaliando Fase de Dificuldade {difficulty_level} com {len(maps_for_stage)} mapas...")

            stage_successes = 0
            for i, map_params in enumerate(maps_for_stage):
                print(f"    Executando Mapa {i + 1}/{len(maps_for_stage)} (Dificuldade {difficulty_level})...")
                # O controller_callable agora inclui o fov_mode
                results = sim_mgr._run_single_episode(
                    controller_callable=lambda scan: reactive_controller_logic(scan, fov_mode=selected_fov),
                    stage=difficulty_level  # Passa o nível de dificuldade diretamente
                )

                if results['success']:
                    stage_successes += 1
                    total_successes += 1
                    print(
                        f"      Mapa {i + 1} SUCESSO! Fitness: {results['fitness']:.2f}, Tempo: {results['elapsed_time']:.2f}s")
                else:
                    print(
                        f"      Mapa {i + 1} FALHOU. Fitness: {results['fitness']:.2f}, Colisão: {results['collided']}, Tempo Esgotado: {results['timeout']}, Sem Movimento: {results['no_movement_timeout']}")
                total_runs += 1

            print(f"  Resumo da Fase {difficulty_level}: {stage_successes}/{len(maps_for_stage)} mapas bem-sucedidos.")

        print(f"\n--- Avaliação do Controlador Reativo Concluída ---")
        print(f"Total de Mapas Executados: {total_runs}")
        print(f"Total de Sucessos: {total_successes}")
        print(f"Taxa de Sucesso Geral: {total_successes / total_runs:.2%}" if total_runs > 0 else "N/A")
        return  # Sai do main após o modo reativo

    # Só executa estas linhas SE não estiver em modo reativo
    training_mode = final_config['mode']
    checkpoint_file = final_config['checkpoint_file']

    # 3. Tratamento de checkpoint (permanece inalterado)
    resume_training = False
    if os.path.exists(checkpoint_file):
        while True:
            resume_choice = input(
                f"\nUm checkpoint para '{training_mode}' foi encontrado.\n"
                f"Deseja retomar o treinamento (s) ou começar do zero (n)? [s/n]: "
            ).lower().strip()
            if resume_choice == 'y':
                resume_training = True
                print(f"Retomando o treinamento anterior de {training_mode}...")
                break
            elif resume_choice == 'n':
                try:
                    os.remove(checkpoint_file)
                    print("Checkpoint removido. Iniciando uma nova sessão.")
                except OSError as e:
                    print(f"Erro ao remover o checkpoint: {e}")
                resume_training = False
                break
            else:
                print("Opção inválida. Por favor, digite 's' ou 'n'.")
    else:
        print(f"Nenhum checkpoint encontrado para '{training_mode}'. Iniciando uma nova sessão.")

    final_config['resume_training'] = resume_training

    # --- FIM DA LÓGICA DE SELEÇÃO E CHECKPOINT ---
    print(f"Limiares: anterior = {final_config['threshold_prev']}, atual = {final_config['threshold_curr']}")

    # Inicializa o supervisor e executa o currículo unificado
    sup = Supervisor()
    print(f"\n--- Iniciando Treinamento {training_mode} ---")

    best_model = run_unified_curriculum(
        supervisor=sup,
        config=final_config
    )

    # Mensagem final
    if best_model:
        print(f"\nTreinamento concluído. O melhor modelo {training_mode} foi salvo.")
    else:
        print("\nTreinamento concluído. Nenhum modelo final foi identificado.")


if __name__ == "__main__":
    main()
