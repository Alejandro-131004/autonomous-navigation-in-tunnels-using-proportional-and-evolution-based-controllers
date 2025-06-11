# run_evaluation_pipeline.py

import os
import sys

# --- INÍCIO DA CORREÇÃO DE CAMINHO ---
# Adiciona o diretório raiz do projeto ao sys.path para garantir que os imports funcionem.
# O script assume que a estrutura é:
# /Robotics
#   /controllers
#   /environment
#   /evaluation  <-- este script está aqui
#   /...
# Obtém o caminho para a pasta 'evaluation' (onde este script está)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Sobe um nível para chegar à raiz do projeto ('Robotics')
project_root = os.path.dirname(current_dir)
# Adiciona a raiz do projeto ao início do sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- FIM DA CORREÇÃO DE CAMINHO ---

from controller import Supervisor
from map_generator import generate_maps
from model_evaluator import evaluate_models
from environment.configuration import MAX_DIFFICULTY_STAGE

if __name__ == "__main__":
    print("--- Pipeline de Avaliação de Modelos Iniciada ---")

    # --- 1. Configurações da Geração de Mapas ---
    MAPS_OUTPUT_DIR = os.path.join(project_root, "evaluation", "maps")
    NUM_MAPS_PER_DIFFICULTY = 100
    TOTAL_DIFFICULTY_STAGES = int(MAX_DIFFICULTY_STAGE - 1)

    # --- 2. Configurações da Avaliação de Modelos ---
    MODEL_FOLDER = os.path.join(project_root, "saved_models")
    RESULTS_OUTPUT_DIR = os.path.join(project_root, "evaluation", "results")

    if not os.path.isdir(MODEL_FOLDER):
        print(f"[ERRO] A pasta de modelos '{MODEL_FOLDER}' não foi encontrada. Por favor, verifique o caminho e tente novamente.")
        sys.exit(1)

    print("Iniciando Supervisor Webots para a pipeline de avaliação...")
    supervisor = Supervisor()

    # --- Etapa 1: Gerar os mapas de teste ---
    generated_map_files = generate_maps(
        maps_output_dir=MAPS_OUTPUT_DIR,
        num_maps_per_difficulty=NUM_MAPS_PER_DIFFICULTY,
        total_difficulty_stages=TOTAL_DIFFICULTY_STAGES
    )

    if not generated_map_files:
        print("[AVISO] Nenhuns mapas foram gerados. A avaliação não pode prosseguir.")
        sys.exit(0)

    # --- Etapa 2: Avaliar os modelos nos mapas gerados ---
    evaluate_models(
        supervisor=supervisor,
        model_folder=MODEL_FOLDER,
        map_files=generated_map_files,
        results_output_dir=RESULTS_OUTPUT_DIR
    )

    print("\n--- Pipeline de Avaliação de Modelos Concluída ---")