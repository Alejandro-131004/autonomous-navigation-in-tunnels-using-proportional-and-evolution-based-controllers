# run_evaluation_pipeline.py
"""
Script principal para executar um pipeline de avaliação comparativa.
Avalia e compara três tipos de controladores:
1. Reativo Clássico (hard-coded)
2. Parâmetros otimizados por AG Clássico
3. Modelos de Redes Neuronais (Neuroevolução)
"""
import os
import sys

# --- Correção de Caminho para Imports ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# -----------------------------------------

from controller import Supervisor
from evaluation.model_evaluator import evaluate_controllers
from evaluation.map_generator import generate_maps
from controllers.reactive_controller import reactive_controller_logic

if __name__ == "__main__":
    supervisor = Supervisor()

    # --- Definições do Pipeline ---
    MAPS_OUTPUT_DIR = "evaluation/maps"
    RESULTS_OUTPUT_DIR = "evaluation/results"
    MODELS_NE_DIR = "saved_models"       # Pasta para modelos de Neuroevolução
    MODELS_GA_DIR = "saved_ga_params"  # Pasta para parâmetros do AG Clássico
    NUM_MAPS_PER_DIFFICULTY = 15
    TOTAL_DIFFICULTY_STAGES = 5

    # --- Etapa 1: Gerar os mapas de teste ---
    map_files = generate_maps(
        maps_output_dir=MAPS_OUTPUT_DIR,
        num_maps_per_difficulty=NUM_MAPS_PER_DIFFICULTY,
        total_difficulty_stages=TOTAL_DIFFICULTY_STAGES
    )

    if not map_files:
        print("[ERRO] Nenhum mapa foi gerado. A avaliação não pode continuar.")
        sys.exit(1)

    # --- Etapa 2: Preparar a lista de controladores para avaliação ---
    controllers_to_evaluate = []

    # 1. Adicionar o controlador reativo clássico
    controllers_to_evaluate.append({
        "name": "Reativo Clássico",
        "type": "function",
        "callable": reactive_controller_logic
    })

    # 2. Adicionar os resultados do AG Clássico
    if os.path.isdir(MODELS_GA_DIR):
        for filename in sorted(os.listdir(MODELS_GA_DIR)):
            if filename.endswith(".pkl"):
                controllers_to_evaluate.append({
                    "name": f"AG Clássico - {os.path.splitext(filename)[0]}",
                    "type": "ga_params",
                    "path": os.path.join(MODELS_GA_DIR, filename)
                })

    # 3. Adicionar os modelos de Neuroevolução
    if os.path.isdir(MODELS_NE_DIR):
        for filename in sorted(os.listdir(MODELS_NE_DIR)):
            if filename.endswith(".pkl"):
                controllers_to_evaluate.append({
                    "name": f"Neuroevolução - {os.path.splitext(filename)[0]}",
                    "type": "neural_network",
                    "path": os.path.join(MODELS_NE_DIR, filename)
                })

    print(f"\n--- {len(controllers_to_evaluate)} controladores serão avaliados ---")
    for controller in controllers_to_evaluate:
        print(f"  - {controller['name']} (Tipo: {controller['type']})")


    # --- Etapa 3: Executar a avaliação comparativa ---
    print("\n--- A iniciar o processo de avaliação comparativa ---")
    evaluate_controllers(
        supervisor=supervisor,
        controllers_to_test=controllers_to_evaluate,
        map_files=map_files,
        results_output_dir=RESULTS_OUTPUT_DIR
    )

    print("\n--- Pipeline de Avaliação Concluído ---")
