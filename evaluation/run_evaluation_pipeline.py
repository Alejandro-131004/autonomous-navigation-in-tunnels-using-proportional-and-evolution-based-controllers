# run_evaluation_pipeline.py
"""
Script principal para executar um pipeline de avaliação comparativa.
Gera um conjunto de mapas de teste e avalia múltiplos controladores
(tanto reativos como modelos de redes neuronais) nesses mapas,
produzindo um relatório comparativo no final.
"""
import os
import sys
import pickle

# --- Correção de Caminho para Imports ---
# Garante que os módulos de outras pastas (controllers, environment) podem ser importados.
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
    MODELS_DIR = "saved_models" # Pasta onde estão os modelos .pkl
    NUM_MAPS_PER_DIFFICULTY = 15
    TOTAL_DIFFICULTY_STAGES = 5

    # --- Etapa 1: Gerar os mapas de teste ---
    # A chamada à função `generate_maps` agora está correta.
    map_files = generate_maps(
        maps_output_dir=MAPS_OUTPUT_DIR,
        num_maps_per_difficulty=NUM_MAPS_PER_DIFFICULTY,
        total_difficulty_stages=TOTAL_DIFFICULTY_STAGES # CORREÇÃO: O nome do parâmetro foi corrigido aqui.
    )

    if not map_files:
        print("[ERRO] Nenhum mapa foi gerado. A avaliação não pode continuar.")
        sys.exit(1)

    # --- Etapa 2: Preparar a lista de controladores para avaliação ---
    controllers_to_evaluate = []

    # Adicionar o controlador reativo clássico à lista
    controllers_to_evaluate.append({
        "name": "Reativo Clássico",
        "type": "function",
        "callable": reactive_controller_logic
    })

    # Adicionar os modelos de redes neuronais da pasta `saved_models`
    if os.path.isdir(MODELS_DIR):
        for filename in sorted(os.listdir(MODELS_DIR)):
            if filename.endswith(".pkl"):
                model_path = os.path.join(MODELS_DIR, filename)
                model_name = os.path.splitext(filename)[0]
                controllers_to_evaluate.append({
                    "name": f"NN - {model_name}",
                    "type": "file",
                    "path": model_path
                })
    else:
        print(f"[AVISO] A pasta de modelos '{MODELS_DIR}' não foi encontrada. Apenas o controlador reativo será testado.")

    print(f"\n--- {len(controllers_to_evaluate)} controladores serão avaliados ---")
    for controller in controllers_to_evaluate:
        print(f"  - {controller['name']}")


    # --- Etapa 3: Executar a avaliação comparativa ---
    print("\n--- A iniciar o processo de avaliação comparativa ---")
    evaluate_controllers(
        supervisor=supervisor,
        controllers_to_test=controllers_to_evaluate,
        map_files=map_files,
        results_output_dir=RESULTS_OUTPUT_DIR
    )

    print("\n--- Pipeline de Avaliação Concluído ---")
