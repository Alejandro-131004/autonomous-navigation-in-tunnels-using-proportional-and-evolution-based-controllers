"""
Main script to run a comparative evaluation pipeline.
Evaluates and compares three types of controllers:
1. Classic Reactive (hard-coded)
2. Parameters optimized by Classic GA
3. Neural Network Models (Neuroevolution)
"""
import os
import sys

# --- Path Fix for Imports ---
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

    # --- Pipeline Settings ---
    MAPS_OUTPUT_DIR = "evaluation/maps"
    RESULTS_OUTPUT_DIR = "evaluation/results"
    MODELS_NE_DIR = "C:\\Users\\joaop\\OneDrive\\Documentos\\Robotics\\saved_models"       # Folder for Neuroevolution models
    MODELS_GA_DIR = "C:\\Users\\joaop\\OneDrive\\Documentos\\Robotics\\saved_ga_params"    # Folder for Classic GA parameters
    NUM_MAPS_PER_DIFFICULTY = 5
    TOTAL_DIFFICULTY_STAGES = 8

    # --- Step 1: Generate test maps ---
    map_files = generate_maps(
        maps_output_dir=MAPS_OUTPUT_DIR,
        num_maps_per_difficulty=NUM_MAPS_PER_DIFFICULTY,
        total_difficulty_stages=TOTAL_DIFFICULTY_STAGES
    )

    if not map_files:
        print("[ERROR] No maps were generated. Evaluation cannot continue.")
        sys.exit(1)

    # --- Step 2: Prepare list of controllers for evaluation ---
    controllers_to_evaluate = []

    # 1. Add the classic reactive controller
    controllers_to_evaluate.append({
        "name": "Classic Reactive",
        "type": "function",
        "callable": reactive_controller_logic
    })

    # 2. Add Classic GA results
    if os.path.isdir(MODELS_GA_DIR):
        for filename in sorted(os.listdir(MODELS_GA_DIR)):
            if filename.endswith(".pkl"):
                controllers_to_evaluate.append({
                    "name": f"Classic GA - {os.path.splitext(filename)[0]}",
                    "type": "ga_params",
                    "path": os.path.join(MODELS_GA_DIR, filename)
                })

    # 3. Add Neuroevolution models
    if os.path.isdir(MODELS_NE_DIR):
        for filename in sorted(os.listdir(MODELS_NE_DIR)):
            if filename.endswith(".pkl"):
                controllers_to_evaluate.append({
                    "name": f"Neuroevolution - {os.path.splitext(filename)[0]}",
                    "type": "neural_network",
                    "path": os.path.join(MODELS_NE_DIR, filename)
                })

    print(f"\n--- {len(controllers_to_evaluate)} controllers will be evaluated ---")
    for controller in controllers_to_evaluate:
        print(f"  - {controller['name']} (Type: {controller['type']})")


    # --- Step 3: Run the comparative evaluation ---
    print("\n--- Starting comparative evaluation process ---")
    evaluate_controllers(
        supervisor=supervisor,
        controllers_to_test=controllers_to_evaluate,
        map_files=map_files,
        results_output_dir=RESULTS_OUTPUT_DIR
    )

    print("\n--- Evaluation Pipeline Completed ---")
