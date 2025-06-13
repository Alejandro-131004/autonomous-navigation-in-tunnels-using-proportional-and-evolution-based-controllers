import os
import sys
import pickle
from environment.simulation_manager import SimulationManager

# Clean up sys.path from wrong controller entries
sys.path = [p for p in sys.path if 'controller' not in p]
sys.path.insert(0, '/Applications/Webots.app/Contents/lib/controller/python')  # Adjust path if needed
from curriculum import run_curriculum
from controller import Supervisor

# --- Checkpoint Configuration ---
CHECKPOINT_FILE = "saved_models/checkpoint.pkl"

if __name__ == "__main__":
    sup = Supervisor()
    resume_training = False

    # Check if a checkpoint exists and ask the user
    if os.path.exists(CHECKPOINT_FILE):
        while True:
            choice = input(
                f"A saved training checkpoint was found at '{CHECKPOINT_FILE}'.\n"
                f"Do you want to continue training (y) or start over (n)? [y/n]: "
            ).lower().strip()
            if choice == 'y':
                resume_training = True
                print("Resuming previous training...")
                break
            elif choice == 'n':
                try:
                    os.remove(CHECKPOINT_FILE)
                    print("Previous training checkpoint removed. Starting from scratch...")
                except OSError as e:
                    print(f"Error removing checkpoint file: {e}")
                resume_training = False
                break
            else:
                print("Invalid option. Please enter 'y' or 'n'.")
    else:
        print("No saved training checkpoint found. Starting from scratch...")

    # Call the main curriculum function
    best_model = run_curriculum(
        supervisor=sup,
        resume_training=resume_training,
        pop_size=5,
        success_threshold=0.1,
        max_failed_generations=0,
        hidden_size=16,
        mutation_rate=0.15,
        elitism=2
    )

    # The logic for saving the final model is already inside run_curriculum.
    # We only print a final message here.
    if best_model:
        print("\nTraining completed. The best model has been periodically saved and saved at the end.")
    else:
        print("\nTraining completed without a final best model.")
