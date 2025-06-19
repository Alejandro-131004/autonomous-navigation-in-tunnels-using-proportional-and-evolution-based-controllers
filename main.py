import os
import sys
#sys.path = [p for p in sys.path if 'controller' not in p]
#sys.path.insert(0, '/Applications/Webots.app/Contents/lib/controller/python') #joao
#sys.path.insert(0, '/Applications/Webots.app/Contents/lib/controller/python') #Mila
from controller import Supervisor

from curriculum import run_unified_curriculum
from environment.configuration import STAGE_DEFINITIONS, get_stage_parameters, generate_intermediate_stage

# Define the paths for the checkpoint files
NE_CHECKPOINT_FILE = "saved_models/ne_checkpoint.pkl"
GA_CHECKPOINT_FILE = "saved_models/ga_checkpoint.pkl"

def main():
    """
    Main function to run the training pipeline.
    """
    # Base configuration and specific settings for each mode
    base_config = {
        'elitism': 2,
        'threshold_prev': 0.7,
        'threshold_curr': 0.7,
    }

    ne_config = {
        'mode': 'NE',
        'pop_size': 20,
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

    # --- MODE SELECTION AND CHECKPOINT LOGIC ---

    # 1. User selects the training mode
    while True:
        mode_choice = input(
            "Select the training mode:\n"
            "  1: Neuroevolution (Neural Network)\n"
            "  2: Genetic Algorithm (Reactive Parameters)\n"
            "Enter your choice (1 or 2): "
        ).strip()
        if mode_choice == '1':
            final_config.update(base_config)
            final_config.update(ne_config)
            break
        elif mode_choice == '2':
            final_config.update(base_config)
            final_config.update(ga_config)
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")

    # 2. User selects the debug mode
    while True:
        debug_choice = input(
            "\nSelect the debug mode:\n"
            "  1: Normal (generation summary)\n"
            "  2: Debug (detailed info and warnings)\n"
            "Enter your choice (1 or 2): "
        ).strip()
        if debug_choice == '1':
            os.environ['ROBOT_DEBUG_MODE'] = '0'
            break
        elif debug_choice == '2':
            os.environ['ROBOT_DEBUG_MODE'] = '1'
            print("\n--- DEBUG MODE ENABLED ---")
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")

    training_mode = final_config['mode']
    checkpoint_file = final_config['checkpoint_file']

    # 3. Checkpoint handling
    resume_training = False
    if os.path.exists(checkpoint_file):
        while True:
            resume_choice = input(
                f"\nA checkpoint for '{training_mode}' was found.\n"
                f"Do you want to resume training (y) or start fresh (n)? [y/n]: "
            ).lower().strip()
            if resume_choice == 'y':
                resume_training = True
                print(f"Resuming previous {training_mode} training...")
                break
            elif resume_choice == 'n':
                try:
                    os.remove(checkpoint_file)
                    print("Checkpoint removed. Starting a new session.")
                except OSError as e:
                    print(f"Error while removing checkpoint: {e}")
                resume_training = False
                break
            else:
                print("Invalid option. Please enter 'y' or 'n'.")
    else:
        print(f"No checkpoint found for '{training_mode}'. Starting a new session.")

    final_config['resume_training'] = resume_training

    # --- END OF SELECTION AND CHECKPOINT LOGIC ---

    # Initialize supervisor and run the unified curriculum
    sup = Supervisor()
    print(f"\n--- Starting {training_mode} Training ---")

    best_model = run_unified_curriculum(
        supervisor=sup,
        config=final_config
    )

    # Final message
    if best_model:
        print(f"\nTraining complete. The best {training_mode} model has been saved.")
    else:
        print("\nTraining complete. No final model was identified.")

if __name__ == "__main__":
    main()
