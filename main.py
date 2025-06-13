import os
import sys
from controller import Supervisor

# Import the single, unified curriculum function
from curriculum import run_unified_curriculum

# Define paths to checkpoint files
NE_CHECKPOINT_FILE = "saved_models/ne_checkpoint.pkl"
GA_CHECKPOINT_FILE = "saved_models/ga_checkpoint.pkl"


def main():
    """
    Main function to run the training pipeline.
    Prompts the user to select a training mode, sets up the configuration,
    and initiates the unified training curriculum.
    """
    # Define base and mode-specific configurations
    base_config = {
        'max_generations': 100,
        'elitism': 2,
    }

    ne_config = {
        'mode': 'NE',
        'pop_size': 30,  # Increased from 5 for better exploration
        'success_threshold': 0.5,
        'hidden_size': 16,
        'mutation_rate': 0.15,
        'top_n': 10,
        'checkpoint_file': NE_CHECKPOINT_FILE,
    }

    ga_config = {
        'mode': 'GA',
        'pop_size': 30,
        'success_threshold': 0.6,
        'mutation_rate': 0.15,
        'top_n': 10,
        'checkpoint_file': GA_CHECKPOINT_FILE,
    }

    # 1. User selects the training mode
    final_config = {}
    while True:
        mode_choice = input(
            "Select training mode:\n"
            "  1: Neuroevolution (Neural Network Controller)\n"
            "  2: Genetic Algorithm (Reactive Controller Parameters)\n"
            "Enter choice (1 or 2): "
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

    training_mode = final_config['mode']
    checkpoint_file = final_config['checkpoint_file']

    # 2. Handle resuming from checkpoint
    resume_training = False
    if os.path.exists(checkpoint_file):
        while True:
            resume_choice = input(
                f"\nA saved '{training_mode}' checkpoint found.\n"
                f"Resume (y) or start over (n)? [y/n]: "
            ).lower().strip()
            if resume_choice == 'y':
                resume_training = True
                break
            elif resume_choice == 'n':
                try:
                    os.remove(checkpoint_file)
                    print("Checkpoint removed. Starting new session.")
                except OSError as e:
                    print(f"Error removing checkpoint: {e}")
                resume_training = False
                break
            else:
                print("Invalid option.")
    else:
        print(f"No '{training_mode}' checkpoint found. Starting new session.")

    final_config['resume_training'] = resume_training

    # 3. Initialize supervisor and run the unified curriculum
    sup = Supervisor()
    print(f"\n--- Starting {training_mode} Training ---")

    best_model = run_unified_curriculum(
        supervisor=sup,
        config=final_config
    )

    # 4. Final message
    if best_model:
        print(f"\nTraining completed. Best {training_mode} model saved.")
    else:
        print("\nTraining completed. No final best model identified.")


if __name__ == "__main__":
    main()
