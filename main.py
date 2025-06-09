import os
import sys
import numpy as np

# Clean up sys.path from wrong controller entries
# sys.path = [p for p in sys.path if 'controller' not in p]
# sys.path.insert(0, '/Applications/Webots.app/Contents/lib/controller/python')  # Adjust path if needed

from controller import Supervisor
from curriculum import run_curriculum  # Import the curriculum function

if __name__ == "__main__":
    sup = Supervisor()
    best_model = run_curriculum(sup,
                                pop_size=20,  # Example parameters, adjust as needed
                                generations_per_difficulty_check=5,
                                success_rate_threshold=0.6,
                                hidden_size=16,
                                mutation_rate=0.1,
                                elitism=1,
                                max_difficulty_level=10)  # Pass max_difficulty_level

    if best_model:
        print(f"\nCurriculum Learning finished. Best overall model found (Fitness: {best_model.fitness:.2f}).")
    else:
        print("\nCurriculum Learning finished, but no best model was found.")

