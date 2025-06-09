import os
import sys
import numpy as np

sys.path = [p for p in sys.path if 'controller' not in p]
sys.path.insert(0, '/Applications/Webots.app/Contents/lib/controller/python')  # Adjust path if needed

from controller import Supervisor
from curriculum import run_curriculum

if __name__ == "__main__":
    sup = Supervisor()
    best_model = run_curriculum(sup,
                                pop_size=20,
                                generations=5,
                                success_threshold=0.6,
                                max_failed_generations=10)
