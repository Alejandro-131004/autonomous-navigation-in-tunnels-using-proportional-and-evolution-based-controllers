from controller import Supervisor
from curriculum import run_curriculum

if __name__ == "__main__":
    sup = Supervisor()
    best_model = run_curriculum(sup,
                                pop_size=20,
                                generations=30,
                                success_threshold=0.6,
                                max_failed_generations=10)
