import sys
import os
import pickle
import random

# 1) Path for Webots Python controller bindings (adjust if necessary)
webots_bindings = "/Applications/Webots.app/Contents/lib/controller/python"
if webots_bindings not in sys.path:
    sys.path.insert(0, webots_bindings)

from controller import Supervisor

from optimizer.population import Population
from environment.simulation_manager import SimulationManager
from environment.configuration import MAX_DIFFICULTY_STAGE

# --- Checkpoint configuration ---
CHECKPOINT_FILE = "saved_models/ga_checkpoint_threshold_0_6.pkl"


def _save_checkpoint(data):
    """Saves training state to a checkpoint file."""
    try:
        os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
        with open(CHECKPOINT_FILE, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"[ERROR] Failed to save checkpoint: {e}")


def _load_checkpoint():
    """Loads training state from checkpoint file."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'rb') as f:
                data = pickle.load(f)
            print(f"|--- GA Checkpoint loaded from {CHECKPOINT_FILE} ---|")
            return data
        except Exception as e:
            print(f"[ERROR] Failed to load checkpoint: {e}")
    return None


def run_ga_curriculum(
    supervisor,
    resume_training: bool = False,
    pop_size: int = 30,
    mutation_rate: float = 0.15,
    elitism: int = 2,
    max_generations: int = 100,
    success_threshold: float = 0.6,
    top_n: int = 10
):
    """
    Genetic Algorithm training with stage-wise difficulty curriculum.
    Uses Population (pure GA) and SimulationManager.run_experiment_with_params.
    """
    sim_mgr = SimulationManager(supervisor)

    checkpoint = _load_checkpoint() if resume_training else None
    if checkpoint:
        pop = checkpoint['population']
        best_overall = checkpoint['best']
        start_stage = checkpoint['stage']
    else:
        pop = Population(pop_size=pop_size, mutation_rate=mutation_rate, elitism=elitism)
        best_overall = None
        start_stage = 1

    current_stage = start_stage
    try:
        while current_stage <= MAX_DIFFICULTY_STAGE:
            print(f"\n===== Stage {current_stage}/{MAX_DIFFICULTY_STAGE} =====")
            for gen in range(1, max_generations + 1):
                print(f"-- Generation {gen}/{max_generations} --")
                # Evaluate population
                pop.evaluate(sim_mgr, current_stage)
                # Sort and select top candidates
                pop.individuals.sort(key=lambda ind: ind.fitness or -float('inf'), reverse=True)
                top_candidates = pop.individuals[:top_n]
                qualified = [ind for ind in top_candidates if ind.successes > 0]

                # Update best individual overall
                gen_best = pop.get_best_individual()
                if not best_overall or (gen_best.fitness and gen_best.fitness > best_overall.fitness):
                    best_overall = gen_best
                    sim_mgr.save_model(best_overall, filename=f"ga_best_stage{current_stage}_gen{gen}.pkl")

                # Advance stage if qualification rate meets threshold
                qualification_rate = len(qualified) / len(top_candidates) if top_candidates else 0
                print(f"Qualified success {len(qualified)}/{len(top_candidates)} → {qualification_rate:.2%}")
                if qualification_rate >= success_threshold and gen > 5:
                    print(f"Advancing to stage {current_stage+1}")
                    current_stage += 1
                    break

                # Create next generation
                pop.create_next_generation()

                # Save checkpoint
                _save_checkpoint({'population': pop, 'best': best_overall, 'stage': current_stage})
            else:
                # If didn't advance, forcibly move to next stage or stop
                if current_stage < MAX_DIFFICULTY_STAGE:
                    current_stage += 1
                else:
                    break
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        _save_checkpoint({'population': pop, 'best': best_overall, 'stage': current_stage})
        print("GA training completed or interrupted.")

    return best_overall


if __name__ == '__main__':
    # Run pure GA curriculum training
    sup = Supervisor()
    best_model = run_ga_curriculum(sup, resume_training=False)
    print(f"\nBest model found: {best_model.get_genes()} with fitness {best_model.fitness:.2f}")
