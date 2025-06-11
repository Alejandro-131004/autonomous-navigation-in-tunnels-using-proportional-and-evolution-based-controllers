import os
import pickle
import numpy as np
from environment.configuration import MAX_DIFFICULTY_STAGE
from environment.simulation_manager import SimulationManager
from optimizer.neuralpopulation import NeuralPopulation

# --- Checkpoint Configuration ---
CHECKPOINT_FILE = "saved_models/checkpoint.pkl"


def _save_checkpoint(data):
    """Saves the training state into a checkpoint file."""
    try:
        os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
        with open(CHECKPOINT_FILE, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"[ERROR] Could not save checkpoint: {e}")


def _load_checkpoint():
    """Loads the training state from a checkpoint file."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'rb') as f:
                data = pickle.load(f)
            print(f"|--- Checkpoint loaded from {CHECKPOINT_FILE} ---|")
            return data
        except Exception as e:
            print(f"[ERROR] Could not load checkpoint: {e}")
    return None


def _qualify_candidates(candidates, sim_mgr, qualification_stage, total_stages):
    """
    Tests a list of candidate individuals on all previous stages.
    Returns only those individuals who pass all tests.
    """
    if qualification_stage <= 1:
        return candidates  # No stages prior to stage 1 to qualify in

    qualified_individuals = []
    previous_stages = list(range(1, qualification_stage))
    print(f"--- Qualifying {len(candidates)} top candidates on previous levels: {previous_stages} ---")

    for ind in candidates:
        # Assume an individual is qualified until proven otherwise
        is_qualified = True
        for stage_to_test in previous_stages:
            # The second element returned by run_experiment_with_network is 'succeeded'
            _, succeeded = sim_mgr.run_experiment_with_network(ind, stage_to_test, total_stages)
            if not succeeded:
                is_qualified = False
                break  # No need to test further stages for this individual

        if is_qualified:
            qualified_individuals.append(ind)

    print(f"--- Qualification completed: {len(qualified_individuals)}/{len(candidates)} candidates qualified. ---")
    return qualified_individuals


def run_curriculum(
        supervisor,
        resume_training: bool = False,
        pop_size: int = 30,
        success_threshold: float = 0.5,
        max_generations: int = 1000,
        hidden_size: int = 16,
        mutation_rate: float = 0.15,
        elitism: int = 2,
        top_n_to_qualify: int = 10
):
    """
    Runs training with a hybrid approach: fast evaluation at the current stage
    and rigorous qualification only for the best candidates.
    """
    sim_mgr = SimulationManager(supervisor)
    input_size = len(np.nan_to_num(sim_mgr.lidar.getRangeImage(), nan=0.0))
    output_size = 2

    # --- Initialization or Loading ---
    population, best_overall_individual, start_stage = None, None, 1
    checkpoint_data = _load_checkpoint()
    if resume_training and checkpoint_data:
        population = checkpoint_data.get('population')
        best_overall_individual = checkpoint_data.get('best_individual')
        start_stage = checkpoint_data.get('stage', 1)

    if population is None:
        population = NeuralPopulation(pop_size, input_size, hidden_size, output_size, mutation_rate, elitism)

    # --- Main Training Loop ---
    try:
        current_stage = start_stage
        while current_stage <= MAX_DIFFICULTY_STAGE:
            print(f"\n\n{'=' * 20} STARTING DIFFICULTY STAGE {current_stage} {'=' * 20}")

            for gen_in_stage in range(1, max_generations + 1):
                print(f"\n--- Generation {gen_in_stage}/{max_generations} (Stage {current_stage}) ---")

                # 1. FAST EVALUATION: Evaluate entire population ONLY at current stage
                print(f"Evaluating population at difficulty level: {current_stage}")
                # The 'evaluate' method updates 'ind.successes' for each individual
                population.evaluate(sim_mgr, [current_stage], MAX_DIFFICULTY_STAGE)

                # 2. ELITE SELECTION AND QUALIFICATION
                # Sort population based on fitness at current stage
                population.individuals.sort(key=lambda ind: ind.fitness if ind.fitness is not None else -np.inf,
                                            reverse=True)
                top_candidates = population.individuals[:top_n_to_qualify]

                # **NEW LOGIC**: Filter elite to only those who succeeded at CURRENT stage
                successful_top_candidates = [ind for ind in top_candidates if ind.successes > 0]
                print(
                    f"--- {len(successful_top_candidates)}/{len(top_candidates)} of top candidates succeeded at current stage. ---")

                # Qualify only top candidates who SUCCEEDED at current stage
                qualified_pool = _qualify_candidates(successful_top_candidates, sim_mgr, current_stage,
                                                     MAX_DIFFICULTY_STAGE)

                # 3. DEFINE PARENT POOL
                parent_pool = qualified_pool
                if not parent_pool:
                    # Fallback: if no one fully qualifies, use best (by fitness) from current stage.
                    print(
                        "[WARNING] No fully qualified candidate (success at current + previous stages). Using best from current stage as parents.")
                    parent_pool = top_candidates

                # Next generation is created from defined parent pool
                population.create_next_generation(parent_pool=parent_pool)

                # 4. PROGRESS CHECK
                # Advancement rate is based on how many of the top_n fully qualified
                qualification_rate = len(qualified_pool) / len(top_candidates) if top_candidates else 0
                print(f"[STATS] Elite Full Qualification Rate: {qualification_rate:.2%}")

                # Save best individual and checkpoint
                gen_best = population.get_best_individual()
                if best_overall_individual is None or (
                        gen_best.fitness is not None and best_overall_individual.fitness is not None and gen_best.fitness > best_overall_individual.fitness):
                    best_overall_individual = gen_best
                    sim_mgr.save_model(best_overall_individual,
                                       filename=f"best_model_stage_{current_stage}_gen_{gen_in_stage}.pkl")

                _save_checkpoint({
                    'population': population,
                    'best_individual': best_overall_individual,
                    'stage': current_stage
                })

                # Advancement condition based on elite full qualification rate
                if qualification_rate >= success_threshold and gen_in_stage > 5:
                    print(
                        f"[ADVANCING] Qualification rate ({qualification_rate:.2%}) reached threshold ({success_threshold:.2%}). Moving forward.")
                    current_stage += 1
                    break

            # If generation loop ends without reaching threshold
            else:
                if current_stage < MAX_DIFFICULTY_STAGE:
                    print(
                        f"[FORCED ADVANCE] Generation limit reached. Moving to stage {current_stage + 1}.")
                    current_stage += 1
                else:
                    print("[TRAINING COMPLETE] Curriculum finished.")
                    break

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
    finally:
        if population and best_overall_individual:
            _save_checkpoint({
                'population': population,
                'best_individual': best_overall_individual,
                'stage': current_stage
            })
        print("Training completed or interrupted.")

    return best_overall_individual
