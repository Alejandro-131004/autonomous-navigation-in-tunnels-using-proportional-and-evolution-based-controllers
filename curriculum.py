# curriculum.py

from environment.configuration import MAX_DIFFICULTY_STAGE
from environment.simulation_manager import SimulationManager
from optimizer.neuralpopulation import NeuralPopulation
import numpy as np

def run_curriculum(
    supervisor,
    pop_size: int = 20,
    generations: int = 30,
    success_threshold: float = 0.6,
    max_failed_generations: int = 10,
    hidden_size: int = 16,
    mutation_rate: float = 0.1,
    elitism: int = 1
):
    """
    Drives a Neuroevolutionary curriculum from difficulty=1 up to MAX_DIFFICULTY_STAGE.

    Args:
        supervisor: Webots Supervisor instance.
        pop_size: Number of individuals per generation.
        generations: (Unused here—see max_failed_generations).
        success_threshold: Fraction of successful runs required to unlock the next level.
        max_failed_generations: Max retries at each difficulty before forcing advance.
        hidden_size: Number of hidden neurons in the MLP.
        mutation_rate: GA mutation probability.
        elitism: Number of elites to carry over each generation.

    Returns:
        best_individual: The individual that performed best across the entire curriculum.
    """
    sim_mgr = SimulationManager(supervisor)

    # infer input/output sizes from sim_mgr and your controller
    sample_scan = np.nan_to_num(sim_mgr.lidar.getRangeImage(), nan=0.0)
    input_size = len(sample_scan)
    output_size = 2  # assuming your controller emits (linear_vel, angular_vel)

    difficulty = 1
    levels = [1]
    best_overall = None

    while difficulty <= MAX_DIFFICULTY_STAGE:
        print(f"\n=== CURRICULUM: difficulties = {levels} ===")
        # build a fresh population
        pop = NeuralPopulation(
            pop_size,
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            mutation_rate=mutation_rate,
            elitism=elitism
        )

        failed = 0
        while failed < max_failed_generations:
            gen_idx = failed + 1
            print(f"\n-- Generation {gen_idx} @ levels {levels} --")

            # evaluate returns two arrays: fitnesses and successes per individual × level
            fitness_mat, success_mat = pop.evaluate(
                sim_mgr,
                difficulty_levels=levels,
                total_stages=int(MAX_DIFFICULTY_STAGE)
            )

            # population metrics
            avg_fitness = fitness_mat.mean()
            total_runs = success_mat.size
            total_successes = success_mat.sum()
            success_rate = total_successes / total_runs

            print(f"[RESULT] avg fitness = {avg_fitness:.1f}")
            print(f"[RESULT] success rate = {success_rate*100:.1f}% "
                  f"({total_successes}/{total_runs})")

            # track the best individual so far
            gen_best = pop.get_best_individual()
            if best_overall is None or gen_best.avg_fitness > best_overall.avg_fitness:
                best_overall = gen_best

            # threshold check → advance difficulty
            if success_rate >= success_threshold:
                print(f"[ADVANCE] {success_rate*100:.1f}% ≥ {success_threshold*100:.0f}% → unlocking level {difficulty+1}")
                difficulty += 1
                if difficulty > MAX_DIFFICULTY_STAGE:
                    print("🎉 Curriculum completed!")
                    return best_overall
                levels.append(difficulty)
                break

            # otherwise breed next generation
            failed += 1
            print(f"[RETRY] failed gens: {failed}/{max_failed_generations}")
            pop.create_next_generation()
        else:
            # ran out of retries → force advance
            difficulty += 1
            if difficulty <= MAX_DIFFICULTY_STAGE:
                levels.append(difficulty)
                print(f"[FORCE ADVANCE] now testing levels {levels}")

    return best_overall
