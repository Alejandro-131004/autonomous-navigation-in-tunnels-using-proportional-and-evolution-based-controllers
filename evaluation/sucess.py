import os
import pickle
import random
from controller import Supervisor
from environment.simulation_manager import SimulationManager
from curriculum import _load_and_organize_maps
from environment.tunnel import TunnelBuilder
MAX_DIFFICULTY_STAGE = 13
NUM_MAPS_PER_DIFFICULTY = 10

def evaluate_individual_success_rate(checkpoint_path, individual_index):
    with open(checkpoint_path, 'rb') as f:
        data = pickle.load(f)

    population = data.get('population')
    if not population or individual_index >= len(population.individuals):
        print("Individual not found.")
        return
    individual = population.individuals[individual_index]

    # choose controller interface
    # ---- controller wrapper -------------------------------------------------
    from optimizer.individual import Individual
    from optimizer.individualNeural import IndividualNeural

    if isinstance(individual, IndividualNeural):          # NE
        controller = lambda scan: individual.act(scan)
    else:                                                 # GA
        distP, angleP = individual.get_genes()
        controller = lambda scan: sim_mgr._process_lidar_for_ga(scan, distP, angleP)

    supervisor = Supervisor()
    sim_mgr = SimulationManager(supervisor)
    map_pool = _load_and_organize_maps()

    total_runs = 0
    total_successes = 0
    for stage in range(MAX_DIFFICULTY_STAGE + 1):
        maps = map_pool.get(stage, [])
        if not maps:
            print(f"No maps for stage {stage}.")
            continue
        maps = random.sample(maps, min(NUM_MAPS_PER_DIFFICULTY, len(maps)))

        stage_successes = 0
        for map_params in maps:
            supervisor.simulationReset()
            supervisor.step(sim_mgr.timestep)
            sim_mgr.tunnel_builder = TunnelBuilder(supervisor)

            results = sim_mgr._run_single_episode(controller, stage)
            stage_successes += int(results['success'])

        stage_rate = stage_successes / len(maps) * 100
        print(f"Stage {stage}: {stage_rate:.1f}% ({stage_successes}/{len(maps)})")
        total_runs += len(maps)
        total_successes += stage_successes

    overall = total_successes / total_runs * 100 if total_runs else 0
    print(f"\nOverall success rate (individual #{individual_index}): {overall:.2f}%")

if __name__ == "__main__":
    path = input("Checkpoint path (.pkl): ").strip()
    idx = int(input("Individual index: "))
    evaluate_individual_success_rate(path, idx)