from controller import Supervisor
from environment.simulation_manager import SimulationManager
from optimizer.genetic import GeneticOptimizer
print("🧠 Inicializando GeneticOptimizer...")

if __name__ == "__main__":
    print("🧠 Inicializando GeneticOptimizer...")
    supervisor = Supervisor()
    print("🧠 Inicializando GeneticOptimizer...")
    sim_manager = SimulationManager(supervisor)

    optimizer = GeneticOptimizer(
        simulation_manager=sim_manager,
        population_size=10,
        generations=5,
        mutation_rate=0.2
    )

    best_params = optimizer.optimize()
    print(f"Best parameters found: distP = {best_params[0]:.3f}, angleP = {best_params[1]:.3f}")
