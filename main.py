
from environment.simulation_manager import SimulationManager
# Assuming GeneticOptimizer is in optimizer/genetic.py
from optimizer.genetic import GeneticOptimizer
# Assuming Population and Individual are in their respective files and imported by GeneticOptimizer/Population
from optimizer.population import Population
from optimizer.individual import Individual
import sys
sys.path = [p for p in sys.path if 'controller' not in p]
#sys.path.insert(0, '/Applications/Webots.app/Contents/lib/controller/python') #joao
sys.path.insert(0, '/Applications/Webots.app/Contents/lib/controller/python') #Mila
from controller import Supervisor

print("🧠 Inicializando o controlador principal...")

if __name__ == "__main__":
    print("🧠 Inicializando Supervisor e SimulationManager...")
    supervisor = Supervisor()
    sim_manager = SimulationManager(supervisor)


    # The GeneticOptimizer needs to manage the training stages.
    # It should have a stage attribute and update it based on performance.
    optimizer = GeneticOptimizer(
        simulation_manager=sim_manager,
        population_size=3, # Example size
        generations_per_stage=1, # Number of generations to run per stage
        max_stage=10, # Maximum difficulty stage
        mutation_rate=0.2,  # Example mutation rate
        performance_threshold=800 # Example average fitness threshold to advance stage
    )

    # The optimize method in GeneticOptimizer should now handle stage progression
    best_params = optimizer.optimize()
    print(f"🚀 Otimização completa. Melhores parâmetros encontrados: distP = {best_params[0]:.3f}, angleP = {best_params[1]:.3f}")

