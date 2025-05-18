
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
import numpy as np



print("🧠 Inicializando o controlador principal...")

'''
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
'''
if __name__ == "__main__":
    # Inicializa supervisor e simulador
    supervisor = Supervisor()
    simulator = SimulationManager(supervisor)

    # Lê tamanho do LIDAR
    lidar_data = np.nan_to_num(simulator.lidar.getRangeImage(), nan=0.0)
    input_size = len(lidar_data)

    # Definições
    generations_per_stage = 5
    max_stage = 10
    performance_threshold = 800
    current_stage = 0

    fitness_history = []
    best_individual = None

    while current_stage <= max_stage:
        print(f"\n================ STAGE {current_stage} ================\n")

        # Redefine função de avaliação para o stage atual
        simulator.evaluate = lambda individual: simulator.run_experiment_with_network(individual, stage=current_stage)

        # Corre evolução para este stage
        best_individual_stage, history_stage = simulator.run_neuroevolution(
            generations=generations_per_stage,
            pop_size=20,
            input_size=input_size,
            hidden_size=16,
            output_size=2,
            mutation_rate=0.1,
            elitism=1
        )

        avg_fitness = np.mean(history_stage)
        print(f"🔎 Stage {current_stage} - Média de fitness: {avg_fitness:.2f}")

        fitness_history.extend(history_stage)

        # Atualiza melhor indivíduo global
        if best_individual is None or best_individual_stage.fitness > best_individual.fitness:
            best_individual = best_individual_stage

        if avg_fitness >= performance_threshold:
            current_stage += 1
        else:
            print("⏸ A performance não foi suficiente para avançar de estágio.")
        

    # Guardar logs
    np.savetxt("fitness_log.txt", fitness_history)
    np.save("best_genome.npy", best_individual.get_genome())

    # Ver simulação final
    print("\n🏁 Executando melhor indivíduo na última configuração...")
    simulator.run_experiment_with_network(best_individual, stage=current_stage - 1)
