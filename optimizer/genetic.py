import random

class GeneticOptimizer:
    def __init__(self, simulation_manager, population_size=20, generations=10, mutation_rate=0.1):
        """
        Genetic Algorithm for optimizing robot navigation parameters.
        """
        self.simulation_manager = simulation_manager
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    def _create_individual(self):
        """
        Creates a random individual [distP, angleP].
        """
        distP = random.uniform(5.0, 20.0)  # Range for proportional gain on distance
        angleP = random.uniform(2.0, 10.0)  # Range for proportional gain on angle
        return [distP, angleP]

    def _mutate(self, individual):
        """
        Mutates an individual by adding small random noise.
        """
        if random.random() < self.mutation_rate:
            individual[0] += random.uniform(-2.0, 2.0)  # Mutate distP
        if random.random() < self.mutation_rate:
            individual[1] += random.uniform(-1.0, 1.0)  # Mutate angleP
        return individual

    def _crossover(self, parent1, parent2):
        """
        Creates two children from two parents using one-point crossover.
        """
        if random.random() < 0.5:
            child1 = [parent1[0], parent2[1]]
            child2 = [parent2[0], parent1[1]]
        else:
            child1 = [parent2[0], parent1[1]]
            child2 = [parent1[0], parent2[1]]
        return child1, child2

    def optimize(self):
        """
        Runs the Genetic Algorithm to optimize parameters.
        """
        best_individual = None
        best_fitness = -float("inf")

        for generation in range(self.generations):
            print(f"\n=== Generation {generation + 1} ===")

            population = [self._create_individual() for _ in range(self.population_size)]
            fitness_scores = []

            for idx, individual in enumerate(population):
                try:
                    distP, angleP = individual
                    print(f"\nIndividual {idx + 1} → distP = {distP:.2f}, angleP = {angleP:.2f}")
                    average_fitness = self.simulation_manager.run_experiment_with_params(distP, angleP)
                    fitness_scores.append((average_fitness, individual))

                    if average_fitness > best_fitness:
                        best_fitness = average_fitness
                        best_individual = individual
                except Exception as e:
                    print(f"[ERROR] Failed to evaluate individual {idx + 1}: {e}")

            print(f"[INFO] Generation {generation + 1} completed.")

        print(
            f"\nBest individual: distP = {best_individual[0]:.3f}, angleP = {best_individual[1]:.3f} with fitness = {best_fitness:.2f}")
        return best_individual