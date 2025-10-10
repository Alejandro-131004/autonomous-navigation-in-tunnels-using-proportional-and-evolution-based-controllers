import random
# Assuming Population and Individual are in their respective files
from optimizer.population import Population
from optimizer.individual import Individual # Import Individual as it's used for the best_individual return type


class GeneticOptimizer:
    def __init__(self, simulation_manager, population_size=20, generations_per_stage=10, max_stage=10, mutation_rate=0.1, performance_threshold=800):
        """
        Genetic Algorithm for optimizing robot navigation parameters with stage-based difficulty.

        Args:
            simulation_manager: An instance of SimulationManager.
            population_size (int): Number of individuals in each generation.
            generations_per_stage (int): Number of generations to run before considering advancing stage.
            max_stage (int): The highest difficulty stage to reach.
            mutation_rate (float): Probability of mutation.
            performance_threshold (float): Average fitness required to advance to the next stage.
        """
        self.simulation_manager = simulation_manager
        self.population_size = population_size
        self.generations_per_stage = generations_per_stage
        self.max_stage = max_stage
        self.mutation_rate = mutation_rate
        self.performance_threshold = performance_threshold
        self.current_stage = 0 # Start at the easiest stage

    def _create_individual(self):
        """
        Creates a random individual (Individual object).
        """
        # The Individual class handles its own random initialization
        return Individual()

    def optimize(self):
        """
        Runs the Genetic Algorithm with stage-based difficulty progression.
        """

        best_individual_overall = None
        best_fitness_overall = -float("inf")

        # Outer loop for stages
        while self.current_stage <= self.max_stage:
            print(f"\n--- Starting Stage {self.current_stage} ---")
            print(f"Population size: {self.population_size}")
            print(f"Threshold: {self.performance_threshold:.2f}")

            # Initialize population for the current stage
            # You might want to seed the population with the best individual from the previous stage
            # or re-initialize randomly depending on your strategy.
            # For simplicity here, we re-initialize randomly each stage.
            population = Population(self.population_size, self.mutation_rate)

            best_individual_this_stage = None
            best_fitness_this_stage = -float("inf")

            # Inner loop for generations within a stage
            for generation in range(self.generations_per_stage):
                print(f"Stage {self.current_stage}, Generation {generation + 1}/{self.generations_per_stage}")

                # Evaluate the population on the current stage's difficulty
                # The evaluate method now needs the stage
                population.evaluate(self.simulation_manager, self.current_stage)

                # Get fitness scores and find the best individual in this generation
                fitness_scores = [ind.fitness for ind in population.individuals]
                current_best_individual = population.get_best_individual()

                if current_best_individual.fitness > best_fitness_this_stage:
                    best_fitness_this_stage = current_best_individual.fitness
                    best_individual_this_stage = current_best_individual

                print(f"Stage {self.current_stage}, Gen {generation+1}: Best Fitness = {best_fitness_this_stage:.2f}")

                # Create the next generation (selection, crossover, mutation)
                print(f"Best fitness this generation: {best_fitness_this_stage:.2f}")
                population.create_next_generation()

            # After completing generations for the stage, evaluate performance
            average_fitness_this_stage = sum(fitness_scores) / len(fitness_scores) if fitness_scores else -float('inf')
            print(f"Stage {self.current_stage} completed. Average Fitness = {average_fitness_this_stage:.2f}")

            # Update overall best individual
            if best_individual_this_stage and best_fitness_this_stage > best_fitness_overall:
                 best_fitness_overall = best_fitness_this_stage
                 best_individual_overall = best_individual_this_stage

            # Stage progression logic
            if average_fitness_this_stage >= self.performance_threshold and self.current_stage < self.max_stage:
                 print(f"Performance threshold met ({average_fitness_this_stage:.2f} >= {self.performance_threshold}). Advancing to next stage.")
                 self.current_stage += 1
            elif self.current_stage == self.max_stage:
                 print("Reached maximum stage. Stopping optimization.")
                 break # Stop if max stage is reached
            else:
                 print(f"Performance threshold not met ({average_fitness_this_stage:.2f} < {self.performance_threshold}). Staying at Stage {self.current_stage}.")
                 # Optionally, add logic to repeat the stage, stop if no improvement, etc.

        print(
            f"\nBest individual overall: distP = {best_individual_overall.distP:.3f}, angleP = {best_individual_overall.angleP:.3f} with fitness = {best_fitness_overall:.2f}")
        return best_individual_overall.get_genes() # Return the gene values
