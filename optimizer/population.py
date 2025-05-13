import random
# Assuming Individual is in individual.py
from optimizer.individual import Individual

class Population:
    def __init__(self, size, mutation_rate=0.1, elitism=1):
        """
        Manages a population of individuals for a Genetic Algorithm.

        Args:
            size (int): The number of individuals in the population.
            mutation_rate (float): The mutation rate for creating the next generation.
            elitism (int): The number of top individuals to carry over directly to the next generation.
        """
        self.size = size
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        # Initialize population with random individuals
        self.individuals = [Individual() for _ in range(size)]

    # Modified evaluate to accept the stage
    def evaluate(self, simulator, stage):
        """
        Evaluates the fitness of all individuals in the population using the provided simulator
        on the specified difficulty stage.

        Args:
            simulator: An instance of SimulationManager.
            stage (int): The current training stage.
        """
        print(f"Evaluating population on Stage {stage}...")
        for i, individual in enumerate(self.individuals):
            distP, angleP = individual.get_genes()
            # Call the simulator's run_experiment_with_params with the current stage
            individual.fitness = simulator.run_experiment_with_params(distP, angleP, stage)
            print(f"Individual {i+1}/{self.size}: Fitness = {individual.fitness:.2f}")


    def select_parents(self):
        """
        Tournament selection: chooses the 2 individuals with the highest fitness
        from a randomly sampled subset.
        """
        tournament_size = max(2, int(self.size * 0.1)) # Use a small percentage of population size, min 2
        # Ensure tournament size does not exceed population size
        tournament_size = min(tournament_size, self.size)

        # Handle case where population size is less than 2
        if self.size < 2:
             return None, None # Cannot select parents

        # Ensure we don't sample more individuals than available
        if tournament_size > len(self.individuals):
             tournament_size = len(self.individuals)

        parents_pool = random.sample(self.individuals, tournament_size)
        parents_pool.sort(key=lambda ind: ind.fitness, reverse=True)  # Higher fitness is better
        return parents_pool[0], parents_pool[1]

    def create_next_generation(self):
        """
        Creates the next generation using elitism + crossover + mutation.
        The top individuals are preserved, and the rest are generated through genetic operations.
        """
        # Sort the current population by fitness
        self.individuals.sort(key=lambda ind: ind.fitness, reverse=True)

        next_generation = []

        # Elitism: copy the best individuals
        # Ensure elitism count does not exceed population size
        num_elitism = min(self.elitism, self.size)
        for i in range(num_elitism):
            next_generation.append(self.individuals[i])

        # Generate the rest of the new generation through crossover and mutation
        while len(next_generation) < self.size:
            parent1, parent2 = self.select_parents()
            # Handle case where parent selection failed (e.g., population size < 2)
            if parent1 is None or parent2 is None:
                 # If we can't select parents, just add random individuals or break
                 print("Warning: Could not select enough parents for crossover. Adding random individuals.")
                 while len(next_generation) < self.size:
                     next_generation.append(Individual())
                 break # Exit the loop

            child = parent1.crossover(parent2)
            child.mutate(mutation_rate=self.mutation_rate)
            next_generation.append(child)

        self.individuals = next_generation

    def get_best_individual(self):
        """
        Returns the best individual in the population.
        Returns None if the population is empty.
        """
        if not self.individuals:
            return None
        return max(self.individuals, key=lambda ind: ind.fitness if ind.fitness is not None else -float('inf')) # Handle None fitness
