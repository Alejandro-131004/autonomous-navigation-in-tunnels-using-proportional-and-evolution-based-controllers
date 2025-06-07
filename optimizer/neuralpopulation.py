from optimizer.individualNeural import IndividualNeural
import random
import numpy as np

class NeuralPopulation:
    def __init__(self, pop_size, input_size, hidden_size, output_size, mutation_rate=0.1, elitism=1):
        """
        Manages a population of neural individuals for neuroevolution.
        """
        self.pop_size = pop_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.mutation_rate = mutation_rate
        self.elitism = elitism

        self.individuals = [
            IndividualNeural(input_size, hidden_size, output_size, id=i)
            for i in range(pop_size)
        ]

    '''def evaluate(self, simulator):
        """
        Evaluates the fitness of each individual using the provided simulator.
        """
        for individual in self.individuals:
            individual.fitness = simulator.evaluate(individual)'''

    def evaluate(self, simulator, base_stage=0, num_stages=5):
        """
            Evaluates the fitness of each individual in the population by running them through multiple simulation stages.

            For each individual:
            - Runs the simulator on 'num_stages' consecutive maps, starting at 'base_stage'.
            - Collects the fitness returned from each stage, summing them up.
            - If all stages run successfully, the final fitness is the average over those stages.
            - If any stage fails (e.g., due to simulation error), that stage is skipped.
            - If all stages fail, assigns a large negative fitness (-1e6) to penalize the individual.

            Args:
                simulator: The simulation manager, which must implement run_experiment_with_network(individual, stage).
                base_stage (int): The index of the first stage to evaluate (e.g., 0).
                num_stages (int): The number of consecutive stages to evaluate per individual (e.g., 5).
            """
        print(
            f"[DEBUG] Evaluating {len(self.individuals)} individuals from stage {base_stage} to {base_stage + num_stages - 1}")
        for individual in self.individuals:
            total_fitness = 0.0
            valid_evaluations = 0
            print(f"\n[EVAL] → Individual {getattr(individual, 'id', '?')} starting multi-stage evaluation")
            for stage_offset in range(num_stages):
                stage = base_stage + stage_offset
                try:
                    print(f"[EVAL] ...Stage {stage}")
                    fitness = simulator.run_experiment_with_network(individual, stage)
                    print(f"[EVAL] Stage {stage} fitness: {fitness:.2f}")
                    total_fitness += fitness
                    valid_evaluations += 1
                except Exception as e:
                    print(f"[ERROR] Stage {stage} failed for individual {individual.id}: {e}")

            if valid_evaluations > 0:
                individual.fitness = total_fitness / valid_evaluations
            else:
                individual.fitness = -1e6  # Heavy penalty if all stages fail
            print(f"[RESULT] Final fitness for Individual {getattr(individual, 'id', '?')}: {individual.fitness:.2f}")

    def select_parents(self, tournament_size=3):
        """
        Selects two parents using tournament selection.
        """
        competitors = random.sample(self.individuals, tournament_size)
        competitors.sort(key=lambda ind: ind.fitness, reverse=True)
        return competitors[0], competitors[1]

    def create_next_generation(self):
        """
        Generates the next generation via elitism, crossover and mutation.
        """
        self.individuals.sort(key=lambda ind: ind.fitness, reverse=True)
        next_generation = self.individuals[:self.elitism]  # Elitism

        while len(next_generation) < self.pop_size:
            parent1, parent2 = self.select_parents()
            child = parent1.crossover(parent2)
            child.mutate(mutation_rate=self.mutation_rate)
            next_generation.append(child)

        self.individuals = next_generation

    def get_best_individual(self):
        """
        Returns the best individual with a valid fitness.
        """
        valid_individuals = [ind for ind in self.individuals if ind.fitness is not None]
        if not valid_individuals:
            raise ValueError("[FATAL] Nenhum indivíduo com fitness válido.")
        return max(valid_individuals, key=lambda ind: ind.fitness)