import random

class Individual:
    def __init__(self, distP=None, angleP=None):
        # Genes (if not provided, initialize randomly)
        self.distP = distP if distP is not None else random.uniform(5.0, 15.0)
        self.angleP = angleP if angleP is not None else random.uniform(3.0, 10.0)
        self.fitness = None  # Fitness will be evaluated later

    def mutate(self, mutation_rate=0.1):
        """
        Small mutation in the genes (±10%).
        """
        if random.random() < mutation_rate:
            self.distP *= random.uniform(0.9, 1.1)
        if random.random() < mutation_rate:
            self.angleP *= random.uniform(0.9, 1.1)

    def crossover(self, other):
        """
        Simple crossover: averages the genes of both parents.
        """
        child_distP = (self.distP + other.distP) / 2.0
        child_angleP = (self.angleP + other.angleP) / 2.0
        return Individual(distP=child_distP, angleP=child_angleP)

    def get_genes(self):
        """
        Returns the genes (used to pass to the robot during simulation).
        """
        return self.distP, self.angleP

    def __repr__(self):
        """
        Returns a human-readable string of the individual for logging or debugging purposes.
        """
        return f"Individual(distP={self.distP:.2f}, angleP={self.angleP:.2f}, fitness={self.fitness})"