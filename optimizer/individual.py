import random
import numpy as np


class Individual:
    def __init__(self, distP=None, angleP=None, id=None):
        self.id = id

        # --- IMPORTANT FIX ---
        # The initial value range (0–20) was too wide, which made learning very slow.
        # The range was reduced to more focused values, which will drastically improve initial performance.
        self.distP = distP if distP is not None else random.uniform(1, 10.0)
        self.angleP = angleP if angleP is not None else random.uniform(-0.1, 0.1)

        self.fitness = 0.0
        # Renamed to 'total_successes' for consistency across the codebase
        self.total_successes = 0

    def get_genes(self):
        """Returns the individual's genes/parameters."""
        return self.distP, self.angleP

    def crossover(self, other, id=None):
        """Performs a blended crossover between 'self' and another individual."""
        alpha = random.random()
        child_distP = alpha * self.distP + (1 - alpha) * other.distP
        child_angleP = alpha * self.angleP + (1 - alpha) * other.angleP
        return Individual(child_distP, child_angleP, id=id)

    def mutate(self, mutation_rate=0.1, mutation_strength=0.5):
        """Mutates the individual's genes with a given rate and mutation strength."""
        if random.random() < mutation_rate:
            self.distP += random.gauss(0, mutation_strength)
        if random.random() < mutation_rate:
            self.angleP += random.gauss(0, mutation_strength)

        # Ensures parameters remain within valid bounds
        self.distP = np.clip(self.distP, 0.1, 15.0)
        self.angleP = np.clip(self.angleP, 0.0, 5.0)
