import random

class Individual:
    def __init__(self, distP=None, angleP=None, id=None):
        self.id = id
        # sensible limits for classic controller parameters
        self.distP  = distP  if distP  is not None else random.uniform(0.0, 20.0)
        self.angleP = angleP if angleP is not None else random.uniform(0.0, 10.0)
        self.fitness = None
        self.successes = 0

    def get_genes(self):
        """Returns the genes/parameters of the individual."""
        return self.distP, self.angleP

    def crossover(self, other, id=None):
        """Performs blend crossover between self and another individual."""
        alpha = random.random()
        child_distP  = alpha * self.distP  + (1 - alpha) * other.distP
        child_angleP = alpha * self.angleP + (1 - alpha) * other.angleP
        return Individual(child_distP, child_angleP, id=id)

    def mutate(self, mutation_rate=0.1, mutation_strength=0.5):
        """Mutates the individual's genes with given mutation rate and strength."""
        if random.random() < mutation_rate:
            self.distP  += random.gauss(0, mutation_strength)
        if random.random() < mutation_rate:
            self.angleP += random.gauss(0, mutation_strength)
        # Optionally keep parameters non-negative
        self.distP  = max(0.0, self.distP)
        self.angleP = max(0.0, self.angleP)
