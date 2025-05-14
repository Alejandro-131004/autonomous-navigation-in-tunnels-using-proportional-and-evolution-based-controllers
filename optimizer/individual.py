import random

class Individual:
    def __init__(self, distP=None, angleP=None):
        # Genes (if not provided, initialize randomly)
        # Use a wider range for initial population diversity
        self.distP = distP if distP is not None else 2
        self.angleP = angleP if angleP is not None else 0
        self.fitness = None  # Fitness will be evaluated later

    def mutate(self, mutation_rate=0.1):
        """
        Mutates the genes by adding small random noise.
        Ensure mutation keeps parameters within reasonable bounds if necessary.
        """
        if random.random() < mutation_rate:
            # Add Gaussian noise, adjust scale based on parameter range
            self.distP += random.gauss(0, 2.0)
            # Clamp values to a reasonable range
            self.distP = max(1.0, min(30.0, self.distP)) # Example clamping

        if random.random() < mutation_rate:
            self.angleP += random.gauss(0, 1.0)
            # Clamp values to a reasonable range
            self.angleP = max(1.0, min(15.0, self.angleP)) # Example clamping

    def crossover(self, other):
        """
        Creates a new individual using uniform crossover.
        Each gene is inherited from either parent with 50% probability.
        """
        child_distP = self.distP if random.random() < 0.5 else other.distP
        child_angleP = self.angleP if random.random() < 0.5 else other.angleP
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
        return f"Individual(distP={self.distP:.2f}, angleP={self.angleP:.2f}, fitness={self.fitness:.2f if self.fitness is not None else None})"
