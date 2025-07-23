import random
import numpy as np


class Individual:
    def __init__(self, distP=None, angleP=None, id=None):
        self.id = id

        # --- CORREÇÃO IMPORTANTE ---
        # A gama de valores inicial era demasiado vasta (0-20), o que tornava a aprendizagem muito lenta.
        # Reduzi a gama para valores mais focados, o que irá melhorar drasticamente a performance inicial.
        self.distP = distP if distP is not None else random.uniform(1, 10.0)
        self.angleP = angleP if angleP is not None else random.uniform(-0.1, 0.1)

        self.fitness = 0.0
        # O nome foi alterado para 'total_successes' para ser consistente com o resto do código
        self.total_successes = 0

    def get_genes(self):
        """Retorna os genes/parâmetros do indivíduo."""
        return self.distP, self.angleP

    def crossover(self, other, id=None):
        """Realiza um crossover combinado entre o 'self' e outro indivíduo."""
        alpha = random.random()
        child_distP = alpha * self.distP + (1 - alpha) * other.distP
        child_angleP = alpha * self.angleP + (1 - alpha) * other.angleP
        return Individual(child_distP, child_angleP, id=id)

    def mutate(self, mutation_rate=0.1, mutation_strength=0.5):
        """Muta os genes do indivíduo com uma dada taxa e força de mutação."""
        if random.random() < mutation_rate:
            self.distP += random.gauss(0, mutation_strength)
        if random.random() < mutation_rate:
            self.angleP += random.gauss(0, mutation_strength)

        # Garante que os parâmetros se mantêm dentro de uma gama válida
        self.distP = np.clip(self.distP, 0.1, 15.0)
        self.angleP = np.clip(self.angleP, 0.0, 5.0)

