from optimizer.mlpController import MLPController
import numpy as np
import random

class IndividualNeural:
    def __init__(self, input_size, hidden_size, output_size, weights_vector=None, id=None):
        """
        Represents an individual with a neural controller encoded as a flat weight vector.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fitness = None
        self.avg_fitness = None
        self.successes = 0
        self.id = id

        if weights_vector is not None:
            if not np.all(np.isfinite(weights_vector)):
                print("[ERROR] Received invalid weights_vector. Reinitializing.")
                weights_vector = None

        self.controller = MLPController(input_size, hidden_size, output_size, weights_vector)

    def get_genome(self):
        return np.concatenate([
            self.controller.weights_input_hidden.flatten(),
            self.controller.bias_hidden,
            self.controller.weights_hidden_output.flatten(),
            self.controller.bias_output
        ])

    def mutate(self, mutation_rate=0.1, mutation_strength=0.1):
        genome = self.get_genome()
        for i in range(len(genome)):
            if random.random() < mutation_rate:
                genome[i] += np.random.normal(0, mutation_strength)

        if not np.all(np.isfinite(genome)):
            print("[ERROR] Mutated genome is invalid. Reinitializing individual.")
            genome = np.random.randn(len(genome)) * 0.1

        self.controller.set_weights(genome)

    def crossover(self, other, id=None):
        genome1 = self.get_genome()
        genome2 = other.get_genome()
        alpha = np.random.uniform(0, 1, size=genome1.shape)
        child_genome = alpha * genome1 + (1 - alpha) * genome2

        if not np.all(np.isfinite(child_genome)):
            print("[ERROR] Crossover result invalid. Creating random child.")
            child_genome = np.random.randn(len(genome1)) * 0.1

        return IndividualNeural(self.input_size, self.hidden_size, self.output_size, child_genome, id=id)

    def act(self, lidar_input):
        lidar_input = np.nan_to_num(lidar_input, nan=0.0, posinf=10.0, neginf=0.0)
        lidar_input = np.clip(lidar_input, 0, 3.0) / 3.0  # Normalize

        output = self.controller.forward(lidar_input)
        lv, av = output[0], output[1]

        lv = np.clip(lv, -1.0, 1.0)
        av = np.clip(av, -1.0, 1.0)

        if not np.isfinite(lv):
            print("[WARNING] Linear velocity is NaN or inf. Setting to 0.")
            lv = 0.0
        if not np.isfinite(av):
            print("[WARNING] Angular velocity is NaN or inf. Setting to 0.")
            av = 0.0

        return lv, av
