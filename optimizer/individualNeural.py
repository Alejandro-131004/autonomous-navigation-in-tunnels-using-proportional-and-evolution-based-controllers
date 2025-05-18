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
        self.id = id 

        if weights_vector is not None:
            if not np.all(np.isfinite(weights_vector)):
                print("[ERROR] Received invalid weights_vector. Reinitializing.")
                weights_vector = None  # fallback to random
        self.controller = MLPController(input_size, hidden_size, output_size, weights_vector)

    def get_genome(self):
        """
        Returns the genome (flattened weights and biases) of this individual.
        """
        return np.concatenate([
            self.controller.weights_input_hidden.flatten(),
            self.controller.bias_hidden,
            self.controller.weights_hidden_output.flatten(),
            self.controller.bias_output
        ])

    def mutate(self, mutation_rate=0.1, mutation_strength=0.1):
        """
        Applies Gaussian noise to some genes with a given probability.
        """
        genome = self.get_genome()
        for i in range(len(genome)):
            if random.random() < mutation_rate:
                genome[i] += np.random.normal(0, mutation_strength)

        # ✅ Proteção contra NaN ou inf
        if not np.all(np.isfinite(genome)):
            print("[ERROR] Mutated genome is invalid. Reinitializing individual.")
            genome = np.random.randn(len(genome)) * 0.1  # regenerar com valores pequenos

        self.controller.set_weights(genome)

    def crossover(self, other):
        """
        Performs arithmetic crossover between two individuals.
        """
        genome1 = self.get_genome()
        genome2 = other.get_genome()
        alpha = np.random.uniform(0, 1, size=genome1.shape)
        child_genome = alpha * genome1 + (1 - alpha) * genome2

        # ✅ Verifica validade
        if not np.all(np.isfinite(child_genome)):
            print("[ERROR] Crossover result invalid. Creating random child.")
            child_genome = np.random.randn(len(genome1)) * 0.1

        return IndividualNeural(self.input_size, self.hidden_size, self.output_size, child_genome)

    def act(self, lidar_input):
        """
        Executes the neural network on LIDAR input and returns safe motor commands.
        """
        lidar_input = np.nan_to_num(lidar_input, nan=0.0, posinf=10.0, neginf=0.0)
        lidar_input = np.clip(lidar_input, 0, 3.0) / 3.0  # alcance max é 3m

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
