from optimizer.mlpController import MLPController
import numpy as np
import random
from environment.configuration import MIN_VELOCITY, MAX_VELOCITY


class IndividualNeural:
    def __init__(self, input_size, hidden_size, output_size, weights_vector=None, id=None):
        """
        Represents an individual with a neural controller.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fitness = None
        self.total_successes = 0  # Variable name corrected for consistency
        self.id = id

        if weights_vector is not None:
            if not np.all(np.isfinite(weights_vector)):
                print("[ERROR] Received invalid 'weights_vector'. Resetting.")
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
            print("[ERROR] Genome after mutation is invalid. Resetting individual.")
            genome = np.random.randn(len(genome)) * 0.1

        self.controller.set_weights(genome)

    def crossover(self, other, id=None):
        genome1 = self.get_genome()
        genome2 = other.get_genome()
        alpha = np.random.uniform(0, 1, size=genome1.shape)
        child_genome = alpha * genome1 + (1 - alpha) * genome2

        if not np.all(np.isfinite(child_genome)):
            print("[ERROR] Crossover result is invalid. Generating random offspring.")
            child_genome = np.random.randn(len(genome1)) * 0.1

        return IndividualNeural(self.input_size, self.hidden_size, self.output_size, child_genome, id=id)

    def act(self, lidar_input):
        lidar_input = np.nan_to_num(lidar_input, nan=0.0, posinf=10.0, neginf=0.0)
        lidar_input = np.clip(lidar_input, 0, 3.0) / 3.0

        # Network output is normalized (-1 to 1)
        output = self.controller.forward(lidar_input)

        # --- UPDATED VELOCITY LOGIC ---
        # Maps the linear velocity output [-1, 1] to [MIN_VELOCITY, MAX_VELOCITY]
        # The +1 shifts to [0, 2], then scales it to the desired range
        linear_output = output[0]
        lv = MIN_VELOCITY + (linear_output + 1) * 0.5 * (MAX_VELOCITY - MIN_VELOCITY) * 1.2
        lv = min(lv, MAX_VELOCITY)

        # Angular velocity is mapped to [-2*MAX_VELOCITY, 2*MAX_VELOCITY] for fast turns
        angular_output = output[1]
        av = angular_output * (MAX_VELOCITY * 2)
        # --- END UPDATED VELOCITY LOGIC ---

        if not np.isfinite(lv):
            print("[WARNING] Linear velocity is NaN or inf. Setting to 0.")
            lv = 0.0
        if not np.isfinite(av):
            print("[WARNING] Angular velocity is NaN or inf. Setting to 0.")
            av = 0.0

        return lv, av
