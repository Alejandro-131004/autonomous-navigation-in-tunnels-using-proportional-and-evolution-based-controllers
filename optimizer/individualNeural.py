from optimizer.mlpController import MLPController
import numpy as np
import random
# Importar as novas constantes de velocidade
from environment.configuration import MIN_VELOCITY, MAX_VELOCITY


class IndividualNeural:
    def __init__(self, input_size, hidden_size, output_size, weights_vector=None, id=None):
        """
        Representa um indivíduo com um controlador neural.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fitness = None
        self.total_successes = 0  # Nome da variável corrigido para consistência
        self.id = id

        if weights_vector is not None:
            if not np.all(np.isfinite(weights_vector)):
                print("[ERRO] Recebido 'weights_vector' inválido. A reiniciar.")
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
            print("[ERRO] O genoma após mutação é inválido. A reiniciar o indivíduo.")
            genome = np.random.randn(len(genome)) * 0.1

        self.controller.set_weights(genome)

    def crossover(self, other, id=None):
        genome1 = self.get_genome()
        genome2 = other.get_genome()
        alpha = np.random.uniform(0, 1, size=genome1.shape)
        child_genome = alpha * genome1 + (1 - alpha) * genome2

        if not np.all(np.isfinite(child_genome)):
            print("[ERRO] O resultado do crossover é inválido. A criar um descendente aleatório.")
            child_genome = np.random.randn(len(genome1)) * 0.1

        return IndividualNeural(self.input_size, self.hidden_size, self.output_size, child_genome, id=id)

    def act(self, lidar_input):
        lidar_input = np.nan_to_num(lidar_input, nan=0.0, posinf=10.0, neginf=0.0)
        lidar_input = np.clip(lidar_input, 0, 3.0) / 3.0

        # A saída da rede é normalizada (-1 a 1)
        output = self.controller.forward(lidar_input)

        # --- LÓGICA DE VELOCIDADE ATUALIZADA ---
        # Mapeia a saída de velocidade linear [-1, 1] para o intervalo [MIN_VELOCITY, MAX_VELOCITY]
        # O valor +1 garante que o resultado é sempre positivo (0 a 2), e o resto escala para o intervalo desejado.
        linear_output = output[0]
        lv = MIN_VELOCITY + (linear_output + 1) * 0.5 * (MAX_VELOCITY - MIN_VELOCITY)

        # A velocidade angular é mapeada para [-MAX_VELOCITY*2, MAX_VELOCITY*2] para permitir viragens rápidas
        angular_output = output[1]
        av = angular_output * (MAX_VELOCITY * 2)
        # --- FIM DA LÓGICA DE VELOCIDADE ATUALIZADA ---

        if not np.isfinite(lv):
            print("[AVISO] A velocidade linear é NaN ou inf. A definir para 0.")
            lv = 0.0
        if not np.isfinite(av):
            print("[AVISO] A velocidade angular é NaN ou inf. A definir para 0.")
            av = 0.0

        return lv, av
