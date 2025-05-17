import numpy as np


class MLPController:
    def __init__(self, input_size, hidden_size, output_size, weights_vector=None):
        """
        Initializes a single-hidden-layer MLP with specified sizes.
        If weights_vector is given, it's used to set the weights and biases directly.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Total number of weights and biases
        self.total_weights = (
            input_size * hidden_size +  # input-to-hidden weights
            hidden_size +               # hidden biases
            hidden_size * output_size + # hidden-to-output weights
            output_size                 # output biases
        )

        if weights_vector is not None:
            self.set_weights(weights_vector)
        else:
            # Initialize randomly if no weights are provided
            self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.1
            self.bias_hidden = np.random.randn(hidden_size) * 0.1
            self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.1
            self.bias_output = np.random.randn(output_size) * 0.1

    
    def set_weights(self, vector):
        """
        Sets the weights and biases from a flat vector.
        """
        assert len(vector) == self.total_weights, "Invalid weight vector size"
        idx = 0

        ih_size = self.input_size * self.hidden_size
        self.weights_input_hidden = vector[idx:idx+ih_size].reshape(self.input_size, self.hidden_size)
        idx += ih_size

        self.bias_hidden = vector[idx:idx+self.hidden_size]
        idx += self.hidden_size

        ho_size = self.hidden_size * self.output_size
        self.weights_hidden_output = vector[idx:idx+ho_size].reshape(self.hidden_size, self.output_size)
        idx += ho_size

        self.bias_output = vector[idx:idx+self.output_size]
        if not np.all(np.isfinite(self.weights_input_hidden)) or \
       not np.all(np.isfinite(self.bias_hidden)) or \
       not np.all(np.isfinite(self.weights_hidden_output)) or \
       not np.all(np.isfinite(self.bias_output)):
            raise ValueError("[FATAL] set_weights recebeu valores inválidos (NaN/inf)")



    def forward(self, x):
        """
        Forward pass through the network.
        """
        hidden = np.dot(x, self.weights_input_hidden) + self.bias_hidden
        hidden = np.maximum(0, hidden)  # ReLU activation
        output = np.dot(hidden, self.weights_hidden_output) + self.bias_output

        return np.clip(output, -1.0, 1.0)  # Limita a saída entre -1 e 1