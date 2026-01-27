from controllers.Controller import Controller
from os import environ
from dotenv import load_dotenv
import numpy as np
import jax.numpy as jnp
from jax import nn

class NN_controller(Controller):
    def __init__(self, learning_rate: float = 0.01):
        load_dotenv()
        super().__init__(learning_rate)
        self.layers_and_nodes = environ.get("LAYERS_AND_NODES")
        self.weights_upper_boundry = float(environ.get("WEIGHT_UPPER_BOUNDRY"))
        self.weights_lower_boundry = float(environ.get("WEIGHT_LOWER_BOUNDRY"))
        self.bias_upper_boundry = float(environ.get("BIAS_UPPER_BOUNDRY"))
        self.bias_lower_boundry = float(environ.get("BIAS_LOWER_BOUNDRY"))

        if self.layers_and_nodes == "":
            self.layers_and_nodes = []
        else:
            self.layers_and_nodes = [int(value) for value in self.layers_and_nodes.split(",")]
        
        self.activation_functions = self.load_activation_functions()
        self.collection_parameters = self.generate_weights_and_bias()
    
    def generate_control_signal(self, error: float, parameters):
        self.error_history.append(error)
        error_derivative = error - self.error_history[-2] if len(self.error_history) > 1 else 0.0
        signal = jnp.array([error, error_derivative, sum(self.error_history)]).T
        print("parameters:", parameters)
        print("Initial signal:", signal)

        for activation_function, (weights, bias) in zip(self.activation_functions, parameters):
            signal = activation_function(jnp.dot(signal, weights) + bias)
            print("Signal after layer:", signal)
        
        print("Raw signal:", signal)
        assert signal.shape == (1, 1)
        signal = signal.squeeze()
        assert signal.shape == ()
        print("compressed signal:", signal)
        return signal

    def reset_state(self):
        self.error_history.clear()

    def update_error_history(self):
        raise NotImplementedError("update_error_history()")
    
    def load_activation_functions(self):
        load_functions = environ.get("ACTIVATION_FUNCTIONS")
        activation_functions = []
        if load_functions == "":
            activation_functions.append(lambda x: x)
            return activation_functions
        else:
            for function in load_functions.split(","):
                print(function)
                match function:
                    case "sigmoid":
                        activation_functions.append(nn.sigmoid)
                    case "tanh":
                        activation_functions.append(nn.tanh)
                    case "relu":
                        activation_functions.append(nn.relu)
            activation_functions.append(lambda x: x)
            return activation_functions

    def generate_weights_and_bias(self):
        input = 3
        collection_parameters = []
        for output in self.layers_and_nodes+[1]:
            weights = (np.random.uniform(self.weights_lower_boundry, self.weights_upper_boundry, (input, output)))
            bias = (np.random.uniform(self.bias_lower_boundry, self.bias_upper_boundry, (1, output)))
            input = output
            collection_parameters.append((weights, bias))
        return collection_parameters