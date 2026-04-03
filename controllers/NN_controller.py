from controllers.Controller import Controller
from os import environ
from dotenv import load_dotenv
import numpy as np
import jax.numpy as jnp
from jax import nn

# class for the neural network controller, which inherits from the base controller class and implements generate_control_signal
class NN_controller(Controller):
    def __init__(self, learning_rate: float = 0.01):
        load_dotenv()
        super().__init__(learning_rate)

        self.activation_functions = self.load_activation_functions()
    
    # function for generating the control signal based on the plant error and the parameters of the neural network
    def generate_control_signal(self, error: float, parameters):
        self.error_history.append(error)
        error_derivative = error - self.error_history[-2] if len(self.error_history) > 1 else 0.0
        signal = jnp.array([error, error_derivative, sum(self.error_history)]).T

        for activation_function, (weights, bias) in zip(self.activation_functions, parameters):
            if activation_function == "sigmoid":
                signal = activation_function(jnp.dot(signal, weights) + bias)
                assert signal.shape[0] == 1
            else:
                signal = activation_function(jnp.dot(signal, weights) + bias)
            
        assert signal.shape == (1, 1)
        signal = signal.squeeze()
        assert signal.shape == ()
        return signal
    
    # function for loading the activation functions for the neural network based on the .env file
    def load_activation_functions(self):
        load_functions = environ.get("ACTIVATION_FUNCTIONS")
        activation_functions = []
        if load_functions == "":
            activation_functions.append(lambda x: x)
            return activation_functions
        else:
            for function in load_functions.split(","):
                match function:
                    case "sigmoid":
                        activation_functions.append(nn.sigmoid)
                    case "tanh":
                        activation_functions.append(nn.tanh)
                    case "relu":
                        activation_functions.append(nn.relu)
            activation_functions.append(lambda x: x)
            return activation_functions