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
        self.layers_and_nodes = [int(value) for value in self.layers_and_nodes.split(",")]
        self.activation_function = self.load_activation_function()


        self.collection_parameters = {

        }
    
    def generate_control_signal(self):
        raise NotImplementedError("generate_control_signal()")
  
    def reset_state(self):
        raise NotImplementedError("reset_state()")

    def update_error_history(self):
        raise NotImplementedError("update_error_history()")
    
    def load_activation_function(self):
        activation_function = environ.get("ACTIVATION_FUNCTION")
        match activation_function:
            case "sigmoid":
                return nn.sigmoid
            case "tanh":
                return nn.tanh
            case "relu":
                return nn.relu
            case "linear":
                return lambda f: f
    
    def generate_weights_and_bias(self, layers_and_nodes):
        pass