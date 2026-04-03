import numpy as np
import jax.numpy as jnp
from jax import tree

# function for generating random weights and bias for the NN-controller based on the .env file
def generate_weights_and_bias(layers_and_nodes_str: str, weights_lower_boundry: float, weights_upper_boundry: float, bias_lower_boundry: float, bias_upper_boundry: float):
    if layers_and_nodes_str == "":
        layers_and_nodes = []
    else:
        layers_and_nodes = [int(value) for value in layers_and_nodes_str.split(",")]

    input = 3
    collection_parameters = []
    for output in layers_and_nodes+[1]:
        weights = (np.random.uniform(weights_lower_boundry, weights_upper_boundry, (input, output)))
        bias = (np.random.uniform(bias_lower_boundry, bias_upper_boundry, (1, output)))
        input = output
        collection_parameters.append((weights, bias))

    return collection_parameters

# function for updating the parameters of the PID-controller
def update_pid_params(old_params, gradient, learning_rate):
    new_params = old_params - learning_rate * gradient
    return new_params

# function for updating the parameters of the NN-controller
def update_nn_params(old_params, gradient, learning_rate):
    clipped_gradient = tree.map(lambda x: jnp.clip(x, -1.0, 1.0), gradient)
    new_params = [[old - clipped_grad * learning_rate for old, clipped_grad in zip(param_layer, grad_layer)] for param_layer, grad_layer in zip(old_params, clipped_gradient)]
    return new_params

# function for updating the historical parameters of the PID-controller for graphing purposes
def update_historical_parameters(historical_k_p, historical_k_d, historical_k_i, params):
    historical_k_p.append(params[0])
    historical_k_d.append(params[1])
    historical_k_i.append(params[2])