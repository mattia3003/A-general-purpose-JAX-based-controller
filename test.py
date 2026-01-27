import jax.numpy as jnp
from controllers.PID_controller import PID_controller
#from plants import Plant, Bathtub_plant
from utils.graphs import draw_PID_parameters, draw_error_history

def generate_mean_square_error(generated_states: list, target: float):
    mse = jnp.mean(jnp.square(target - generated_states))
    return mse

print(generate_mean_square_error([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 4.0))