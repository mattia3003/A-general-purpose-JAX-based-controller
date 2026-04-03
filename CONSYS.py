from controllers.PID_controller import PID_controller
from controllers.NN_controller import NN_controller
from plants.Bathtub_plant import Bathtub_plant
from plants.Cournot_plant import Cournot_plant
from plants.Population_plant import Population_plant
from utils.graphs import draw_error_history, draw_PID_parameters
from utils.parameter_functions import generate_weights_and_bias, update_nn_params, update_pid_params, update_historical_parameters

from os import environ
from dotenv import load_dotenv
import numpy as np
import jax.numpy as jnp
from jax import jit, value_and_grad

# function for loading the controller and plant based on the .env file
def load_controller_and_plant():
    controller_type = environ.get("CONTROLLER_TYPE")
    plant_type = environ.get("PLANT_TYPE")
    learning_rate = float(environ.get("LEARNING_RATE"))

    if controller_type == "pid":
        controller = PID_controller(learning_rate)
    elif controller_type == "nn":
        controller = NN_controller(learning_rate)
    else:
        raise ValueError("Invalid CONTROLLER_TYPE")

    if plant_type == "bathtub":
        plant = Bathtub_plant()
    elif plant_type == "cournot":
        plant = Cournot_plant()
    elif plant_type == "population":
        plant = Population_plant()
    else:
        raise ValueError("Invalid PLANT_TYPE")

    return controller, plant

# function for generating disturbance
def generate_disturbance(timesteps, disturbance_lower_bound, disturbance_upper_bound):
    disturbance = np.random.uniform(disturbance_lower_bound, disturbance_upper_bound, timesteps)
    return disturbance

# function for running a single timestep of the system
def run_timestep(plant, controller, target, control_signal, disturbance, collection_parameters):
    error = target - plant.generate_plant_output(control_signal, disturbance)
    new_signal = controller.generate_control_signal(error, collection_parameters)
    return error, new_signal

# function for running a single epoch of the system
def run_epoch(collection_parameters):
    controller, plant = load_controller_and_plant()
    timesteps = int(environ.get("TIMESTEPS"))
    target_state = plant.target_state
    disturbance_lower_boundry = float(environ.get("DISTURBANCE_LOWER_BOUNDRY"))
    disturbance_upper_boundry = float(environ.get("DISTURBANCE_UPPER_BOUNDRY"))
    disturbance = generate_disturbance(timesteps, disturbance_lower_boundry, disturbance_upper_boundry)
    epoch_errors = []
    control_signal = 0.0
    for i in range(timesteps):
        timestep_error, next_signal = run_timestep(plant, controller, target_state, control_signal, disturbance[i], collection_parameters)
        epoch_errors.append(timestep_error)
        control_signal = next_signal
    return jnp.mean(jnp.square(jnp.array(epoch_errors)))

# function for running the entire system
def run_system():
    DEV = True if environ.get("DEV") == "true" else False
    PID = True if environ.get("CONTROLLER_TYPE") == "pid" else False
    params = None
    historical_k_p = []
    historical_k_d = []
    historical_k_i = []

    if PID:
        params = jnp.array([float(environ.get("K_P")), float(environ.get("K_D")), float(environ.get("K_I"))])
    else:
        params = generate_weights_and_bias(environ.get("LAYERS_AND_NODES"), float(environ.get("WEIGHT_LOWER_BOUNDRY")), float(environ.get("WEIGHT_UPPER_BOUNDRY")), float(environ.get("BIAS_LOWER_BOUNDRY")), float(environ.get("BIAS_UPPER_BOUNDRY")))

    epochs = int(environ.get("EPOCHS"))
    learning_rate = float(environ.get("LEARNING_RATE"))
    mse_error_history = []
    jax_gradient_function = value_and_grad(run_epoch, argnums=0)
    jit_func = jit(jax_gradient_function)

    for _ in range(epochs):
        epoch_mse, gradient = jit_func(params)
        mse_error_history.append(epoch_mse)

        if PID:
            params = update_pid_params(params, gradient, learning_rate)
            update_historical_parameters(historical_k_p, historical_k_d, historical_k_i, params)
        else:
            params = update_nn_params(params, gradient, learning_rate)

        if (DEV):
            print(f"Epoch: {_+1}/{epochs}")
            print(f"MSE: {epoch_mse}")
            print("-------------------------")

    if PID:
        draw_PID_parameters(historical_k_p, historical_k_d, historical_k_i)
    draw_error_history(mse_error_history)


if __name__ == "__main__":
    load_dotenv()
    run_system()