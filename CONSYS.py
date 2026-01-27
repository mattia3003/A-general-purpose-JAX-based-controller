from controllers.PID_controller import PID_controller
from controllers.NN_controller import NN_controller
from plants.Bathtub_plant import Bathtub_plant
from plants.Cournot_plant import Cournot_plant
from plants.Population_plant import Population_plant
from utils.graphs import draw_error_history, draw_PID_parameters


from os import environ
from dotenv import load_dotenv
import numpy as np
import jax.numpy as jnp
from jax import jit, value_and_grad, random
from jax.random import PRNGKey, uniform

class CONSYS():
    def __init__(self, controller, plant):
        load_dotenv()
        self.controller = controller(float(environ.get("LEARNING_RATE")))
        self.plant = plant()
        self.target_state = self.plant.target_state
        self.epochs = int(environ.get("EPOCHS"))
        self.timesteps = int(environ.get("TIMESTEPS"))
        self.disturbance_upper_boundry = float(environ.get("DISTURBANCE_UPPER_BOUNDRY"))
        self.disturbance_lower_boundry = float(environ.get("DISTURBANCE_LOWER_BOUNDRY"))

        self.historical_k_p = []
        self.historical_k_d = []
        self.historical_k_i = []
    
    def run_timestep(self, plant, controller, target, control_signal, disturbance, collection_parameters):
        error = target - plant.generate_plant_output(control_signal, disturbance)
        new_signal = controller.generate_control_signal(error, collection_parameters)
        return error, new_signal
    
    def run_epoch(self, collection_parameters):
        controller = NN_controller(float(environ.get("LEARNING_RATE")))
        plant = Bathtub_plant()
        #self.controller.reset_state()
        #self.plant.reset_state()
        disturbance = self.generate_disturbance(self.timesteps, self.disturbance_lower_boundry, self.disturbance_upper_boundry)
        epoch_errors = []
        control_signal = 0.0
        for i in range(self.timesteps):
            timestep_error, next_signal = self.run_timestep(plant, controller, self.target_state, control_signal, disturbance[i], collection_parameters)
            epoch_errors.append(timestep_error)
            control_signal = next_signal
        print("controller error history:", controller.error_history)
        mse = jnp.mean(jnp.square(jnp.array(epoch_errors)))

        return mse

    def run_system(self):
        params = self.controller.collection_parameters
        mse_error_history = []
        jax_gradient_function = value_and_grad(self.run_epoch, argnums=0)
        jit_func = jit(jax_gradient_function)
        for _ in range(self.epochs):
            epoch_mse, gradient = jit_func(params)
            mse_error_history.append(epoch_mse)
            print("gradient:", gradient)
            #print("params before update:", params)
            #params = self.update_params(params, gradient)
            params = self.update_nn_params(params, gradient)
            #self.update_historical_parameters(params)
            #print("params after update:", params)
            #self.controller.update_parameters(gradient)
        #draw_PID_parameters(self.historical_k_p, self.historical_k_d, self.historical_k_i)
        draw_error_history(mse_error_history)

    def generate_mean_square_error(self, generated_states: list, target: float):
        #print(generated_states)
        mse = jnp.mean(jnp.square(target - jnp.array(generated_states)))
        #print(mse)
        return mse

    def generate_disturbance(self, timesteps, disturbance_lower_bound, disturbance_upper_bound):
        disturbance = np.random.uniform(disturbance_lower_bound, disturbance_upper_bound, timesteps)
        return disturbance

    def update_params(self, old_params, gradient):
        new_params = old_params - self.controller.learning_rate * gradient
        return new_params

    def update_nn_params(self, old_params, gradient):
        print("Old params:", old_params)
        new_params = [[old - gradient * self.controller.learning_rate for old, gradient in zip(param_layer, grad_layer)] for param_layer, grad_layer in zip(old_params, gradient)]
        print("New params:", new_params)
        return new_params

    def update_historical_parameters(self, params):
        self.historical_k_p.append(params[0])
        self.historical_k_d.append(params[1])
        self.historical_k_i.append(params[2])


system = CONSYS(NN_controller, Bathtub_plant)
system.run_system()