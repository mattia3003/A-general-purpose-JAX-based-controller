from controllers.PID_controller import PID_controller
from plants.Bathtub_plant import Bathtub_plant
from plants.Cournot_plant import Cournot_plant
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

    def run_system(self):
        jax_gradient_function = value_and_grad(self.run_epoch)
        mse_error_history = []
        for _ in range(self.epochs):
            epoch_mse, gradient = jax_gradient_function(self.controller.collection_parameters)
            mse_error_history.append(epoch_mse)
            #print(gradient)
            self.controller.update_parameters(gradient)
        print(len(mse_error_history))
        draw_PID_parameters(self.controller.historical_k_p, self.controller.historical_k_i, self.controller.historical_k_d)
        draw_error_history(mse_error_history)

    def run_epoch(self, collection_parameters):
        self.controller.reset_state()
        self.plant.reset_state()
        disturbance = self.generate_disturbance(self.timesteps, self.disturbance_lower_boundry, self.disturbance_upper_boundry)
        plant_states = jnp.array([])
        control_signal = 0.0
        error_sum = 0.0
        for i in range(self.timesteps):
            plant_output, next_signal = self.run_timestep(control_signal, disturbance[i], collection_parameters, error_sum)
            plant_states = jnp.append(plant_states, plant_output)
            control_signal = next_signal
            error_sum += self.controller.last_error
        mse = self.generate_mean_square_error(plant_states, self.target_state)
        return mse
    
    def run_timestep(self, control_signal, disturbance, collection_parameters, error_sum):
        plant_output = self.plant.generate_plant_output(control_signal, disturbance)
        new_signal = self.controller.generate_control_signal(self.target_state, plant_output, collection_parameters, self.controller.error_history, error_sum)
        return plant_output, new_signal

    def generate_mean_square_error(self, generated_states: list, target: float):
        print(generated_states)
        mse = jnp.mean(jnp.square(target - generated_states))
        print(mse)
        return mse

    def generate_disturbance(self, timesteps, disturbance_lower_bound, disturbance_upper_bound):
        key = random.key(0)
        #for _ in range(timesteps):
        disturbance = uniform(key, shape=timesteps, dtype=jnp.float32, minval=disturbance_lower_bound, maxval=disturbance_upper_bound)
          #disturbance.append(np.uniform(disturbance_lower_bound, disturbance_upper_bound)) 
        return disturbance


system = CONSYS(PID_controller, Bathtub_plant)
system.run_system()