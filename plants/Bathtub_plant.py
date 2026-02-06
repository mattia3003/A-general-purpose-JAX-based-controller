from plants.Plant import Plant
from os import environ
from dotenv import load_dotenv
import jax.numpy as jnp

# class for the bathtub plant, which inherits from the base plant class and implements generate_plant_output and get_initial_and_target_state
class Bathtub_plant(Plant):
    def __init__(self):
        load_dotenv()
        super().__init__()
        self.get_initial_and_target_state()
        self.water_height = self.initial_state
        self.target_height = self.target_state
        self.area = float(environ.get("CSA_A"))
        self.drain = float(environ.get("CSA_C"))
    
    # function for calculating the velocity of the water flowing out of the bathtub based on the current water height
    def calculate_velocity(self):
        return jnp.sqrt(2 * 9.8 * self.water_height)

    # function for calculating the flow rate of the water flowing out of the bathtub based on the velocity and the drain size
    def calculate_flow_rate(self):
        return self.drain * self.calculate_velocity()

    # function for generating the plant output based on the control signal, disturbance and plant dynamics
    def generate_plant_output(self, control_signal: float, disturbance: float):
        flow_rate = self.calculate_flow_rate()
        volume_change = control_signal + disturbance - flow_rate
        self.water_height = jnp.maximum(0.0, self.water_height + (volume_change) / self.area)
        print("Water height:", self.water_height)

        return self.water_height
    
    # function for getting the initial and target state of the plant from the .env file
    def get_initial_and_target_state(self):
        self.initial_state = float(environ.get("INITIAL_HEIGHT"))
        self.target_state = float(environ.get("TARGET_HEIGHT"))