from plants.Plant import Plant
from os import environ
from dotenv import load_dotenv
import jax.numpy as jnp

class Bathtub_plant(Plant):
    def __init__(self):
        load_dotenv()
        super().__init__()
        self.get_initial_and_target_state()
        self.water_height = self.initial_state
        self.target_height = self.target_state
        self.area = float(environ.get("CSA_A"))
        self.drain = self.area / float(environ.get("CSA_C_DIVIDER"))

    def generate_plant_output(self, control_signal: float, disturbance: float):
        flow_rate = self.calculate_flow_rate()
        volume_change = control_signal + disturbance - flow_rate
        water_height_change = volume_change / self.area
        self.water_height = self.water_height + water_height_change
        return self.water_height

    def reset_state(self):
        self.water_height = self.initial_state
    
    def calculate_velocity(self):
        return jnp.sqrt(2 * 9.8 * self.water_height)
    
    def calculate_flow_rate(self):
        return self.drain * self.calculate_velocity()
    
    def get_initial_and_target_state(self):
        self.initial_state = float(environ.get("INITIAL_HEIGHT"))
        self.target_state = float(environ.get("TARGET_HEIGHT"))