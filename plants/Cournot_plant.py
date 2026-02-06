from plants.Plant import Plant
from os import environ
from dotenv import load_dotenv
import jax.numpy as jnp

# class for the Cournot plant, which inherits from the base plant class and implements generate_plant_output and get_initial_and_target_state
class Cournot_plant(Plant):
    def __init__(self):
        load_dotenv()
        super().__init__()
        self.get_initial_and_target_state()
        self.production = self.initial_state
        self.target_profit = self.target_state
        self.competitor_production = self.production
        self.max_price = float(environ.get("MAX_PRICE"))
        self.production_cost = float(environ.get("PRODUCTION_COST"))
        self.profit = 0.0

    # function for generating the plant output based on the control signal, disturbance and plant dynamics
    def generate_plant_output(self, control_signal: float, disturbance: float):
        self.production += control_signal
        self.competitor_production += disturbance
        total_production = jnp.clip(self.production, 0.0, 1.0) + jnp.clip(self.competitor_production, 0.0, 1.0)
        price = jnp.maximum(self.max_price - total_production, 0.0)
        self.profit = self.production * (price - self.production_cost)
        print("profit:", self.profit)
        return self.profit
    
    # function for getting the initial and target state of the plant from the .env file
    def get_initial_and_target_state(self):
        self.initial_state = float(environ.get("INITIAL_PRODUCTION"))
        self.target_state = float(environ.get("TARGET_PROFIT"))