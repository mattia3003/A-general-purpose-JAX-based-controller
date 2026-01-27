from plants.Plant import Plant
from os import environ
from dotenv import load_dotenv
import jax.numpy as jnp

class Population_plant(Plant):
    def __init__(self):
        load_dotenv()
        super().__init__()
        self.get_initial_and_target_state()
        self.population = self.initial_state
        self.target_population = self.target_state
        self.ferility_rate = float(environ.get("FERTILITY_RATE"))
        self.mortality_rate = float(environ.get("MORTALITY_RATE"))
        self.carrying_capacity = float(environ.get("CARRYING_CAPACITY"))

    def generate_plant_output(self, control_signal: float, disturbance: float):
        current_density = self.population / self.carrying_capacity
        growth_factor = 1 - current_density
        natural_growth = (self.ferility_rate*growth_factor - self.mortality_rate) * self.population
        self.population = jnp.maximum(0.0, self.population + natural_growth + control_signal + disturbance)
        print("Population:", self.population)
        return self.population

    def reset_state(self):
        self.population = self.initial_state
    
    def get_initial_and_target_state(self):
        self.initial_state = float(environ.get("INITIAL_POPULATION"))
        self.target_state = float(environ.get("TARGET_POPULATION"))