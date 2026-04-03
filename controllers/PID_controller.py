from controllers.Controller import Controller
from os import environ
from dotenv import load_dotenv
import jax.numpy as jnp

# class for the PID controller, which inherits from the base controller class and implements generate_control_signal
class PID_controller(Controller):
    def __init__(self, learning_rate: float = 0.01):
        load_dotenv()
        super().__init__(learning_rate)
    
    # function for generating the control signal based on the plant error and the parameters of the PID controller
    def generate_control_signal(self, error: float, parameters):
        self.error_history.append(error)
        error_derivative = error - self.error_history[-2] if len(self.error_history) > 1 else 0.0
        change = jnp.array([error, error_derivative, sum(self.error_history)])

        signal = jnp.dot(parameters, change)
        
        return signal