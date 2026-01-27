from controllers.Controller import Controller
from os import environ
from dotenv import load_dotenv
import jax.numpy as jnp

class PID_controller(Controller):
    def __init__(self, learning_rate: float = 0.01):
        load_dotenv()
        super().__init__(learning_rate)
        self.k_p = float(environ.get("K_P"))
        self.k_d = float(environ.get("K_D"))
        self.k_i = float(environ.get("K_I"))

        self.collection_parameters = jnp.array([self.k_p, self.k_d, self.k_i])

        self.historical_k_p = []
        self.historical_k_d = []
        self.historical_k_i = []

        self.update_historical_parameters()
    
    def generate_control_signal(self, error: float, parameters):
    
        self.error_history.append(error)
        print("error", error)
        error_derivative = error - self.error_history[-2] if len(self.error_history) > 1 else 0.0
        change = jnp.array([error, error_derivative, sum(self.error_history)])
        print("Change:", change)

        signal = jnp.dot(parameters, change)

        print("Control signal:", signal)
        
        return signal
    
    def reset_state(self):
        self.current_error = 0.0
        self.last_error = 0.0
        self.current_error_change = 0.0
        self.error_history.clear()
    
    def update_parameters(self, gradient):
        self.collection_parameters = self.collection_parameters - self.learning_rate * gradient
        #print("Updated parameters to:", self.collection_parameters)
        self.update_historical_parameters()
    
    def update_historical_parameters(self):
        self.historical_k_p.append(self.collection_parameters[0])
        self.historical_k_d.append(self.collection_parameters[1])
        self.historical_k_i.append(self.collection_parameters[2])
    
    def update_error_history(self, new_error):
        self.error_history.append(new_error)