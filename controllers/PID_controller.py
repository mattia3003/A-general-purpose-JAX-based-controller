from controllers.Controller import Controller
from os import environ
from dotenv import load_dotenv
import numpy as np

class PID_controller(Controller):
    def __init__(self, learning_rate: float = 0.01):
        load_dotenv()
        super().__init__(learning_rate)
        self.k_p = float(environ.get("K_P"))
        self.k_i = float(environ.get("K_I"))
        self.k_d = float(environ.get("K_D"))

        self.collection_parameters = {
            "k_p": self.k_p,
            "k_i": self.k_i,
            "k_d": self.k_d,
        }

        self.historical_k_p = []
        self.historical_k_i = []
        self.historical_k_d = []

        self.update_historical_parameters()
    
    def generate_control_signal(self, target_state: float, current_state: float, parameters: dict, error_history: list, error_sum: float):
        self.current_error = target_state - current_state
        self.update_error_history(self.current_error)
        self.current_error_change = error_history[-2] - error_history[-1]

        self.last_error = self.current_error

        signal = parameters["k_p"] * self.current_error + parameters["k_i"] * (error_sum + self.current_error) + parameters["k_d"] * self.current_error_change
        return signal
    
    def reset_state(self):
        self.current_error = 0.0
        self.last_error = 0.0
        #self.current_sum = 0.0
        self.current_error_change = 0.0
        self.error_history.clear()
        self.error_history.append(0.0)
    
    def update_parameters(self, gradient: dict):
        self.collection_parameters["k_p"] = self.collection_parameters["k_p"] - self.learning_rate * gradient["k_p"]
        self.collection_parameters["k_i"] = self.collection_parameters["k_i"] - self.learning_rate * gradient["k_i"]
        self.collection_parameters["k_d"] = self.collection_parameters["k_d"] - self.learning_rate * gradient["k_d"]
        self.update_historical_parameters()
    
    def update_historical_parameters(self):
        self.historical_k_p.append(self.collection_parameters["k_p"])
        self.historical_k_i.append(self.collection_parameters["k_i"])
        self.historical_k_d.append(self.collection_parameters["k_d"])
    
    def update_error_history(self, new_error):
        self.error_history.append(new_error)