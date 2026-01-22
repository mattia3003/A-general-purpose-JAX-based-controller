from abc import ABC, abstractmethod

class Controller(ABC):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.current_error = 0.0
        self.last_error = 0.0
        #self.current_error_sum = 0.0
        self.current_error_change = 0.0
        self.error_history = [0.0]

    @abstractmethod
    def generate_control_signal(self):
        raise NotImplementedError("generate_control_signal()")

    @abstractmethod
    def reset_state(self):
        raise NotImplementedError("reset_state()")
    
    def error_change(self):
        return self.last_error - self.current_error

    @abstractmethod
    def update_error_history(self):
        raise NotImplementedError("update_error_history()")