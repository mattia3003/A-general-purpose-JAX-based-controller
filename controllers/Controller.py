from abc import ABC, abstractmethod

class Controller(ABC):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.current_error = 0.0
        self.last_error = 0.0
        self.current_error_change = 0.0
        self.error_history = []

    # @abstractmethod
    def generate_control_signal(self):
        # raise NotImplementedError("generate_control_signal()")
        pass

    # @abstractmethod
    def reset_state(self):
        # raise NotImplementedError("reset_state()")
        pass

    # @abstractmethod
    def update_error_history(self):
        # raise NotImplementedError("update_error_history()")
        pass