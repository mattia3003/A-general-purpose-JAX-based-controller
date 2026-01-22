from abc import ABC, abstractmethod

class Plant(ABC):
    def __init__(self):
        self.initial_state = 0.0
        self.target_state = 0.0

    @abstractmethod
    def generate_plant_output(self):
        raise NotImplementedError("generate_plant_outout()")

    @abstractmethod
    def reset_state(self):
        raise NotImplementedError("reset_state()")
    
    @abstractmethod
    def get_initial_and_target_state(self):
        raise NotImplementedError("get_initial_and_goal_state()")