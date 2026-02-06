
# base plant class
class Plant():
    def __init__(self):
        self.initial_state = 0.0
        self.target_state = 0.0

    # function to be implemented by the specific plant
    def generate_plant_output(self):
        # raise NotImplementedError("generate_plant_outout()")
        pass

    # function to be implemented by the specific plant
    def get_initial_and_target_state(self):
        # raise NotImplementedError("get_initial_and_goal_state()")
        pass