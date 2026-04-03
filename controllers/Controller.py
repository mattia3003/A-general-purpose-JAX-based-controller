
# base controller class
class Controller():
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.error_history = []

    # function to be implemented by the specific controller
    def generate_control_signal(self):
        # raise NotImplementedError("generate_control_signal()")
        pass