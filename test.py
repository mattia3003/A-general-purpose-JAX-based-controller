import numpy as np
from controllers.PID_controller import PID_controller
#from plants import Plant, Bathtub_plant
from utils.graphs import draw_PID_parameters, draw_error_history


controller = PID_controller()

print(controller.learning_rate)
print(controller.error_history)
controller.update_error_history(10.0)
print(controller.error_history)

print(controller.k_p)
print(controller.k_i)
print(controller.k_d)

draw_PID_parameters(controller.historical_k_p, controller.historical_k_i, controller.historical_k_d)
draw_error_history(controller.error_history)

elements = [1, 2, 3]
number = 2

print(number * elements)