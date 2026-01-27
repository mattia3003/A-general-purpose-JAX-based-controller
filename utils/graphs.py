import matplotlib.pyplot as plt

def draw_PID_parameters(historical_k_p: list, historical_k_d: list, historical_k_i: list):
    plt.title("PID parameters")
    plt.plot(historical_k_p, label = "k_p values", color = "blue")
    plt.plot(historical_k_d, label = "k_d values", color = "orange")
    plt.plot(historical_k_i, label = "k_i values", color = "green")
    plt.legend()
    #plt.savefig("PID_parameters_pop_jit.png")
    plt.show()

def draw_error_history(error_history: list):
    plt.title("error history")
    plt.plot(error_history, label = "error", color = "black")
    plt.legend()
    #plt.savefig("error_history_pop_jit.png")
    plt.show()