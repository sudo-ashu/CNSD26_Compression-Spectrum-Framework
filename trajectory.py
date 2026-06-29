import numpy as np

def get_trajectory_logistic(initial_value, length, a):
    """
    Logistic map:
        x_{n+1} = a * x_n * (1 - x_n)

    Returns:
        trajectory
        Lyapunov exponent
    """

    x = np.zeros(length)
    x[0] = initial_value
    lyap = 0

    for i in range(length - 1):
        x[i + 1] = a * x[i] * (1 - x[i])
        derivative = abs(a * (1 - 2 * x[i]))

        if derivative > 0:
            lyap += np.log(derivative)
    lyap /= (length - 1)
    return x, lyap