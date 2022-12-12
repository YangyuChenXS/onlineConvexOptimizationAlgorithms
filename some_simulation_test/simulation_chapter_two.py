from chapter_two import gradient_descent
import numpy

if __name__ == '__main__':
    time_horizon = 10
    initial_x = numpy.array([100, 50])
    step_sizes = 0.2
    x_value = gradient_descent.gradient_descent_fun(time_horizon, initial_x, step_sizes)
    print(x_value)

    x_value = gradient_descent.gradient_descent_polyak_stepsize(time_horizon, initial_x)
    print(x_value)