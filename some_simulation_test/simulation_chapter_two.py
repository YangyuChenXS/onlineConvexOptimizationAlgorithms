from chapter_two import gradient_descent
import numpy
import time
from scipy.optimize import minimize


if __name__ == '__main__':
    # ################################################测试gradient_descent_fun#################
    # time_horizon = 100
    # initial_x = numpy.array([100, 50])
    # step_sizes = 0.2
    # x_value = gradient_descent.gradient_descent_fun(time_horizon, initial_x, step_sizes)
    # print(x_value)
    # ##################################################测试gradient_descent_polyak_stepsize#######
    # x_value = gradient_descent.gradient_descent_polyak_stepsize(time_horizon, initial_x)
    # print(x_value)
    #
    # # 定义约束条件
    # # cons = (
    # #         {'type': 'ineq',
    # #          'fun': lambda x: numpy.array([x[0] - 1]),
    # #          'jac': lambda x: numpy.array([1, 0])
    # #          },
    # #         {'type': 'ineq',
    # #          'fun': lambda x: numpy.array([-x[0] + 3]),
    # #          'jac': lambda x: numpy.array([-1, 0])
    # #          },
    # #
    # #         {'type': 'ineq',
    # #          'fun': lambda x: numpy.array([x[1] - 2]),
    # #          'jac': lambda x: numpy.array([0, 1])
    # #          },
    # #
    # #         {'type': 'ineq',
    # #          'fun': lambda x: numpy.array([-x[1] + 5]),
    # #          'jac': lambda x: numpy.array([0, -1])
    # #          }
    # #     )
    # #
    # # # 定义初始解x0
    # # x0 = numpy.array([0, 0])
    # # y_value = numpy.array([6, 6])
    # #
    # # def func(x, sign=1.0):
    # #     # scipy.minimize默认求最小，求max时只需要sign*(-1)，跟下面的args对应
    # #     return sign * ((x[0] - y_value[0]) ** 2 + (x[1] - y_value[1]) ** 2 + 2)
    # # # 定义目标函数的梯度
    # # def func_deriv(x, sign=1):
    # #     jac_x0 = sign * (2 * (x[0] -  y_value[0]))
    # #     jac_x1 = sign * (2 * (x[1] -  y_value[1]))
    # #     return numpy.array([jac_x0, jac_x1])
    # #
    # # # 使用SLSQP算法求解
    # # res = minimize(func, x0, args=(1,), jac=func_deriv, method='SLSQP', options={'disp': False}, constraints=cons)
    # # # args是传递给目标函数和偏导的参数，此例中为1，求min问题。args=-1时是求解max问题
    # # print(res.x)
    #
    ##################################################测试basic_gradient_descent#######
    # time_horizon = 10
    # initial_x = numpy.array([4, 6])
    # step_sizes = 0.5
    # x_value = gradient_descent.basic_gradient_descent(time_horizon, initial_x, step_sizes)
    # print(x_value)
    ##################################################测试svm_training_via_subgradient_descent#######
    time_horizon = 10000
    #train_set = numpy.array([[1, 0, 1], [0, 1, -1]])  # numpy.sum(x_bar * numpy.array([0, 1]))
    #train_set = numpy.array([[1, 0, 0, 1], [1, 1, 0, 1], [0, 0, 1, -1]])
    train_set = numpy.array([[1, 0, 0, 0, 1], [1, 0, 0, 1,  1], [0, 1, 0, 0, -1], [0, 1, 1, 0, -1], [0, 0, 1, 0, -1]])
    initial_x = numpy.array([0, 0, 0, 0])
    lamda = 1
    start = time.perf_counter()
    x_bar = gradient_descent.svm_training_via_subgradient_descent(time_horizon, train_set, initial_x, lamda)
    end = time.perf_counter()
    runTime = end - start
    print("运行时间：", runTime)
    print(x_bar)


