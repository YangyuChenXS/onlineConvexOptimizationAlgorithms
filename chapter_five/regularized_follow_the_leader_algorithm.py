from scipy.optimize import minimize
import numpy
import math


def obj_fun(x, iteration_round):
    """
    这里以函数f_t(x,y) = (x-1)^2 + y^2 + logt 为例
        定义域{(x,y)| -100<=x<=100, -100<=y<=100}
        所以 定义域大小上界D=200\sqrt{2}
            梯度范数上界 G=2\sqrt{101^2+100^2}
    :param x: x为二元向量, 类型为numpy的一维数组
    :param iteration_round: 迭代的轮数，从1开始计算
    :return:
    """
    return math.pow(x[0]-1, 2) + math.pow(x[1], 2) + math.log(iteration_round)


def regularizaion_function(x):
    """
    正则化函数为 R(x,y) = x^2 + y^2
    :param x: x为二元向量, 类型为numpy的一维数组
    :return:
    """
    return math.pow(x[0], 2) + math.pow(x[1], 2)


# Chapter 5 Page 74
def regularized_follow_the_leader(time_horizon):
    """
    实现 Regularized Follow The Leader Alogrithm
    这里以函数f_t(x,y) = (x-1)^2 + y^2 + logt 为例
        定义域{(x,y)| -100<=x<=100, -100<=y<=100}
        所以 定义域大小上界D=200\sqrt{2}
            梯度范数上界 G=2\sqrt{101^2+100^2}
       正则化函数为 R(x,y) = x^2 + y^2
           G_R = 101^2+{100^2}/4
           eta_value = 1/{\sqrt{2*T}*G_R}
    :param time_horizon:
    :return:
    """
    g_r_value = math.pow(101, 2) + math.pow(100, 2)/4
    eta_value = 1/(math.sqrt(2*time_horizon)*g_r_value)
    fun = lambda x: math.pow(x[0], 2) + math.pow(x[1], 2)
    cons = ({'type': 'ineq', 'fun': lambda x: x[0] + 100},
            {'type': 'ineq', 'fun': lambda x: -x[0] + 100},
            {'type': 'ineq', 'fun': lambda x: x[1] + 100},
            {'type': 'ineq', 'fun': lambda x: -x[1] + 100}
            )
    compute_x_initial = numpy.array((100.0, 100.0))  # 设置初始值
    x_one_value = minimize(fun, compute_x_initial, method='SLSQP', constraints=cons)
    # print(x_one_value.fun)
    for t in range(time_horizon):
        pass
    pass


