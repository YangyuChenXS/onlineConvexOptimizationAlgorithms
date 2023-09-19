from scipy.optimize import minimize
import numpy
import math

if __name__ == '__main__':
    # e = 1e-10 # 非常接近0的值
    # fun = lambda x : (x[0] - 0.667) / (x[0] + x[1] + x[2] - 2) # 约束函数
    # cons = ({'type': 'eq', 'fun': lambda x: x[0] * x[1] * x[2] - 1}, # xyz=1
    #         {'type': 'ineq', 'fun': lambda x: x[0] - e}, # x>=e，即 x > 0
    #         {'type': 'ineq', 'fun': lambda x: x[1] - e},
    #         {'type': 'ineq', 'fun': lambda x: x[2] - e}
    #        )
    # x0 = numpy.array((1.0, 1.0, 1.0)) # 设置初始值
    # res = minimize(fun, x0, method='SLSQP', constraints=cons)
    # print('最小值：',res.fun)
    # print('最优解：',res.x)
    # print('迭代终止是否成功：', res.success)
    # print('迭代终止原因：', res.message)

    # matrix_a = numpy.mat([[1, 2], [3, 3]])
    # # sum_value = numpy.sum(matrix_a*[[1], [0]])
    # # print(sum_value)
    # fun = lambda x: numpy.sum(matrix_a*numpy.array([x]).T)  # 这里的fun是固定名字，改变会报错
    # cons = ({'type': 'ineq', 'fun': lambda x: x[0] - 2},
    #         {'type': 'ineq', 'fun': lambda x: x[1] - 3}
    #        )
    # x0 = numpy.array((50.0, 100.0)) # 设置初始值
    # res = minimize(fun, x0, method='SLSQP', constraints=cons)
    # print('最小值：',res.fun)
    # print('最优解：',res.x)
    # print('迭代终止是否成功：', res.success)
    # print('迭代终止原因：', res.message)

    fun = lambda x: math.pow(x[0], 2) + math.pow(x[1], 2)
    cons = ({'type': 'ineq', 'fun': lambda x: x[0] + 100},
            {'type': 'ineq', 'fun': lambda x: -x[0] + 100},
            {'type': 'ineq', 'fun': lambda x: x[1] + 100},
            {'type': 'ineq', 'fun': lambda x: -x[1] + 100}
            )
    compute_x_initial = numpy.array((100.0, 100.0))  # 设置初始值
    x_one_value = minimize(fun, compute_x_initial, method='SLSQP', constraints=cons)
    print(x_one_value.fun)
    print(x_one_value.x)
    x_initial_one = x_one_value.x
    print(type(x_initial_one))