# Chapter two Page-22
def gradient_descent_fun(time_horizon, initial_x, step_sizes):
    '''
    最原始的梯度下降法 Example: f(x,y) = x^2 + y^2 +2
    :param time_horizon: 迭代次数
    :param initial_x: 初始点-用数组的形式 例如：numpy.array([1,2])
    :param step_sizes: 迭代步长
    :return: 返回最小值
    '''
    x_value = initial_x
    for i in range(time_horizon):
        x_value = x_value - step_sizes * 2 * x_value
    return x_value

