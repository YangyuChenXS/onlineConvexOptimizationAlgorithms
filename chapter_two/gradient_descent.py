# Chapter two Page-22
import numpy


# Chapter 2 Page-22
def gradient_descent_fun(time_horizon, initial_x, step_sizes):
    '''
    最原始的梯度下降法 Example: f(x,y) = x^2 + y^2 +2
    :param time_horizon: 迭代次数
    :param initial_x: 初始点-用数组的形式 例如：numpy.array([1,2])
    :param step_sizes: 迭代步长
    :return: 返回使得f(x)最小的x
    '''
    x_value = initial_x
    x_value_matrix = initial_x  # 存储历次的X的迭代值
    for i in range(time_horizon):
        x_value = x_value - step_sizes * 2 * x_value
        x_value_matrix = numpy.vstack((x_value_matrix, x_value))  # 将本次x_value存入x_value_vector
    min_f_value = initial_x[0]**2 + initial_x[1]**2 + 2
    x_bar = initial_x
    for i in range(time_horizon):
        temp_value = x_value_matrix[i][0]*x_value_matrix[i][0] + x_value_matrix[i][1]*x_value_matrix[i][1] + 2
        if min_f_value >= temp_value:
            min_f_value = temp_value
            x_bar = x_value_matrix[i]
    return x_bar

# Chapter 2 Page-22
def gradient_descent_polyak_stepsize(time_horizon, initial_x):
    """
    Gradient descent with ployak stepsize, 这个方法的缺点是步长依赖于最优值f(x^*),即step_sizes=[f(x_t)-f(x^*)]/梯度范数的平方
    Example: f(x,y) = x^2 + y^2 +2
    :param time_horizon: 迭代次数
    :param initial_x: 初始点-用数组的形式 例如：numpy.array([1,2])
    :return: 返回使得f(x)最小的x
    """
    x_value = initial_x
    x_value_matrix = initial_x  # 存储历次的X的迭代值
    min_f_ture_value = 2  # 在这个例子下，最小值为2
    for i in range(time_horizon):
        step_sizes = (x_value[0]**2 + x_value[1]**2 + 2 - 2)/(4*x_value[0]*x_value[0]+4*x_value[1]*x_value[1])
        x_value = x_value - step_sizes * 2 * x_value
        x_value_matrix = numpy.vstack((x_value_matrix, x_value))  # 将本次x_value存入x_value_vector
    min_f_value = initial_x[0]**2 + initial_x[1]**2 + 2
    x_bar = initial_x
    for i in range(time_horizon):
        temp_value = x_value_matrix[i][0]*x_value_matrix[i][0] + x_value_matrix[i][1]*x_value_matrix[i][1] + 2
        if min_f_value >= temp_value:
            min_f_value = temp_value
            x_bar = x_value_matrix[i]
    return x_bar






