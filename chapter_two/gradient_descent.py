# Chapter two Page-22
import numpy
from scipy.optimize import minimize


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
    min_f_value = initial_x[0] ** 2 + initial_x[1] ** 2 + 2
    x_bar = initial_x
    for i in range(time_horizon):
        temp_value = x_value_matrix[i][0] * x_value_matrix[i][0] + x_value_matrix[i][1] * x_value_matrix[i][1] + 2
        if min_f_value >= temp_value:
            min_f_value = temp_value
            x_bar = x_value_matrix[i]
    return x_bar


##############################################################################################################
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
        step_sizes = (x_value[0] ** 2 + x_value[1] ** 2 + 2 - 2) / (
                    4 * x_value[0] * x_value[0] + 4 * x_value[1] * x_value[1])
        x_value = x_value - step_sizes * 2 * x_value
        x_value_matrix = numpy.vstack((x_value_matrix, x_value))  # 将本次x_value存入x_value_vector
    min_f_value = initial_x[0] ** 2 + initial_x[1] ** 2 + 2
    x_bar = initial_x
    for i in range(time_horizon):
        temp_value = x_value_matrix[i][0] * x_value_matrix[i][0] + x_value_matrix[i][1] * x_value_matrix[i][1] + 2
        if min_f_value >= temp_value:
            min_f_value = temp_value
            x_bar = x_value_matrix[i]
    return x_bar


##############################################################################################################
# Chapter 3 Page-27
def basic_gradient_descent(time_horizon, initial_x, step_sizes):
    """
    Basic gradient descent
    Example: f(x,y) = (1/2)x^2 + y^2 +2  这里定义域限定在{(x,y)| 1<=x<=3, 2<=y<=5 }
    求f(x,y)的Hessen矩阵后，\beta可以取2, 所以这里的step_sizes=1/2
    :param time_horizon: 迭代次数
    :param initial_x:  初始点-用数组的形式 例如：numpy.array([1,2])
    :param step_sizes: 步长，根据算法定义step_zises=\frac{1}{\beta}, 这里的\beta是 \beta-smooth凸函数
    :return:
    """
    x_value = initial_x
    for i in range(time_horizon):
        delta_value = numpy.array([x_value[0], 2 * x_value[1]])
        x_value = x_value - step_sizes * delta_value
        print(x_value)

        # 需要做投射，这里看成一个凸优化问题-最小化欧氏距离
        # https://blog.csdn.net/qq_44444503/article/details/124194711?ops_request_misc=&request_id=&biz_id=102&utm_term=python%20%20%E6%9C%80%E4%BC%98%E5%8C%96%E6%B1%82%E8%A7%A3&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-124194711.142^v68^control,201^v4^add_ask,213^v2^t3_esquery_v3&spm=1018.2226.3001.4187
        # 这里为了实现后面的basic_gradient_descent算法，定义了一些函数, 例子是f(x,y) = x^2 + y^2 +2 这里定义域限定在{(x,y)| 1<=x<=3, 2<=y<=5 }
        # 定义目标函数
        def func(x, sign=1.0):
            # scipy.minimize默认求最小，求max时只需要sign*(-1)，跟下面的args对应
            return sign * 0.5 * ((x[0] - x_value[0]) ** 2 + (x[1] - x_value[1]) ** 2 + 2)

        # 定义目标函数的梯度
        def func_deriv(x, sign=1):
            jac_x0 = sign * ( x[0] - x_value[0])
            jac_x1 = sign * (2 * (x[1] - x_value[1]))
            return numpy.array([jac_x0, jac_x1])

        # 定义约束条件
        cons = (
            {'type': 'ineq',
             'fun': lambda x: numpy.array([x[0] - 1]),
             'jac': lambda x: numpy.array([1, 0])
             },
            {'type': 'ineq',
             'fun': lambda x: numpy.array([-x[0] + 3]),
             'jac': lambda x: numpy.array([-1, 0])
             },

            {'type': 'ineq',
             'fun': lambda x: numpy.array([x[1] - 2]),
             'jac': lambda x: numpy.array([0, 1])
             },

            {'type': 'ineq',
             'fun': lambda x: numpy.array([-x[1] + 5]),
             'jac': lambda x: numpy.array([0, -1])
             }
        )

        # 定义初始解x0
        x0 = numpy.array([0, 0])
        # 使用SLSQP算法求解
        res = minimize(func, x0, args=(1,), jac=func_deriv, method='SLSQP', options={'disp': False}, constraints=cons)
        x_value = res.x
        print(x_value)
    return x_value
