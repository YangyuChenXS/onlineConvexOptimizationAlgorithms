import numpy
def online_gradient_descent(time_horizon, initial_x):
    """
    与上一章的 basic gradient descent 最大的不同是每一轮的函数可以是不同的
    这里以函数f_t(x,y) = (x-1)^2 + y^2 + logt 为例
        定义域{(x,y)| -100<=x<=100, -100<=y<=100}
        所以 定义域大小上界D=200\sqrt{2}
            梯度范数上界 G=2\sqrt{101^2+100^2}
    此外，这里不做投射->若需要投射，变成求范数最小值的优化问题即可
    :param time_horizon: 迭代次数
    :param initial_x: 初始点-用数组的形式 例如：numpy.array([1,2])
    :return: x_value_vector
    """
    x_value_vector = []  # 用来存储每一轮的决策
    x_value = initial_x
    x_value_vector.append(initial_x)
    d_value = 200*numpy.sqrt(2)
    g_value = 2 * numpy.sqrt(101**2 + 100**2)
    for t in range(time_horizon):
        eta_value = d_value/g_value/numpy.sqrt(t+1)
        gradient_value = numpy.array([2*(x_value[0]-1), 2*x_value[1]])
        if t < time_horizon-1:
            x_value = x_value - eta_value * gradient_value
            x_value_vector.append(x_value)
        # 这里不做投射->若需要投射，变成求范数最小值的优化问题即可
    return x_value_vector  # 理论上是不需要返回的，决策结束
