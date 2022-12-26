import numpy
import math
import numpy
import matplotlib.pyplot as plt

# Chapter 3 Page-42
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


# Chapter 3 Page-42 Theorem 3.2
def proving_thm_three_two(time_horizon):
    """
    # 验证Theorem 3.2最后一个不等式  C取1.6的时候最佳
    :param time_horizon:
    :return:
    """
    y_value = []
    y_sqrt_value = []
    x_value=[]
    temp_one = 1
    temp_two = 1
    temp_three = 1
    for t in range(2, time_horizon+2, 2):
        x_value.append(t)
        y_sqrt_value.append(math.sqrt(t))
        for i in range(1, t):
            temp_one = temp_one*i
        for i in range(1, t//2+1):
            temp_two = temp_two*i
        for i in range(1, t//2):
            temp_three = temp_three*i
        y_value.append( t/(2**(t-2)) * (temp_one/temp_two/temp_three) )
        temp_one = 1
        temp_two = 1
        temp_three = 1
    #print(y_value)
    #print(y_sqrt_value)
    for i in range(len(y_sqrt_value)):
        y_sqrt_value[i] = y_sqrt_value[i]*1.6  # 下界C值

    difference_value = [] # sqrt_value 与 y_value的差值
    for i in range(len(y_sqrt_value)):
        difference_value.append(y_sqrt_value[i]-y_value[i])
    #print(difference_value)

    fig, axes = plt.subplots(1, 1)  # 定义几张图
    fig.set_tight_layout(True)  # 紧凑布局
    fig.set_size_inches(8, 4.5)  # 定义画布大小

    axes.plot(x_value, y_value, 'o-', label="expectations")
    axes.plot(x_value, y_sqrt_value, 'o-', label="sqrt_value")
    axes.plot(x_value, difference_value, '-', label="difference_value")

    plt.grid(ls='--', lw=0.01, color='black')  # 生成网格
    # plt.legend(fontsize=20, loc='upper left')  # 标签字体大小
    plt.legend(fontsize=20)  # 标签字体大小
    plt.xticks(fontsize=15)  # x轴刻度字体大小
    plt.yticks(fontsize=15)  # y轴刻度字体大小
    plt.xlim(0, time_horizon+0.5)
    # plt.ylim(0, 100)
    plt.show()

