import numpy
import math
import numpy
import matplotlib.pyplot as plt
import random


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
def proving_thm_three_two_last_equality(time_horizon):
    """
    # 验证Theorem 3.2最后一个不等式  C取0.8左右的时候最佳，这与理论分析吻合
    :param time_horizon:
    :return:
    """
    y_value = []
    y_sqrt_value = []
    divide_value = [] # y_value除以sqrt_value
    x_value = []
    temp_one = 1
    temp_two = 1
    #temp_three = 1
    for t in range(2, time_horizon+2, 2):
        x_value.append(t)
        y_sqrt_value.append(math.sqrt(t))
        for i in range(1, t+1):
            temp_one = temp_one*i
        for i in range(1, t//2+1):
            temp_two = temp_two*i
        # for i in range(1, t//2):
        #     temp_three = temp_three*i
        y_value.append( t/(2**(t)) * (temp_one/temp_two/temp_two) )
        divide_value.append( math.sqrt(t)/(2**(t)) * (temp_one/temp_two/temp_two) )
        temp_one = 1
        temp_two = 1
        #temp_three = 1
    #print(y_value)
    #print(y_sqrt_value)
    print(divide_value)
    for i in range(len(y_sqrt_value)):
        y_sqrt_value[i] = y_sqrt_value[i]*0.8  # 下界C值

    difference_value = []  # y_value与sqrt_value的差值
    for i in range(len(y_sqrt_value)):
        difference_value.append(y_value[i]-y_sqrt_value[i])
    print(difference_value)

    fig, axes = plt.subplots(1, 1)  # 定义几张图
    fig.set_tight_layout(True)  # 紧凑布局
    fig.set_size_inches(8, 4.5)  # 定义画布大小

    axes.plot(x_value, y_value, 'o-', label="expectations")
    axes.plot(x_value, y_sqrt_value, 'o-', label="sqrt_value")
    axes.plot(x_value, difference_value, '-', label="difference_value")
    axes.plot(x_value, divide_value, '-', label="divide_value")

    plt.grid(ls='--', lw=0.01, color='black')  # 生成网格
    # plt.legend(fontsize=20, loc='upper left')  # 标签字体大小
    plt.legend(fontsize=20)  # 标签字体大小
    plt.xticks(fontsize=15)  # x轴刻度字体大小
    plt.yticks(fontsize=15)  # y轴刻度字体大小
    plt.xlim(0, time_horizon+0.5)
    # plt.ylim(0, 100)
    plt.show()


# Chapter 3 Page-49 Algorithm 9
def stochastic_gradient_descent():
    """
    抽象算法
    :return:
    """
    pass


##############################################################################################################
# Chapter 3 Page-50 Algorithm 10
def stochastic_gradient_descent_svm_training(time_horizon, train_set, initial_x, lamda):
    """
    使用SGD算法重新解决Chapter 2 Page-31的问题，计算速度会变快
    SVM training via subgradient descent  # 针对邮件问题，求的超平面也是过原点的，要注意
    :param time_horizon: 迭代轮数
    :param train_set: 训练集，这里传入的是一个n*m维数组，即有n个行向量，每一个行向量的前m-1个元素是特征[注意按照这里的设定，元素取值为0或1]，第m个是标签1或-1
    :param initial_x: 初始值取0  m-1维的数组
    :param lamda: 参数
    :return: 超平面【注意按照这里的设定，这个超平面会过原点】
    """
    x_value = initial_x
    length = train_set.shape
    x_return = initial_x
    for i in range(time_horizon):
        t_random = random.randint(0, length[0]-1)
        a_value = train_set[t_random][:length[1] - 1]
        if train_set[t_random][length[1] - 1] * numpy.sum(x_value * a_value) <= 1:
            single_subgradient = lamda * (- train_set[t_random][length[1] - 1] * a_value) + x_value
        else:
            single_subgradient = x_value
        x_value = x_value - 1 / math.sqrt(i+1) * single_subgradient
        if i != time_horizon-1:
            x_return = x_return + 2*(i+1)/time_horizon * x_value
    x_return = x_return / time_horizon
    return x_return




