import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)

import math
import time
from chapter_three import first_order_algorithms
import numpy
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # ################################################测试online_gradient_descent#################
    # initial_x = numpy.array([20, 40])
    # time_horizon = 1000
    # x_value_vector = first_order_algorithms.online_gradient_descent(time_horizon, initial_x)
    # for i in range(len(x_value_vector)):
    #     print(x_value_vector[i])
    # # 查看画出regret
    # x_value_picture = []
    # y_value_picture = []
    # regret_value = 0
    # for i in range(len(x_value_vector)):
    #     x_value_picture.append(i+1)
    #     regret_value = regret_value + (x_value_vector[i][0]-1)**2 + x_value_vector[i][1]**2
    #     y_value_picture.append(regret_value)
    # for i in range(len(x_value_vector)):
    #     y_value_picture[i] = y_value_picture[i]/(i+1)  # 截至到第t轮的平均regret
    # fig, axes = plt.subplots(1, 1)  # 定义几张图
    # fig.set_tight_layout(True)  # 紧凑布局
    # fig.set_size_inches(8, 4.5)  # 定义画布大小
    #
    # axes.plot(x_value_picture, y_value_picture, '-', label="regret")
    #
    # plt.grid(ls='--', lw=0.01, color='black')  # 生成网格
    # plt.legend(fontsize=20, loc='upper right')  # 标签字体大小
    # plt.xticks(fontsize=15)  # x轴刻度字体大小
    # plt.yticks(fontsize=15)  # y轴刻度字体大小
    # plt.xlim(0, time_horizon+1)
    # plt.ylim(0, 100)
    # plt.show()
    # ################################################测试proving_thm_three_two#################
    # first_order_algorithms.proving_thm_three_two_last_equality(time_horizon=200)

    ##################################################测试stochastic_gradient_descent_svm_training#######
    time_horizon = 10000
    #train_set = numpy.array([[1, 0, 1], [0, 1, -1]])  # numpy.sum(x_bar * numpy.array([0, 1]))
    #train_set = numpy.array([[1, 0, 0, 1], [1, 1, 0, 1], [0, 0, 1, -1]])
    train_set = numpy.array([[1, 0, 0, 0, 1], [1, 0, 0, 1,  1], [0, 1, 0, 0, -1], [0, 1, 1, 0, -1], [0, 0, 1, 0, -1]])
    initial_x = numpy.array([0, 0, 0, 0])
    lamda = 1
    # 对比gradient_descent.svm_training_via_subgradient_descent(time_horizon, train_set, initial_x, lamda)速度会更快
    start = time.perf_counter()
    x_bar = first_order_algorithms.stochastic_gradient_descent_svm_training(time_horizon, train_set, initial_x, lamda)
    end = time.perf_counter()
    runTime = end - start
    print("运行时间：", runTime)
    print(x_bar)

