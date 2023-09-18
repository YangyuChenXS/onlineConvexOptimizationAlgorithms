from scipy.optimize import minimize
import numpy


# Chapter 5 Page 74
def regularized_follow_the_leader(time_horizon, eta_value = ):
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
    :param eta_value:
    :return:
    """

