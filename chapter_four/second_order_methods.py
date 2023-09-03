import math
import numpy

# Chapter 4 Page 62
def online_newton_step(time_horizon,x_initial):
    """
    实现online Newton step算法
    注意这里矩阵求逆运算使用了The Sherman-Morrison formula,a.k.a. the matrix inversin lemma-Page 66
    这里以函数f_t(w_1,w_2) = (w_1-1)^2 
        定义域{(w_1,w_2)| 0<=w_1<=1, 0<=w_2<=1}
        所以 定义域大小上界D=\sqrt{2} # 这里指的是upper bound on Euclidean diameter
        This function is convex function and 0.5-exp-concave function, \alpha = 0.5
        梯度上界G为2  upper bound on norm of subgradients
    此外，这里不做投射->若需要投射，变成求范数最小值的优化问题即可
    :param time_horizon: 决策轮数
    :param x_initial: 初始值, 列向量,  用numpy.mat表示
    :return: 
    """
    g_value = 2  # upper bound on norm of subgradients
    d_value = math.sqrt(2)  # upper bound on Euclidean diameter
    alpha_value = 0.5  # 0.5-exp-concave function
    min_array = [1/g_value/d_value, alpha_value]
    gamma_value = 0.5*min(min_array)  # the condition of theorem
    epsilon_value = 1/(gamma_value*gamma_value*d_value*d_value) # the condition of theorem
    x_value = x_initial
    capital_a_inverse = epsilon_value * numpy.mat([[1,0],[0,1]])
    regret = math.pow(x_initial[0,0]-1,2) - 0
    for t_round in range(time_horizon):
        x_component_one = x_value[0,0]
        gradient_t_round = numpy.mat([[2*(x_component_one-1)],[0]])
        capital_a_inverse = capital_a_inverse - capital_a_inverse*gradient_t_round*numpy.transpose(gradient_t_round)*capital_a_inverse/(1+numpy.transpose(gradient_t_round)*capital_a_inverse*gradient_t_round)
        x_value = x_value - 1/gamma_value * capital_a_inverse * gradient_t_round
        regret = regret + math.pow(x_value[0,0]-1,2) - 0
        print(regret - 2*(1/alpha_value+g_value*d_value)*2*math.log(t_round+2))
        




if __name__ == '__main__':
    """     
    matrix_a = numpy.mat([[1,0],[0,1]])
    matrix_a_inverse = numpy.linalg.inv(matrix_a)
    print(matrix_a_inverse)
    x_initial = numpy.mat([[0],[1]])
    print(numpy.dot(x_initial, numpy.transpose(x_initial)))
    print(x_initial[1][0])
    print(5*numpy.mat([[1,0],[0,1]]))
    print(x_initial*numpy.transpose(x_initial)) 
    """
    time_horizon = 100
    x_initial= numpy.mat([[0],[0]])
    online_newton_step(time_horizon,x_initial)
    
    

