import math

import numpy


def generate_real_decisions():
    """
    这里产生每轮正确的决定，这里的逻辑也是随机生成。可按需修改产生逻辑
    :return: 返回每轮正确的决定
    """
    real_decision = numpy.random.choice([0, 1], size=1, p=[0.01, 0.99])
    return real_decision


def expert_make_decision(n_expert):
    """
    这里的专家决策逻辑是随机生成0或者1，也就是专家只做可选两个的决策。可按需修改其它决策逻辑
    0表示决策A，1表示决策B
    :param n_expert: n个专家
    :return: 返回n个专家的决策
    """
    # 等概率随机生成0或1
    expert_decision = numpy.random.choice([0, 1], size=n_expert, p=[0.5, 0.5])
    # expert_decision[0] = 0
    # expert_decision[n_expert-1] = 1
    return expert_decision


def loss_value(n_expert):
    """
    可以按需定义损失逻辑
    :param n_expert: 专家人数
    :return: 返回n个专家的损失值
    """
    loss_vector = numpy.random.uniform(0, 1, (1, n_expert))[0]
    return loss_vector

# Chapter 1 Page-9
def the_weighted_majority(n_expert, t_round, epsilon):
    """
    决策A用0表示，决策B用1表示
    举手表决，少数服从多数
    :param n_expert: 专家人数
    :param t_round: 决策轮数
    :param epsilon: 算法参数 值在(0,1)
    :return: 返回算法每轮做的决定
    """
    expert_weight = numpy.ones(n_expert)
    weight_sum_a = 0
    weight_sum_b = 0
    algorithm_decision = []
    for i_round in range(t_round):
        # print(expert_weight)
        # 获取n个专家的建议
        expert_decision = expert_make_decision(n_expert)
        # 算法处理专家权重
        for i in range(n_expert):
            if expert_decision[i] == 0:
                weight_sum_a = weight_sum_a + expert_weight[i]
            else:
                weight_sum_b = weight_sum_b + expert_weight[i]
        # 产生算法本轮所做的决定
        if weight_sum_a > weight_sum_b:
            algorithm_decision.append(0)
        else:
            algorithm_decision.append(1)
        # 获得本轮正确的决定
        real_decision = generate_real_decisions()
        # 更新权重
        for i in range(n_expert):
            if expert_decision[i] != real_decision:
                expert_weight[i] = expert_weight[i] * (1 - epsilon)
    return algorithm_decision

# Chapter 1 Page-11
def randomized_weighted_majority(n_expert, t_round, epsilon):
    """
    决策A用0表示，决策B用1表示
    按照投票的比例作为概率，产生决定
    :param n_expert: 专家人数
    :param t_round: 决策轮数
    :param epsilon: 算法参数 值在(0,1)
    :return: 返回算法每轮做的决定
    """
    expert_weight = numpy.ones(n_expert)
    weight_sum_a = 0
    weight_sum_b = 0
    algorithm_decision = []
    for i_round in range(t_round):
        # print(expert_weight)
        # 获取n个专家的建议
        expert_decision = expert_make_decision(n_expert)
        # 算法处理专家权重
        for i in range(n_expert):
            if expert_decision[i] == 0:
                weight_sum_a = weight_sum_a + expert_weight[i]
            else:
                weight_sum_b = weight_sum_b + expert_weight[i]
        weight_sum = weight_sum_a + weight_sum_b
        # 产生算法本轮所做的决定
        temp = numpy.random.choice([0, 1], size=1, p=[weight_sum_a / weight_sum, weight_sum_b / weight_sum])
        algorithm_decision.append(temp[0])
        # 获得本轮正确的决定
        real_decision = generate_real_decisions()
        # 更新权重
        for i in range(n_expert):
            if expert_decision[i] != real_decision:
                expert_weight[i] = expert_weight[i] * (1 - epsilon)
    return algorithm_decision

# Chapter 1 Page-12
def hedge_algorithm(n_expert, t_round, epsilon):
    expert_weight = numpy.ones(n_expert)
    weight_sum_a = 0
    weight_sum_b = 0
    algorithm_decision = []
    loss_vector = loss_value(n_expert)
    for i_round in range(t_round):
        # print(expert_weight)
        # 获取n个专家的建议
        expert_decision = expert_make_decision(n_expert)
        # 算法处理专家权重
        for i in range(n_expert):
            if expert_decision[i] == 0:
                weight_sum_a = weight_sum_a + expert_weight[i]
            else:
                weight_sum_b = weight_sum_b + expert_weight[i]
        weight_sum = weight_sum_a + weight_sum_b
        # 产生算法本轮所做的决定
        temp = numpy.random.choice([0, 1], size=1, p=[weight_sum_a / weight_sum, weight_sum_b / weight_sum])
        algorithm_decision.append(temp[0])
        # 获得本轮正确的决定
        real_decision = generate_real_decisions()
        # 更新权重
        for i in range(n_expert):
            expert_weight[i] = expert_weight[i] * math.exp(-epsilon * loss_vector[i])
    return algorithm_decision
