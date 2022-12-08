import math
import numpy
from collections import namedtuple


from chapter_one import learning_from_expert_advice

if __name__ == '__main__':
    # algorithm_decision = learning_from_expert_advice.the_weighted_majority(5, 50, 0.3)
    # print(algorithm_decision)
    # algorithm_decision = learning_from_expert_advice.randomized_weighted_majority(5, 50, 0.3)
    # print(algorithm_decision)
    # algorithm_decision = learning_from_expert_advice.hedge_algorithm(5, 50, 0.3)
    # print(algorithm_decision)
    # 定义一个namedtuple类型User，并包含name，sex和age属性。
    # User = namedtuple('User', ['name', 'sex', 'age'])
    # # 创建一个User对象
    # user = User(name='kongxx', sex='male', age=21)
    # print(user)
    # print(user.name)
    ProposalID = namedtuple('ProposalID', ['number', 'uid'])
    pro = ProposalID._make(('111', '222'))
    print(pro)










