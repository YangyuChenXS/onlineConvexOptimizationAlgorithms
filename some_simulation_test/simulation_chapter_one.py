import math
import numpy
from chapter_one import learning_from_expert_advice

if __name__ == '__main__':
    algorithm_decision = learning_from_expert_advice.the_weighted_majority(5, 50, 0.3)
    print(algorithm_decision)
    algorithm_decision = learning_from_expert_advice.randomized_weighted_majority(5, 50, 0.3)
    print(algorithm_decision)




