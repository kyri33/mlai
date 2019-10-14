import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set_style('darkgrid')
# %pylab inline
import random

gamma = 1 # discounting rate
rewardSize = -1
gridSize = 4
terminationStates = [[0, 0], [gridSize - 1, gridSize - 1]]
actions = [[-1, 0], [1, 0], [0, 1], [0, -1]] # left, right, up, down
numIterations = 1000

def actionRewardFunction(initialPosition, action):
    if initialPosition in terminationStates:
        return initialPosition, 0
    
    reward = rewardSize
    finalPosition = np.array(initialPosition) + np.array(action)
    if -1 in finalPosition or 4 in finalPosition:
        finalPosition = initialPosition
    
    return finalPosition, reward

valueMap = np.zeros((gridSize, gridSize))
states = [[i, j] for i in range(gridSize) for j in range(gridSize)]

deltas = []
for it in range(numIterations):
    copyValueMap = np.copy(valueMap)
    deltaState = []
    for state in states:
        weightedRewards = 0
        for action in actions:
            finalPosition, reward = actionRewardFunction(state, action)
            weightedRewards += (1 / len(actions)) * (reward + (gamma * valueMap[finalPosition[0], finalPosition[1]]))
        deltaState.append(np.abs(copyValueMap[state[0], state[1]] - weightedRewards))
        copyValueMap[state[0], state[1]] = weightedRewards
    deltas.append(deltaState)
    valueMap = copyValueMap
    if it in [0, 1, 2, 9, 99, numIterations - 1]:
        print("Iteration {}".format(it+1))
        print(valueMap)
        print("")