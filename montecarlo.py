import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

gamma = 0.6
rewardSize = -1
gridSize = 4
terminationStates = [[0, 0], [gridSize - 1, gridSize - 1]]
actions = [[-1, 0], [1, 0], [0, -1], [0, 1]]
numIterations = 10000

V = np.zeros((gridSize, gridSize))
returns = {(i, j):list() for i in range(gridSize) for j in range(gridSize)}

deltas = {(i, j):list() for i in range(gridSize) for j in range(gridSize)}

states = [[i, j] for i in range(gridSize) for j in range(gridSize)]

def generateEpisode():
    initState = random.choice(states[1:-1])
    episode = []
    while True:
        if list(initState) in terminationStates:
            return episode
        action = random.choice(actions)
        finalState = np.array(initState) + np.array(action)
        if -1 in list(finalState) or gridSize in list(finalState):
            finalState = initState
        episode.append([list(initState), action, rewardSize, list(finalState)])
        initState = finalState

for it in tqdm(range(numIterations)):
    episode = generateEpisode()
    print(episode[::-1][:])
    G = 0
    for i, step in enumerate(episode[::-1]):   # Enumerate adds index for i and the arr is going backwards ! with ::-1
        #print(i, step)
        G = gamma*G + step[2] # reward
        print([x[0] for x in episode[::-1][len(episode) - i:]])
        if step[0] not in [x[0] for x in episode[::-1][len(episode) - i:]]:
            idx = (step[0][0], step[0][1])
            returns[idx].append(G)
            newValue = np.average(returns[idx])
            deltas[idx].append(np.average(V[idx[0], idx[1]] - newValue))
            V[idx[0], idx[1]] = newValue
    exit()

generateEpisode()