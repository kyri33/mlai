import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

gamma = 0.6
rewardSize = -1
gridSize = 4
terminationStates = [[0, 0], [gridSize - 1, gridSize - 1]]
action = [[-1, 0], [1, 0], [0, -1], [0, 1]]
numIterations = 10000

V = np.zeros((gridSize, gridSize))
returns = {(i, j):list() for i in range(gridSize) for j in range(gridSize)}
print(returns)