import numpy as np

R = np.matrix([ [0,0,0,0,1,0],
                [0,0,0,1,0,1],
                [0,0,100,1,0,0],
                [0,1,1,0,1,0],
                [1,0,0,1,0,0],
                [0,1,0,0,0,0]])
Q = np.matrix(np.zeros([6,6]))
gamma = 0.8

agent_s_state = 3

def possible_actions(state):
    current_state_row = R[state,]
    possible_act = np.where(current_state_row > 0)[1]
    return possible_act

def ActionChoice(available_actions_range):
    next_action = np.random.choice(available_actions_range, 1)
    return next_action

def reward(current_state, action, gamma):
    Max_State = np.where(Q[action,] == np.max(Q[action,]))[1]
    if Max_State.shape[0] > 1:
        Max_State = int(np.random.choice(Max_State, 1))
    else:
        Max_State = int(Max_State)
    MaxValue = Q[action, Max_State]
    Q[current_state, action] = R[current_state, action] + gamma * MaxValue


for i in range(100000):
    current_state = np.random.randint(0, int(Q.shape[0]))
    reward(current_state, ActionChoice(possible_actions(current_state)), gamma)
    if (i % 10000) == 0:
        print(Q)

norm_Q = Q/np.max(Q) * 100
print(norm_Q)

