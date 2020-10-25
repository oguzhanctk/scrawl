#REINFORCEMENT LEARNING EXAMPLE

#%%
import matplotlib.pyplot as plt
from networkx.classes.function import number_of_nodes
import numpy as np
import networkx as nx

#%%
points_list = [(0,1), (1,5), (5,6), (5,4), (1,2), (2,3), (2,7)]
GOAL = 7


G = nx.Graph()
G.add_edges_from(points_list)
pos = nx.spring_layout(G)
nx.draw(G, with_labels=True, node_color="red", font_weight="bold")
plt.show()

# %%
MATRIX_SIZE = 8

#create R(state-action) array
R = np.ones(shape=(MATRIX_SIZE, MATRIX_SIZE))
R *= -1

for point in points_list:
    if point[1] == GOAL:
        R[point] = 100
    else:
        R[point] = 0

    if point[0] == GOAL:
        R[point[::-1]] = 100
    else:
        R[point[::-1]] = 0

R[GOAL, GOAL] = 100
# %%
Q = np.zeros(shape=(MATRIX_SIZE, MATRIX_SIZE))
gamma = 0.8

def avaliable_actions(state):
    s = np.where(R[state,] >= 0)[0]
    return s

def iter_next_act(avaliable_state_range):
    next_action = int(np.random.choice(avaliable_state_range, 1))
    return next_action

def update(current_state, action, gamma):
    max_index = np.where(Q[action,] == np.max(Q[action,]))[0]
    if max_index.shape[0] > 1:
        max_index = int(np.random.choice(max_index, 1))
    else:
        max_index = int(max_index)
    max_value = Q[action, max_index]

    update_value = R[current_state, action] + gamma * max_value
    Q[current_state, action] = update_value
    # print(f"update value : {update_value}")

    return np.sum(Q)

#%%
#training
scores = []
for _i in range(700):
    current_state = np.random.randint(0, MATRIX_SIZE)
    avaliable_act = avaliable_actions(current_state)
    action = iter_next_act(avaliable_act)
    score = update(current_state, action, gamma)
    scores.append(score)

print(np.max(Q))
plt.plot(list(range(700)), scores)
plt.show()

#%%
#testing
current_state = 6
path = [current_state]
while current_state != 7:
    next_step_index = np.where(Q[current_state,] == np.max(Q[current_state]))[0]
    if next_step_index.shape[0] > 1:
        next_step_index = int(np.random.choice(next_step_index, 1))
    else:
        next_step_index = int(next_step_index)
    path.append(next_step_index)
    current_state = next_step_index
print(path)
#%%
#OOP REINFORCEMENT
import matplotlib.pyplot as plt 
import numpy as np

class RL:

    gamma = 0.8
    
    def __init__(self, num_of_node, points, goal):
        self.num_of_node = num_of_node
        self.points = points
        self.goal = goal
        self.R = self.fill_R_matrix(self.points)
        self.Q = np.zeros(shape=(self.num_of_node, self.num_of_node))

    def fill_R_matrix(self, points):
        R = np.ones(shape=(self.num_of_node, self.num_of_node)) * -1
        for point in points:
            if point[1] == self.goal:
                R[point] = 100
            else:
                R[point] = 0
            if point[0] == self.goal:
                R[point[::-1]] = 100
            else:
                R[point[::-1]] = 0
        R[self.goal, self.goal] = 100
        return R

    def find_avaliable_acts(self, state):
        state_row = self.R[state, ]
        acts = np.where(state_row >= 0)[0]
        return acts
    
    def sample_next_act(self, acts_range):
        act = int(np.random.choice(acts_range, 1))
        return act
    
    def update_Q(self, current_state, action):
        max_values = np.where(self.Q[action,] == max(self.Q[action,]))[0]
        max_index = None
        
        if len(max_values) > 1:
            max_index = int(np.random.choice(max_values))
        else:
            max_index = int(max_values)
        self.Q[current_state, action] = self.R[current_state, action] + self.gamma * self.Q[action, max_index] 
        
    def sum_Q(self):
        return np.sum(self.Q)
    

# %%
#training
rl = RL(8, [(0,1), (1,5), (5,6), (5,4), (1,2), (2,3), (2,7)], 7)

sums = []
points = [0, 1, 2, 3, 4, 5, 6, 7]

for _i in range(500):
    state = int(np.random.choice(points, 1))
    act_range = rl.find_avaliable_acts(state)
    action = rl.sample_next_act(act_range)
    rl.update_Q(state, action)
    sums.append(rl.sum_Q())

#testing
current_state = 4
steps = [current_state]
while current_state != 7:
    max_values = np.where(rl.Q[current_state,] == np.max(rl.Q[current_state]))[0]
    if max_values.shape[0] > 1:
        max_index = int(np.random.choice(max_values, 1))
    else:
        max_index = int(max_values)
    current_state = max_index
    steps.append(current_state)
    
print(steps)

# %%


# %%
