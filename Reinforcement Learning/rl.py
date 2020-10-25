import gym
import numpy as np


def state2disc(s):
     disc_s = (s - env.observation_space.low) // chunk_size
     return tuple(disc_s.astype(np.int))

env = gym.make("MountainCar-v0")

episodes = 25000
per = 3000
discount = 0.95
learning_rate = 0.1 
epsilon = 0.5 #higher epsilon, higner courage for exploration
eps_decay_start = 1 
eps_decay_end = episodes // 2
decay_value = epsilon / (eps_decay_end - eps_decay_start)

# now we know that observation space is consisted by 2 dimensional space
discrete_states = [20] * len(env.observation_space.low)
chunk_size = (env.observation_space.high - env.observation_space.low) / discrete_states

Q = np.random.uniform(-2, 0, size=(discrete_states + [env.action_space.n]))

# print(state2disc(env.reset()))

for e in range(episodes):
      state = state2disc(env.reset())
      done = False
      if e % per == 0:
          print(e)
          render = True
      else:
          render = False
      
      while not done:
          # if np.random.random() > epsilon:
          action = np.argmax(Q[state])
          # else:
          #      action = np.random.randint(0, env.action_space.n)
          obs, rew, done, info = env.step(action)
          new_state = state2disc(obs)
          
          if render == True:
               env.render()
          
          if not done:
               #update Q matrix
               Q_next = (1 - learning_rate) * Q[state + (action, )] + learning_rate * (rew + discount * np.max(Q[new_state]))               
               Q[state + (action, )] = Q_next

          elif obs[0] >= env.goal_position:
               Q[state, (action, )] = 0
               print("reached goal position : {}".format(e))
               
          state = new_state
      if eps_decay_end >= episodes >= eps_decay_start:
           epsilon -= decay_value()
           
env.close()

