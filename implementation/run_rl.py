import gym
import numpy as np
import pickle
# load the class defining the custom Open AI Gym problem
from uofgsocsai import LochLomondEnv
from gym import spaces
import sys

try:
    print("This is the name input number as problem_id: ", sys.argv[1])
    temp_id = int(sys.argv[1])
except IndexError as identifier:
    print("There is no input number so the problem id is set to default 0.")
    temp_id = 0

# Setup the parameters for the specific problem (you can change all of these if you want to)
# problem_id \in [0:7] generates 8 diffrent problems on which you can train/fine-tune your agent
problem_id = temp_id
# should be less than or equal to 0.0 (you can fine tune this  depending on you RL agent choice)
reward_hole = -0.02
# should be False for A-star (deterministic search) and True for the RL agent
is_stochastic = True

# Load Environment and Q-table structure
env = LochLomondEnv(problem_id=problem_id,is_stochastic=is_stochastic, reward_hole=reward_hole)

#  learn episode times!
max_episodes = 10000

rev_list = []  # rewards per episode calculate
step_list = []
""" This is the Reinforcement learning agent using tabular Q-learning """
def q_learning_agent(env,problem_id,max_episodes):  
    output_file = f'out_rl_{problem_id}.pkl'
    # initiate Q table
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    # Parameters of Q-leanring
    # learning rate
    eta = 0.1
    # discount rate
    gma = 0.8
    # actions can be executed per episode
    max_iter_per_episode = 2000
   
    # Q-learning Algorithm
    for episode in range(max_episodes):
        # Reset environment
        state = env.reset()
        rewardAll = 0
        j = 0
        if(episode % 100 == 0):
            print("learning times: " + str(episode))
        # The Q-Table learning algorithm
        while j < max_iter_per_episode:
            j+=1 
            # Choose an action by greedily (with noise) picking from Q table
            action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n)*(1./(episode+1)))
            # Get new state & reward from environment
            state2, reward, done, info = env.step(action)
            # learn and update Q-Table with new knowledge
            Q[state, action] = Q[state, action] + eta * (reward + gma*np.max(Q[state2, :]) - Q[state, action])

            rewardAll += reward
            state = state2
            if done:
                break
        step_list.append(j)      
        rev_list.append(rewardAll)
        # env.render()
    #print("Reward rate on all episodes " + rev_list)
    print("Final Values Q-Table: ")
    print(Q)
    with open(output_file, 'wb') as f:
        pickle.dump(Q, f)
    print("The Q table has been saved into file: " + output_file)
    return Q

# a simple run the agent to learn   
if __name__ == '__main__':
    q_learning_agent(env, problem_id,max_episodes)
    