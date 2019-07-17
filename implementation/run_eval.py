import gym
import numpy as np
import pickle
# load the class defining the custom Open AI Gym problem
from uofgsocsai import LochLomondEnv
from gym import spaces

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from matplotlib import lines

# Setup the parameters for the specific problem (you can change all of these if you want to)
# problem_id \in [0:7] generates 8 diffrent problems on which you can train/fine-tune your agent
problem_id = 6
# should be less than or equal to 0.0 (you can fine tune this  depending on you RL agent choice)
reward_hole = -0.02
# should be False for A-star (deterministic search) and True for the RL agent
is_stochastic = True
# Load Environment and Q-table structure
env = LochLomondEnv(problem_id=problem_id,
                    is_stochastic= is_stochastic, reward_hole= reward_hole)



# you can decide you rerun the problem many times thus generating many episodes... you can learn from them all!
max_episodes = 10000
# you decide how many iterations/actions can be executed per episode
max_iter_per_episode = 2000

""" This is the random agent """
def random_agent():
    n_states = env.observation_space.n
    random_agent_dict  = {} # define a dict to save random actions
    # set random action to a dict
    for state in range(n_states):
        random_agent_dict[state] = env.action_space.sample()

    print("Print rand item" )
    print(random_agent_dict)
    # return a dict for use 
    return random_agent_dict


reward_random_list = []  # rewards per episode calculate
def evaluate_random():    
    reward_random_accumulate = 0
    reward_random_total = 0
    for episode in range(max_episodes):
        state = env.reset()
        step=0
        reward_random = 0
        for step in range(max_iter_per_episode):

            #action = random_agent_dict.get(state)
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)

            reward_random_accumulate += reward

            if(step == max_iter_per_episode-1):
                print("step over")
            if(done and reward == reward_hole):
                print("hole :-( ")
                break
            if (done and reward == +1.0):
                reward_random_total = reward + reward_random_total
                print("We have reached the goal :-) [stop trying to move; we can't]. That's ok we have achived the goal]")
                #print("Number of steps", step)
                break
        reward_random_list.append(reward_random_accumulate)

    print("The total reward is  = " + str(reward_random_total))
    return reward_random_total/max_episodes

# evaluate the random agent
temp1 = evaluate_random()


fig = plt.figure()
plt.plot(reward_random_list)
fig.suptitle('Reward accumulate - Random agent', fontsize=16)
plt.xlabel('episodes', fontsize=14)
plt.ylabel('reward accumulate', fontsize=14)


rev_list = []  # rewards per episode calculate
step_list = [] # step number per episode
""" This is the Reinforcement learning agent using tabular Q-learning """
def q_learning_agent():  
    #output_file = f'out_rl_{problem_id}.pkl'
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
    #print("Reward rate on all episodes " + str(sum(rev_list)/1000))
    print("Final Values Q-Table: ")
    print(Q)
    return Q


Q = q_learning_agent()

fig = plt.figure(figsize=(30,15))
plt.plot(step_list)
fig.suptitle('Step number display - Q learning', fontsize=26)
plt.xlabel('episodes', fontsize=20)
plt.ylabel('step number per episode', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)




j = 0
score = 0
temp_list = []
for i in rev_list:    
    j=j+1
    score = i+score
    if (j%100 == 0):
        temp_list.append(score/100)
        score = 0
        j = 0
    
fig = plt.figure(figsize=(30,15))
plt.plot(temp_list)
fig.suptitle('convergence - Q learning' ,fontsize=26)
plt.xlabel('Max episodes divide 100',fontsize=20)
plt.ylabel('reward(average over 100 episodes)',fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)



rev_q_list = [] 
'''evaluate the learning agent'''
def evaluate_q():
    reward_q_accumulate = 0
    reward_q_total = 0

    for episode in range(max_episodes):
        state = env.reset()
        step=0
        
        for step in range(max_iter_per_episode):

            action = np.argmax(Q[state, :] + np.random.randn(1,env.action_space.n)*(1./(episode+1)))

            state2, reward, done, info = env.step(action)

            reward_q_accumulate += reward

            if(step == max_iter_per_episode-1):
                print("step over")
            if(done and reward == reward_hole):
                #print("hole :-( ")
                break
            if (done and reward == +1.0):
                reward_q_total = reward + reward_q_total
                #print("We have reached the goal :-) [stop trying to move; we can't]. That's ok we have achived the goal]")
                print("Goal!!!!!Number of steps", step)
                break
            state = state2

        rev_q_list.append(reward_q_accumulate)

    print("The total reward is  = " + str(reward_q_total))


evaluate_q()

fig = plt.figure()
plt.plot(rev_q_list)
fig.suptitle('Reward accumulate - Q learning', fontsize=16)
plt.xlabel('episodes', fontsize=14)
plt.ylabel('reward accumulate', fontsize=14)



plt.show() 

