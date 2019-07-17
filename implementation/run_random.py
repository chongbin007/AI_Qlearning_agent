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
reward_hole = 0.0
# should be False for A-star (deterministic search) and True for the RL agent
is_stochastic = True

# Load Environment and Q-table structure
env = LochLomondEnv(problem_id=problem_id,
                    is_stochastic=is_stochastic, reward_hole=reward_hole)

#  learn episode times!
max_episodes = 10000
# you decide how many iterations/actions can be executed per episode
max_iter_per_episode = 2000


# random agent
def random_agent(env, problem_id, max_episodes):
    output_file = f'out_random_{problem_id}.pkl'
    n_states = env.observation_space.n
    random_agent_dict = {}  # define a dict to save random actions
    # set random action to a dict
    for state in range(n_states):
        random_agent_dict[state] = env.action_space.sample()

    print("The dict has been saved into file: " + output_file)
    with open(output_file, 'wb') as f:
        pickle.dump(random_agent_dict, f)
    print(random_agent_dict)
    # return a dict for use
    return random_agent_dict


# a simple run random agent
if __name__ == '__main__':
    agent_dict = random_agent(env,problem_id,max_episodes)
    reward_random_accumulate = 0
    reward_random_total = 0
    for episode in range(max_episodes):
        state = env.reset()
        step = 0
        reward_random = 0
        for step in range(max_iter_per_episode):
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)

            reward_random_accumulate += reward

            if(step == max_iter_per_episode-1):
                print("step over")
            if(done and reward == reward_hole):
                #print("hole :-( ")
                break
            if (done and reward == +1.0):
                reward_random_total = reward + reward_random_total
                print("We have reached the goal :-) [stop trying to move; we can't]. That's ok we have achived the goal]")
                #print("Number of steps", step)
                break
        # reward_random_list.append(reward_random_accumulate)

    print("The total reward is  = " + str(reward_random_total))
