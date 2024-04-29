import numpy as np
import gym
import random
# Example usage
goal_positions = [(0,0),(49,49)]  # Define the goal position
    # agent_positions = [(2, 2), (3, 3)]  # Example agent positions

    # rewards = reward_function(agent_positions, goal_position)
    # print("Rewards:", rewards)
class MultiAgentGridWorldEnv(gym.Env):
    def __init__(self, grid_size=(50, 50), num_agents=2):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.action_space = [gym.spaces.Discrete(4) for _ in range(self.num_agents)]
        self.observation_space = [gym.spaces.Box(low=0, high=1, shape=(grid_size[0], grid_size[1], 1)) for _ in range(self.num_agents)]
        self.state = [np.zeros((grid_size[0], grid_size[1], 1)) for _ in range(self.num_agents)]
        # self.rand_x = random.randint(0, (self.grid_size[0] - 1))
        # self.rand_y = random.randint(0, (self.grid_size[0] - 1))
        self.rand_x=self.rand_y=5
        self.agent_positions = [(self.rand_x, self.rand_y) for _ in range(self.num_agents)]
        # print("action space\n",self.action_space,"obs space\n",self.observation_space)
        # print("state \n",self.state)

    def reward_function(self,agent_positions, goal_positions):
        rewards = []

        for i in range(self.num_agents):
            if agent_positions[i] == goal_positions[i]:
                rewards.append(0)  # no cost for reaching the goal
            elif agent_positions[i] == agent_positions[(i+1)%self.num_agents]:
                rewards.append(0.25)  # High cost for collision    
            else:
                rewards.append(0.5)  # Small cost for each step
        # print("rewards", rewards)            
        return rewards


    def step(self, actions):
        rewards = self.reward_function(self.agent_positions, goal_positions)

        for i, action in enumerate(actions):
            x, y = self.agent_positions[i]
            # print("x,y",x,y)
            # print("action",action)
            # action=action.any()
            if action == 0 and x > 0:
                x -= 1
            elif action == 1 and x < self.grid_size[0] - 1:
                x += 1
            elif action == 2 and y > 0:
                y -= 1
            elif action == 3 and y < self.grid_size[1] - 1:
                y += 1
            self.agent_positions[i] = (x, y)
            self.state[i] = np.zeros((self.grid_size[0], self.grid_size[1], 1))
            self.state[i][x][y] = 1
        
        return self.state, rewards, False

    def reset(self):
        self.state = [np.zeros((self.grid_size[0], self.grid_size[1], 1)) for _ in range(self.num_agents)]
        self.agent_positions = [(self.rand_x, self.rand_y) for _ in range(self.num_agents)]
        for i, pos in enumerate(self.agent_positions):
            x, y = pos
            self.state[i][x][y] = 1
        # print("reset: state",self.state)    
        return self.state

    # def render(self, mode='human'):
    #     for i in range(self.num_agents):
            # print(f'Agent {i+1} Position: {self.agent_positions[i]}')
        # print('\n')

# Example usage
# env = MultiAgentGridWorldEnv(grid_size=(4, 4), num_agents=2)
# obs = env.reset()
# # env.render()

# for _ in range(3):
#     actions = [np.random.choice(env.action_space[i].n) for i in range(env.num_agents)]
#     obs, rewards, done, info = env.step(actions)
    # print(obs, rewards, done, info)
    # env.render()
