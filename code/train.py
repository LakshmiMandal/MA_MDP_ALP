import numpy as np
from scipy.optimize import linprog
import grid_world_env

# class MultiagentEnvironment:
#     def __init__(self):
#         # Define your environment here
        
#     def step(self, actions):
#         # Implement the step function for the environment
#         return next_state, rewards, done
    
#     def reset(self):
#         # Implement the reset function for the environment
#         return initial_state

class MultiagentALP:
    def __init__(self, num_agents, num_actions, state_dim, alpha=0.1, gamma=0.99):
        self.num_agents = num_agents
        self.num_actions = num_actions
        self.state_dim = state_dim
        self.size= int(np.abs(np.sqrt(state_dim)))
        self.alpha = alpha
        self.gamma = gamma
        self.policy = np.random.randint(0, num_actions, size=(num_agents, state_dim))  # Initialize random policy
        # print("policy",self.policy)

    def approximate_linear_programming(self, state, rewards):
        c = np.zeros(self.num_agents * self.num_actions)
        A_ub = np.zeros((self.num_actions * self.state_dim, self.num_agents * self.num_actions))
        b_ub = np.zeros(self.num_actions * self.state_dim)
        
        for a in range(self.num_actions):
            for i in range(self.num_agents):
                c[i*self.num_actions+a] = -rewards[i]  # Maximize rewards
                
        for s in range(self.state_dim):
            for a in range(self.num_actions):
                A_ub[s*self.num_actions+a, :] = self.get_feature_vector_sa(s)
                b_ub[s*self.num_actions+a] = np.dot(self.get_feature_vector(s), self.policy[:, s]) - self.gamma * np.dot(self.get_feature_vector(s), self.get_transition_matrix(a, s))

        res = linprog(c, A_ub=A_ub, b_ub=b_ub)
        q_values = res.x.reshape((self.num_agents, self.num_actions))
        
        return q_values

    def get_feature_vector_sa(self, st_ind):
        # Example: One-hot encoding for each agent's state
        feature_vector = np.zeros((self.num_agents, self.num_actions))
       
        for i in range(self.num_agents):
            for a in range( self.num_actions):
             feature_vector[i, a] = st_ind
        
        return feature_vector.flatten()
    def get_feature_vector(self, st_ind):
        # Example: One-hot encoding for each agent's state
        feature_vector = np.zeros((self.num_agents, self.state_dim))
        for i in range(self.num_agents):
            feature_vector[i, st_ind] = 1
        
        return feature_vector

    def get_transition_matrix(self, action, state_ind):
        # Example: deterministic transition matrix
        transition_matrix = np.zeros((self.num_agents, self.state_dim))
        for i in range(self.num_agents):
            next_state = (state_ind[i]+ action) % self.state_dim  # Example: simple wrap-around environment
            transition_matrix[i, next_state] = 1
        
        return transition_matrix

    def update_policy(self, state_ind, rewards):
        q_values = self.approximate_linear_programming(state_ind, rewards)
        print("q_values",q_values)
        self.policy = np.argmax(q_values, axis=1)

    def train(self, env, num_episodes):
        for _ in range(num_episodes):
            state = env.reset()
            # print(state)
            done = False
            while not done:
                actions = [self.policy[i, self.size *(np.where(state[i]==1)[0][0])+np.where(state[i]==1)[1][0]] for i in range(self.num_agents)]
                print("actions in train",actions)
                next_state, rewards, done = env.step(actions)
                next_st_ind=np.zeros(self.num_agents)
                for i in range(self.num_agents):
                    next_st_ind[i]= self.size *(np.where(next_state[i]==1)[0][0])+np.where(next_state[i]==1)[1][0]
                    # print("next_st_ind:::::",next_state[i],"\n",next_st_ind)
                nx_st_ind=int(np.mean(next_st_ind))    
                self.update_policy(nx_st_ind, rewards)
                state = next_state

# Example usage
# num_agents = 10
# num_actions = 5
# state_dim = 2500
# env = MultiagentEnvironment()
env = grid_world_env.MultiAgentGridWorldEnv()
num_agents=env.num_agents
print("num_agents",num_agents)
# env.render()
state_dim = env.grid_size[0]* env.grid_size[1]#24 #env.observation_space.shape[0]
num_actions = env.action_space[0].n
max_ep_len=30

total_runs = 1
avg_reward=[]
alp = MultiagentALP(num_agents, num_actions, state_dim)
alp.train(env, num_episodes=1000)
