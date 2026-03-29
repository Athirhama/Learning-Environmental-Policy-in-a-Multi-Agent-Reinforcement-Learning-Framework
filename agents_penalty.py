import numpy as np
import random

class GovernmentAgent:
    def __init__(self, lr=0.1, discount=0.95, epsilon=1.0):
        # Actions : (Type, Valeur)
        # Type 0 = Taxe, Type 1 = Quota
        self.actions = [ (0, v) for v in [0, 1, 2, 3, 4, 5] ] + \
                       [ (1, v) for v in [10, 8, 6, 4, 2] ]
        
        self.lr = lr
        self.discount = discount # Gamma
        self.epsilon = epsilon
        self.eps_decay = 0.9999 
        self.q_table = {} 

    def discretize_state(self, pollution):
        """
        The state is simplified to pollution alone to stabilize learning.
        """
        if isinstance(pollution, np.ndarray):
            val = pollution[-1] 
        else:
            val = pollution
            
        return int(np.clip(val / 5, 0, 19)) # 20 pollution levels

    def get_action(self, state):
        s = self.discretize_state(state)
        
        if s not in self.q_table:
            self.q_table[s] = np.zeros(len(self.actions))
        
        if random.random() < self.epsilon:
            return random.randint(0, len(self.actions) - 1)
        
        # Select the action with the highest Q-value, breaking ties randomly
        q_values = self.q_table[s]
        max_q = np.max(q_values)
        candidates = np.where(q_values == max_q)[0]
        return np.random.choice(candidates)

    def learn(self, s, a_idx, r, ns, done): 
        s_dis = self.discretize_state(s)
        ns_dis = self.discretize_state(ns)
        
        if s_dis not in self.q_table: self.q_table[s_dis] = np.zeros(len(self.actions))
        if ns_dis not in self.q_table: self.q_table[ns_dis] = np.zeros(len(self.actions))
        
        
        if done:
            td_target = r  # no future reward if episode ended
        else:
            td_target = r + self.discount * np.max(self.q_table[ns_dis])
    
        self.q_table[s_dis][a_idx] += self.lr * (td_target - self.q_table[s_dis][a_idx])

    def learn_expected_sarsa(self, s, a_idx, r, ns, done): 
        s_dis = self.discretize_state(s)
        ns_dis = self.discretize_state(ns)
        
        if s_dis not in self.q_table:
            self.q_table[s_dis] = np.zeros(len(self.actions))
        if ns_dis not in self.q_table:
            self.q_table[ns_dis] = np.zeros(len(self.actions))
            
        # Expected SARSA : with epsilon-greedy policy 
        if done:
            expected_q = 0.0
        else:
            n_actions = len(self.actions)
            best_next_action = np.argmax(self.q_table[ns_dis])
            probs = np.ones(n_actions) * (self.epsilon / n_actions)
            probs[best_next_action] += (1.0 - self.epsilon)
            expected_q = np.dot(self.q_table[ns_dis], probs)

        td_target = r + self.discount * expected_q
        self.q_table[s_dis][a_idx] += self.lr * (td_target - self.q_table[s_dis][a_idx])
            
    def decay_epsilon(self):
        self.epsilon = max(0.01, self.epsilon * self.eps_decay)


class FirmAgent:
    def __init__(self, lr=0.1, epsilon=1.0):
        # Actions : production quantity in tons allowing producing floating point values
        self.actions = np.linspace(0, 15, 31)
        self.lr = lr
        self.epsilon = epsilon
        self.eps_decay = 0.9999
        self.q_table = {}

    def get_action(self, action_type, action_value):
        """the firm observes the government's action and decides its production quantity"""
        state_key = (int(action_type), round(float(action_value), 1))
        
        if state_key not in self.q_table:
            # optimistic initialization to encourage exploration
            self.q_table[state_key] = np.ones(len(self.actions)) * 20.0
        
        if random.random() < self.epsilon:
            return random.randint(0, len(self.actions) - 1)
        
        return np.argmax(self.q_table[state_key])

    def learn(self, action_type, action_value, a_idx, r):
        """Learning with discount factor of 0, teh firms only learn to maximize immediate profit"""
        state_key = (int(action_type), round(float(action_value), 1))
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.ones(len(self.actions)) * 20.0
            
        self.q_table[state_key][a_idx] += self.lr * (r - self.q_table[state_key][a_idx])

    def decay_epsilon(self):
        self.epsilon = max(0.01, self.epsilon * self.eps_decay)
