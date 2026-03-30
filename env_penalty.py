import numpy as np

class EcoSystem:
    def __init__(self, n_firms=5, rho=0.9, alpha=0.5, price=15.0):
        self.n_firms = n_firms
        self.rho = rho      
        self.alpha = alpha  
        self.p = price      
        self.a_i = np.linspace(1, 5, n_firms) 
        self.b = 0.5
        self.lambda_env = 1  
        self.pollution_threshold = 150 #pollution level at which catastrophe occurs, to adjust based on the scale of pollution in the model 
        self.death_penalty = -10000 #penalty for social welfare in case of catastrophe, to adjust based on the scale of profits and welfare in the model
        self.reset()

    def reset(self):
        self.pollution = 0.0
        self.total_q = 0.0
        self.prev_policy_val = 0.0
        return self._get_state()

    def _get_state(self):
        return np.array([self.prev_policy_val, self.total_q, self.pollution])

    def step_multi_firms(self, action_type, action_value, q_list):
        individual_profits = []
        current_q_total = 0
        total_tax = 0 
        
        for i in range(self.n_firms):
            q_chosen = q_list[i]
            
            if action_type == 0: # TAXE
                tax_cost = action_value * self.alpha * q_chosen
                q_eff = q_chosen
            else: # QUOTA
                tax_cost = 0
                q_eff = min(q_chosen, action_value)

            total_tax += tax_cost 
            
            cost = self.a_i[i] * q_eff + self.b * (q_eff**2)
            profit = (self.p * q_eff) - cost - tax_cost
            
            individual_profits.append(profit)
            current_q_total += q_eff

        self.pollution = self.rho * self.pollution + self.alpha * current_q_total
        self.total_q = current_q_total

        # Check for catastrophe
        done = False

        if self.pollution >= self.pollution_threshold:
            done = True
    
            # Termination penalty: Triggers an immediate episode end with a heavily negative reward (death penalty)
            social_welfare = self.death_penalty

            return self._get_state(), social_welfare, individual_profits, done
    

        env_damage = self.lambda_env * self.pollution 
        social_welfare = sum(individual_profits) + total_tax - env_damage
        return self._get_state(), social_welfare, individual_profits, done 
