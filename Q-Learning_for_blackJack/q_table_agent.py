from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

def default_array():
    return np.zeros(2)

class BlackJackAgent:
    def __init__(self, env, learning_rate, discount_factor, epsilon):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
    
        self.q_table = defaultdict(default_array)
        self.visit_counts = defaultdict(default_array)


    def get_state_key(self, s):
        return tuple(s) if isinstance(s, (list, np.ndarray)) else s
    
    def get_action(self, s):
        s_key = self.get_state_key(s)

        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            q_table = self.q_table[s_key]
            return np.argmax(q_table)
        
    def update_q_table(self, s, a, r, next_s, done):
        s_key = self.get_state_key(s)
        next_s_key = self.get_state_key(next_s)

        self.visit_counts[s_key][a] +=1

        current_q = self.q_table[s_key][a]
        next_max_q = np.max(self.q_table[next_s_key]) if not done else 0

        td_error = r + self.discount_factor * next_max_q - current_q

        self.q_table[s_key][a] += self.learning_rate * td_error

    def plot_policy(self):
        player_sum = np.arange(21,3,-1)
        dealer_card = np.arange(1,11)


        policy_matrix_no_ace = np.zeros((len(player_sum), len(dealer_card)))
        policy_matrix_ace = np.zeros((len(player_sum), len(dealer_card)))

        for i, player in enumerate(player_sum):
            for j, dealer in enumerate(dealer_card):
                state_no_ace = (player, dealer, 0)
                state_ace = (player, dealer, 1)

                policy_matrix_no_ace[i,j]  = np.argmax(self.q_table[state_no_ace])
                policy_matrix_ace[i,j] = np.argmax(self.q_table[state_ace])
        
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6))

        #No ace 
        im1 = ax1.imshow(policy_matrix_no_ace, cmap='RdYlBu')
        ax1.set_title('Policy without Usable Ace')
        ax1.set_xlabel('Dealer Card')
        ax1.set_ylabel('Player Card')
        ax1.set_xticks(np.arange(len(dealer_card)))
        ax1.set_yticks(np.arange(len(player_sum)))
        ax1.set_xticklabels(dealer_card)
        ax1.set_yticklabels(player_sum)

        #ace
        im2 = ax2.imshow(policy_matrix_ace, cmap='RdYlBu')
        ax2.set_title('Policy Usable Ace')
        ax2.set_xlabel('Dealer Card')
        ax2.set_ylabel('Player Card')
        ax2.set_xticks(np.arange(len(dealer_card)))
        ax2.set_yticks(np.arange(len(player_sum)))
        ax2.set_xticklabels(dealer_card)
        ax2.set_yticklabels(player_sum)

        plt.colorbar(im1, ax=ax1, label = 'Action(0: Stick, 1: Hit)', fraction=0.046, pad=0.04)
        plt.colorbar(im2, ax=ax2, label = 'Action(0: Stick, 1: Hit)', fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(f'policy_heatmap.png', dpi = 300)
        plt.show()