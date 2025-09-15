import os
import time
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from q_table_agent import BlackJackAgent

env = gym.make("Blackjack-v1", sab=True)
ACTION_SAPCE = env.action_space.n
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.95
EPSILON = 0.1

current_path = os.path.dirname(os.path.realpath(__file__))
timestamp = time.strftime("%Y%m%d%H%M%S")

agent = BlackJackAgent(env=env,learning_rate = LEARNING_RATE,discount_factor = DISCOUNT_FACTOR, epsilon = EPSILON)

NUM_EPISODE = 500000
REWARD_BUFFER = np.empty(NUM_EPISODE)
AVG_REWARD_BUFFER = np.empty(NUM_EPISODE)
PRINT_INTERVAL = 10000

for episode_i in range(NUM_EPISODE):
    state, _ = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action = agent.get_action(state)
        observation, reward, done, _, _ = env.step(action)
        agent.update_q_table(state, action, reward, observation, done)
        state = observation
        episode_reward += reward

    REWARD_BUFFER[episode_i] = episode_reward
    AVG_REWARD_BUFFER[episode_i] = np.mean(REWARD_BUFFER[0:episode_i+1])   

    if(episode_i + 1) % PRINT_INTERVAL == 0:
        print(f"Episode {episode_i + 1}, Average Reward: {AVG_REWARD_BUFFER[episode_i]:.3f}")


plt.figure(figsize = (10, 5))
plt.plot(np.arange(len(AVG_REWARD_BUFFER)), AVG_REWARD_BUFFER, color = 'purple', linewidth = 1, label = 'Avg. Reward')
plt.legend()
plt.title('Training Reward')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.grid(True)
plt.savefig(current_path + f'/training_reward_{timestamp}.png', dpi = 300)
plt.show()

agent.plot_policy()

