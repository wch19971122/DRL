import random
import gymnasium as gym
from matplotlib.pylab import random_sample
import numpy as np
from sympy import print_fcode
import torch
import torch.nn as nn
from lunar_lander_agent import Agent

env = gym.make("LunarLander-v3")
state, info = env.reset()
EPSILON_DECAY = 100000
EPSILON_START = 1.0
EPSILON_END = 0.5
TARGET_NET_UPDATE_FRE = 20

n_episode = 500
n_time_step = 1000

state_dim = len(state)
action_dim = env.action_space.n

agent = Agent(state_dim, action_dim)

REWARD_BUFFER = np.empty(n_episode)

for episode_i in range(n_episode):
    episode_reward = 0
    for step_i in range(n_time_step):
        epsilon = np.interp(
            episode_i * n_time_step + step_i,
            [0, EPSILON_DECAY],
            [EPSILON_START, EPSILON_END],
        )
        random_sample = random.random()
        if random_sample <= epsilon:
            action = env.action_space.sample()
        else:
            action = agent.online_net.get_action(state)

        next_state, reward, terminated, info, _ = env.step(action)
        agent.replay_buffer.add_experience(
            state, action, reward, terminated, next_state
        )

        state = next_state
        episode_reward += reward

        if terminated or step_i == n_time_step - 1:
            state, info = env.reset()
            REWARD_BUFFER[episode_i] = episode_reward

        batch_state, batch_action, batch_reward, batch_done, batch_next_state = (
            agent.replay_buffer.sample_batch()
        )

        # comput target Q values
        target_q_values = agent.target_net(batch_next_state)
        max_target_q_values = target_q_values.max(1, True)[0]
        targets = batch_reward + agent.GAMMA * (1 - batch_done) * max_target_q_values

        # comput online Q values
        q_values = agent.online_net(batch_state)
        a_q_values = torch.gather(q_values, 1, batch_action)

        # compute loss
        loss = nn.functional.smooth_l1_loss(targets, a_q_values)

        # gradient descent
        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()

    if episode_i != 0 and episode_i % 50 == 0:  # 每100个episode查看一次
        render_env = gym.make("LunarLander-v3", render_mode="human")
        test_s, info = render_env.reset()

        done = False
        truncated = False
        while not done and not truncated:  # 查看200步
            a = agent.online_net.get_action(test_s)
            test_s, r, done, truncated, info = render_env.step(a)

            if done or truncated:
                render_env.close()
                break

    if episode_i % TARGET_NET_UPDATE_FRE == 0:
        agent.target_net.load_state_dict(agent.online_net.state_dict())

        # show training progress
        print(
            "Episode: {}",
            episode_i,
            "Avg Reward: {}",
            np.mean(REWARD_BUFFER[: episode_i + 1]),
        )
env.close()
