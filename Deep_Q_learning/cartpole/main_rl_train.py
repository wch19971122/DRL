import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from agent import Agent

env = gym.make("CartPole-v1")
s, info = env.reset()

EPSILON_DECAY = 100000
EPSILON_START = 1.0
EPSILON_END = 0.02
TARGET_UPDATE_FREQUENCY = 10

n_episode = 500
n_time_step = 1000

n_state = len(s)
n_action = env.action_space.n
agent = Agent(n_state, n_action)

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
            a = env.action_space.sample()
        else:
            a = agent.online_net.act(s)  # TODO

        s_, r, done, info, _ = env.step(a)
        agent.memo.add_memo(s, a, r, done, s_)  # TODO
        s = s_
        episode_reward += r

        if done:
            s, info = env.reset()
            REWARD_BUFFER[episode_i] = episode_reward
            break

        batch_s, batch_a, batch_r, batch_done, batch_s_ = agent.memo.sample()  # TODO

        # compute target
        target_q_values = agent.target_net(batch_s_)  # TODO
        max_target_q_values = target_q_values.max(1, True)[0]
        targets = batch_r + agent.GAMMA * (1 - batch_done) * max_target_q_values  # TODO

        # compute Q values
        q_values = agent.online_net(batch_s)  # TODO
        a_q_values = torch.gather(input=q_values, dim=1, index=batch_a)

        # compute loss
        loss = nn.functional.smooth_l1_loss(targets, a_q_values)

        # gradient descent
        agent.optimizer.zero_grad()  # TODO
        loss.backward()
        agent.optimizer.step()  # TODO

    if episode_i % 50 == 0:  # 每500个episode查看一次
        render_env = gym.make("CartPole-v1", render_mode="human")
        test_s, info = render_env.reset()

        for step in range(200):  # 查看200步
            a = agent.online_net.act(test_s)
            test_s, r, done, truncated, info = render_env.step(a)

            if done or truncated:
                break

    # update target Q net parameter
    if episode_i % TARGET_UPDATE_FREQUENCY == 0:
        agent.target_net.load_state_dict(agent.online_net.state_dict())  # TODO

        # show training progress
        print(
            "Episode: {}",
            episode_i,
            "Avg Reward: {}",
            np.mean(REWARD_BUFFER[: episode_i + 1]),
        )

env.close()
