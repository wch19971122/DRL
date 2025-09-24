import random
import numpy as np
import torch
import torch.nn as nn


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.BUFFER_SIZE = 10000
        self.BATCH_SIZE = 256

        # pre allocate memory
        self.all_states = np.empty((self.BUFFER_SIZE, self.state_dim), dtype=np.float32)
        self.all_actions = np.random.randint(
            low=0, high=action_dim, size=self.BUFFER_SIZE, dtype=np.uint8
        )
        self.all_rewards = np.empty(self.BUFFER_SIZE, dtype=np.float32)
        self.all_dones = np.random.randint(
            low=0, high=2, size=self.BUFFER_SIZE, dtype=np.uint8
        )
        self.all_next_states = np.empty(
            (self.BUFFER_SIZE, self.state_dim), dtype=np.float32
        )
        self.total_size = 0
        self.current_index = 0

    def add_experience(self, state, action, reward, done, next_state):
        self.all_states[self.current_index] = state
        self.all_actions[self.current_index] = action
        self.all_rewards[self.current_index] = reward
        self.all_dones[self.current_index] = done
        self.all_next_states[self.current_index] = next_state
        self.total_size = max(self.total_size, self.current_index + 1)
        self.current_index = (self.current_index + 1) % self.BUFFER_SIZE

    def sample_batch(self):
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_dones = []
        batch_next_states = []

        if self.total_size >= self.BATCH_SIZE:
            indices = random.sample(range(0, self.total_size), self.BATCH_SIZE)
        else:
            indices = random.sample(range(self.total_size), self.total_size)

        for idx in indices:
            batch_states.append(self.all_states[idx])
            batch_actions.append(self.all_actions[idx])
            batch_rewards.append(self.all_rewards[idx])
            batch_dones.append(self.all_dones[idx])
            batch_next_states.append(self.all_next_states[idx])

        batch_states_tensor = torch.as_tensor(
            np.asarray(batch_states), dtype=torch.float32
        ).to(self.device)
        batch_actions_tensor = (
            torch.as_tensor(np.asarray(batch_actions), dtype=torch.int64)
            .unsqueeze(-1)
            .to(self.device)
        )
        batch_rewards_tensor = (
            torch.as_tensor(np.asarray(batch_rewards), dtype=torch.float32)
            .unsqueeze(-1)
            .to(self.device)
        )
        batch_dones_tensor = (
            torch.as_tensor(np.asarray(batch_dones), dtype=torch.float32)
            .unsqueeze(-1)
            .to(self.device)
        )
        batch_next_states_tensor = torch.as_tensor(
            np.asarray(batch_next_states), dtype=torch.float32
        ).to(self.device)

        return (
            batch_states_tensor,
            batch_actions_tensor,
            batch_rewards_tensor,
            batch_dones_tensor,
            batch_next_states_tensor,
        )


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, device):
        super(DQN, self).__init__()
        self.device = device
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )
        # 将网络移到指定设备
        self.to(device)

    def forward(self, state):
        return self.net(state)

    def get_action(self, state):
        # 如果输入是numpy数组，转换为tensor并移到正确设备
        if isinstance(state, np.ndarray):
            state_tensor = (
                torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            )
        else:
            state_tensor = state  # 假设已经是tensor

        with torch.no_grad():
            q_values = self.forward(state_tensor)
            max_q_index = torch.argmax(q_values, dim=1)
            action = max_q_index.detach().item()
        return action


class Agent:
    def __init__(self, state_dim, action_dim, device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.GAMMA = 0.99
        self.learning_rate = 1e-3

        self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, self.device)
        self.online_net = DQN(self.state_dim, self.action_dim, self.device)
        self.target_net = DQN(self.state_dim, self.action_dim, self.device)

        # 初始化目标网络与在线网络相同的权重
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()  # 目标网络设置为评估模式

        self.optimizer = torch.optim.Adam(
            self.online_net.parameters(), lr=self.learning_rate
        )

    def to(self, device):
        """将agent的所有组件移到指定设备"""
        self.device = device
        self.online_net = self.online_net.to(device)
        self.target_net = self.target_net.to(device)
        return self

    def update_target_net(self):
        """更新目标网络"""
        self.target_net.load_state_dict(self.online_net.state_dict())
