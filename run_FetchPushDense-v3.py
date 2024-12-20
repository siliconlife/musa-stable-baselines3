#!/usr/bin/env python

import gymnasium as gym
import gymnasium_robotics
import torch
import torch_musa
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

gym.register_envs(gymnasium_robotics)

env = gym.make('FetchPushDense-v3', max_episode_steps=100) # Setting max_episode_steps here

# Check if MUSA is available and set device accordingly
device = torch.device("musa" if torch.musa.is_available() else "cpu")
print(f"Using device: {device}")

# Define the Actor network
class Actor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # Output action in range [-1, 1]
        return x

# Define the Critic network
class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def __len__(self):
        return len(self.buffer)

# DDPG Agent
class DDPGAgent:
    def __init__(self, state_dim, action_dim, action_range, capacity=100000, batch_size=128, lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=0.005):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.target_actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim + action_dim).to(device)
        self.target_critic = Critic(state_dim + action_dim).to(device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.replay_buffer = ReplayBuffer(capacity)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.action_range = action_range

        # Add noise for exploration
        self.noise_std = 0.1

    def get_action(self, state, explore=True):
        state = torch.FloatTensor(state).to(device)
        action = self.actor(state).detach().cpu().numpy()

        if explore:
            noise = np.random.normal(0, self.noise_std, size=action.shape)
            action = np.clip(action + noise, -1, 1)

        return action * self.action_range

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).unsqueeze(1).to(device)

        # Critic update
        next_action = self.target_actor(next_state)
        target_q = self.target_critic(next_state, next_action)
        y = reward + (1 - done) * self.gamma * target_q
        q = self.critic(state, action)

        critic_loss = nn.MSELoss()(q, y.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# Training loop
num_episodes = 500
#max_episode_steps was set when creating env

# Get the state and action dimensions and ranges
state_dim = env.observation_space['observation'].shape[0] + env.observation_space['desired_goal'].shape[0] # Concatenate observation and goal
action_dim = env.action_space.shape[0]
action_range = env.action_space.high[0]

agent = DDPGAgent(state_dim, action_dim, action_range)

for episode in range(num_episodes):
    state = env.reset()[0]
    state = np.concatenate([state['observation'], state['desired_goal']])
    episode_reward = 0

    for step in range(env.spec.max_episode_steps):
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_state = np.concatenate([next_state['observation'], next_state['desired_goal']])

        agent.replay_buffer.push(state, action, reward, next_state, done)
        agent.update()

        state = next_state
        episode_reward += reward

        if done:
            break

    print(f"Episode: {episode+1}, Reward: {episode_reward:.2f}")

env.close()