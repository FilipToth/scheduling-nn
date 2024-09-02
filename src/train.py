import os
import gymnasium as gym
import math
import random
import environments.env as environments
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

env = environments.MachineEnvironment()
env.reset()

# enable pyplot interactive
plt.ion()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity) -> None:
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions) -> None:
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


BATCH_SIZE = 128
GAMMA = 0.99 # discount factor
EPS_START = 0.9 # epsilon start value
EPS_END = 0.05 # final epsilon value
EPS_DECAY = 1000 # controls the rate of exponential decay of epsilon, higher means a slower decay
TAU = 0.005 # update rate of target network
LR = 1e-4 # learning rate

OUT_DIR = '../out'

n_actions = env.action_space.n
print(n_actions)
state, info = env.reset()
n_observations = len(state)
print(state.shape)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
    global steps_done

    sample = random.random()
    eps_treshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * steps_done / EPS_DECAY)
    steps_done += 1

    if sample > eps_treshold:
        with torch.no_grad():
            # pick largest expected reward
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_rewards = []

latest_figure = None
def plot_rewards(show_result=False):
    global latest_figure

    latest_figure = plt.figure(1)
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)

    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training')

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(rewards_t.numpy())

    # take 100 episode averages and plot them
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    # pause for plots to update
    plt.pause(0.001)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)

    # also need to transpose the batch
    batch = Transition(*zip(*transitions))

    # compute non-final mask and concat elements of the batch
    # a final state is a state after the simulation ended
    non_final_mask  = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    # compute expected Qs
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # compute Hubert loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # optimize the model
    optimizer.zero_grad()
    loss.backward()

    # in-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


num_episodes = 50
if torch.cuda.is_available():
    num_episodes = 700

for i_episode in range(num_episodes):
    # environment initialization and get its state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    cumulative_reward = 0
    for _ in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        cumulative_reward += reward

        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # store transition in memory
        memory.push(state, action, next_state, reward)

        # move to the next state
        state = next_state

        # perform optimization step
        optimize_model()

        # soft update network weights
        # \theta' <- \theta + (1 - \tau)\theta'

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()

        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)

        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_rewards.append(cumulative_reward)
            plot_rewards()
            break

print('Done!')
plot_rewards(show_result=True)
plt.ioff()
plt.show()

num_files = 0
for file in os.listdir(OUT_DIR):
    path = os.path.join(OUT_DIR, file)
    if not os.path.isfile(path):
        continue

    num_files += 1

plot_path = os.path.join(OUT_DIR, f'{num_files}.png')
latest_figure.savefig(plot_path)
