import random
import math
from itertools import count

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from DQN import DQN
from ReplayMemeoty import ReplayMemory, Transition
from Cube import CubeEnvironment as CubeEnv

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

SUFFLE_COUNT = 1
MAX_ITER = 20

device = 'cuda' if torch.cuda.is_available() else 'cpu'
steps_done = 0


def select_action(state, policy_net, env):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor(np.array([np.random.choice(env.actions, size=(1,))]), device=device, dtype=torch.long)


def optimize_model(memory, policy_net, target_net, optimizer):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)

    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(
            non_final_next_states).max(1)[0]

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values,
                     expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def train(env, memory, policy_net, target_net, optimizer):
    if torch.cuda.is_available():
        num_episodes = 600
    else:
        num_episodes = 50

    for i_episode in range(num_episodes):
        env.reset()
        env.suffle(SUFFLE_COUNT)
        state = torch.tensor(env.tensor.reshape(-1), dtype=torch.float32,
                             device=device).unsqueeze(0)

        print('New episode started')
        for t in count():
            action = select_action(state, policy_net, env)

            env.move(action.item())
            reward = int(env.done)
            print(f'reward: {reward}')

            reward = torch.tensor([reward], device=device)
            done = int(env.done)

            next_state = torch.tensor(
                env.tensor.reshape(-1), dtype=torch.float32, device=device).unsqueeze(0)

            if t == MAX_ITER:
                done = True

            memory.push(state, action, next_state, reward)
            state = next_state

            optimize_model(memory, policy_net, target_net, optimizer)

            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * \
                    TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                break

    print('Complete')


def main():
    env = CubeEnv()
    n_actions = len(env.actions)

    env.suffle(SUFFLE_COUNT)

    policy_net = DQN(3 * 3 * 3, 100, n_actions).to(device)
    target_net = DQN(3 * 3 * 3, 100, n_actions).to(device)
    # target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)

    train(env, memory, policy_net, target_net, optimizer)


if __name__ == '__main__':
    main()
