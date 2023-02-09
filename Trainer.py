import random
import math
from itertools import count

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from DQN import DQN
from ReplayMemeoty import ReplayMemory, Transition
from Cube import CubeEnvironment as CubeEnv

BATCH_SIZE = 64
GAMMA = 1
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-3

UPDATE_FREQ = 2

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
    # criterion = nn.MSELoss()
    loss = criterion(state_action_values,
                     expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()

    # torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def train(env, memory, policy_net, target_net, optimizer, shuffle_count, max_iter):
    log = []
    win = 0
    last_rate = 0
    epsilon = 1e-5
    early_stopping_counter = 0
    patience = 50

    for i_episode in count():
        env.reset()
        env.suffle(np.random.choice(shuffle_count) + 1)
        # env.suffle(suffle_count)
        state = torch.tensor(env.tensor.reshape(-1), dtype=torch.float32,
                             device=device).unsqueeze(0)

        if (i_episode + 1) % 100 == 0:
            print(f'100 more eposids are done --> {i_episode + 1}, rate: {last_rate}, depth: {shuffle_count}')
        # print(f'New episode started ({i_episode+1})')
        for t in count():
            action = select_action(state, policy_net, env)

            env.move(action.item())
            # reward = int(env.done) * 10
            reward = env.score - 7
            reward = torch.tensor([reward], device=device)
            done = int(env.done)

            next_state = torch.tensor(
                env.tensor.reshape(-1), dtype=torch.float32, device=device).unsqueeze(0)

            if done:
                reward = reward = torch.tensor([50], device=device)
                win += 1

            log.append(win / (i_episode + 1))
            if t == max_iter:
                reward = reward = torch.tensor([-10], device=device)
                done = True
                next_state = None

            memory.push(state, action, next_state, reward)
            state = next_state

            optimize_model(memory, policy_net, target_net, optimizer)

            # if t % UPDATE_FREQ == 0:
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * \
                    TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)
            # target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        rate = win / (i_episode + 1)
        # if (i_episode + 1) % 100 == 0:
        #     print(rate - last_rate)
        if rate - last_rate < epsilon or rate > .99:
            early_stopping_counter += 1
        else:
            early_stopping_counter = 0

        last_rate = rate
        if early_stopping_counter == patience:
            break

    print('Complete')
    return np.array(log)


def calculate_mean_reward(log, window_size):
    avg = []

    total_lenght = len(log)
    for i in range(total_lenght - window_size):
        avg.append(log[i:i+window_size].mean())

    return avg


def main():
    global steps_done

    env = CubeEnv()
    n_actions = len(env.actions)
    latent_dim = 512
    policy_net = DQN(3 * 3 * 3, latent_dim, n_actions).to(device)
    target_net = DQN(3 * 3 * 3, latent_dim, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    # optimizer = optim.Adam(policy_net.parameters(), lr=LR, weight_decay=1e-2)
    optimizer = optim.RMSprop(policy_net.parameters(), lr=LR, weight_decay=1e-2)
    memory = ReplayMemory(10000)

    for i in range(1, 11, 2):
        memory.reset()
        steps_done = 0

        # update_freq_offset = 0 if i == 1 else 150

        log = train(env, memory, policy_net, target_net, optimizer, i, MAX_ITER)
        # EPS_DECAY *= 1.2
        # avg_log = calculate_mean_reward(log, 50)
        # cum_log = np.cumsum(log) / np.cumsum(np.ones(log.shape))
        plt.plot(log)
        plt.show()


if __name__ == '__main__':
    main()
