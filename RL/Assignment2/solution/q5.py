import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm

H_WORLD = 4
W_WORLD = 12
EPSILON = 0.1
ALPHA = 0.5
GAMMA = 1
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
ACTIONS = [UP, DOWN, LEFT, RIGHT]
START = [3, 0]
GOAL = [3, 11]
def step(state, action):
    i, j = state
    if action == UP:
        next_state = [max(i - 1, 0), j]
    elif action == LEFT:
        next_state = [i, max(j - 1, 0)]
    elif action == RIGHT:
        next_state = [i, min(j + 1, W_WORLD - 1)]
    elif action == DOWN:
        next_state = [min(i + 1, H_WORLD - 1), j]
    else:
        assert False

    reward = -1
    if (action == DOWN and i == 2 and 1 <= j <= 10) or (
        action == RIGHT and state == START):
        reward = -100
        next_state = START

    return next_state, reward

def choose_action(state, q_value):
    if np.random.binomial(1, EPSILON) == 1:
        return np.random.choice(ACTIONS)
    else:
        values_ = q_value[state[0], state[1], :]
        return np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

def sarsa(q_value, expected=False, step_size=ALPHA):
    state = START
    action = choose_action(state, q_value)
    rewards = 0.0
    while state != GOAL:
        next_state, reward = step(state, action)
        next_action = choose_action(next_state, q_value)
        rewards += reward
        if not expected:
            target = q_value[next_state[0], next_state[1], next_action]
        else:
            target = 0.0
            q_next = q_value[next_state[0], next_state[1], :]
            best_actions = np.argwhere(q_next == np.max(q_next))
            for action_ in ACTIONS:
                if action_ in best_actions:
                    target += ((1.0 - EPSILON) / len(best_actions) + EPSILON / len(ACTIONS)) * q_value[next_state[0], next_state[1], action_]
                else:
                    target += EPSILON / len(ACTIONS) * q_value[next_state[0], next_state[1], action_]
        target *= GAMMA
        q_value[state[0], state[1], action] += step_size * (
                reward + target - q_value[state[0], state[1], action])
        state = next_state
        action = next_action
    return rewards

def q_learning(q_value, step_size=ALPHA):
    state = START
    rewards = 0.0
    while state != GOAL:
        action = choose_action(state, q_value)
        next_state, reward = step(state, action)
        rewards += reward
        q_value[state[0], state[1], action] += step_size * (
                reward + GAMMA * np.max(q_value[next_state[0], next_state[1], :]) -
                q_value[state[0], state[1], action])
        state = next_state
    return rewards

def print_optimal_policy(q_value):
    optimal_policy = []
    for i in range(0, H_WORLD):
        optimal_policy.append([])
        for j in range(0, W_WORLD):
            if [i, j] == GOAL:
                optimal_policy[-1].append('G')
                continue
            bestAction = np.argmax(q_value[i, j, :])
            if bestAction == UP:
                optimal_policy[-1].append('U')
            elif bestAction == DOWN:
                optimal_policy[-1].append('D')
            elif bestAction == LEFT:
                optimal_policy[-1].append('L')
            elif bestAction == RIGHT:
                optimal_policy[-1].append('R')


def figure_to_plot():
    episodes = 500
    runs = 50
    rewards_sarsa = np.zeros(episodes)
    rewards_q_learning = np.zeros(episodes)
    for r in tqdm(range(runs)):
        q_sarsa = np.zeros((H_WORLD, W_WORLD, 4))
        q_q_learning = np.copy(q_sarsa)
        for i in range(0, episodes):
            rewards_sarsa[i] += sarsa(q_sarsa)
            rewards_q_learning[i] += q_learning(q_q_learning)

    rewards_sarsa /= runs
    rewards_q_learning /= runs

    plt.plot(rewards_sarsa, label='Sarsa',color='red')
    plt.plot(rewards_q_learning, label='Q-Learning',color='gray')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')
    plt.ylim([-100, 0])
    plt.legend()

    plt.show()

#    print('Sarsa Optimal Policy:')
#    print_optimal_policy(q_sarsa)
#    print('Q-Learning Optimal Policy:')
#    print_optimal_policy(q_q_learning)


if __name__ == '__main__':
    figure_to_plot()

