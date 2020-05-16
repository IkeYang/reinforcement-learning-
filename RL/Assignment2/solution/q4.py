import numpy as np
import random
import math
import matplotlib.pyplot as plt
random.seed(42)
num_runs = 100
num_episodes = 100

state_list = {'left_end': 0,
                'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5,
                'right_end': 6}
action_list = [-1, +1]

def mc_random_sample(start):
    sample_list = [start]
    while start != 0 and start != 6:
        action = random.choice(action_list)
        start += action
        sample_list.append(start)
    return sample_list

def rms_error(value_list):
    true_value = np.array([1/6, 2/6, 3/6, 4/6, 5/6])
    return math.sqrt(sum((value_list[1:6]-true_value)**2)/5)

def plot(value_list):
    x = list(range(num_episodes))
    plt.figure(figsize=(10,6))
    ax1 = plt.subplot(111)
    plt.plot(x, value_list)
    plt.xlabel('Walks/Episodes', fontdict={'weight': 'normal', 'size': 10})
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    plt.show()
    
def mc(alpha):
    rmse_list = np.zeros(num_episodes)
    for run_idx in range(num_runs):
        v_list = np.zeros(7)
        v_list[1:6] = 0.5
        v_list[6] = 0
        for episode in range(num_episodes):
            rmse_list[episode] += rms_error(v_list)
            start = 3
            sample_list = mc_random_sample(start)
            reward = 1 if sample_list[len(sample_list)-1] == 6 else 0
            for sample in sample_list[:len(sample_list)-1]:
                v_list[sample] = v_list[sample] + alpha*(reward-v_list[sample])
    rmse_list /= num_runs
    return rmse_list

def td_action():
    return random.choice(action_list)

def td(alpha):
    rmse_list = np.zeros(num_episodes)
    for run_idx in range(num_runs):
        v_list = np.zeros(7)
        v_list[1:6] = 0.5
        v_list[6] = 0
        for episode in range(num_episodes):
            rmse_list[episode] += rms_error(v_list)
            old_state = state = 3
            while state != 0 and state != 6:
                old_state = state
                action = td_action()
                state += action
                reward = 1 if state == 6 else 0
                v_list[old_state] = v_list[old_state] + alpha*(reward+v_list[state]-v_list[old_state])
    rmse_list /= num_runs
    return rmse_list

def main():
    plt.figure(figsize=(10,6))
    ax1 = plt.subplot(111)
    td_alpha = [0.15, 0.1, 0.05]
    mc_alpha = [0.01, 0.02, 0.03, 0.04]
    for method in ['TD', 'MC']:
        if method == 'TD':
            for alpha in td_alpha:
                rms_error = td(alpha)
                plt.plot(rms_error, label=method+', alpha=%.02f'%(alpha))
        else:
            for alpha in mc_alpha:
                rms_error = mc(alpha)
                plt.plot(rms_error, label=method+', alpha=%.02f'%(alpha))
    plt.ylabel('RMSE', fontdict={'weight': 'normal'})
    plt.xlabel('Walks/Episodes', fontdict={'weight': 'normal'})
    plt.legend()
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    plt.show()

if __name__ == "__main__":
    main()

