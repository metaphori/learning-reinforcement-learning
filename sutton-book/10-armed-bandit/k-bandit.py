from typing import List, Dict, Tuple

import functools
import random

import numpy as np
import matplotlib.pyplot as plt

# Types

Action = int
ActionValue = float
ActionEntry = Tuple[int, float]
Problem = List[ActionEntry]
History = List[ActionEntry]
ValueEstimates = Dict[Action, ActionValue]


def estimate_value(action: Action, history: History, action_data) -> float:
    num_adoptions, tot_reward = action_data.get(action, (0,0))
        #functools.reduce(lambda a, b: map(sum, zip(*[a, b])),
        #                 [(step[1], 1) for step in history if step[0] == action],
        #                 (0, 0))
    if num_adoptions == 0:
        return 0
    else:
        return tot_reward / num_adoptions


def greedy_action_selection(problem: Problem, value_estimates: ValueEstimates, epsilon: float) -> Action:
    if random.random() >= epsilon:  # greedy action selection
        #print("Greedy selection")
        index = np.argmax(list(value_estimates.values())) # NB: if multiple maxes, chooses the first
        # NOTE: dict.values() returns a dict_items() which breaks np.argmax
    else:  # explore randomly
        #print("Random selection")
        index = random.randrange(len(problem))
    return index


def single_run_step(problem: Problem, epsilon: float, estimated_value, history: History, action_data) -> None:
    # Estimate action value (sample-average method)
    for (action, _) in problem:
        estimated_value[action] = estimate_value(action, history, action_data)
    chosen_action = greedy_action_selection(problem, estimated_value, epsilon)
    mean = problem[chosen_action][1]
    variance = 1
    actual_reward = np.random.normal(mean, variance)

    prev = action_data.get(chosen_action, [0,0])
    #print("Given {}. \nChoose {} with reward {}\naction data {}\nhistory {}".format(estimated_value, chosen_action, actual_reward, action_data, history))
    #print("{}+1={}, {}+{}={}".format(prev[0],prev[0]+1,prev[1],actual_reward,prev[1]+actual_reward))
    action_data[chosen_action] = (prev[0]+1, prev[1]+actual_reward)

    history.append((chosen_action, actual_reward))

def single_run(problem: Problem, epsilon: float, nsteps: int) -> History:
    history: int = list()
    action_data: Dict[Action, (int,ActionValue)] = dict() # [(a[0], (1,np.random.normal(a[1]))) for a in problem] )
    estimated_value: Dict[ActionEntry] = dict() # dict( [(a[0], np.random.normal(a[1])) for a in problem] )
    for i in range(nsteps):
        single_run_step(problem, epsilon, estimated_value, history, action_data)
        # print("Step {}: choice {}".format(i,history[i]))
    return history

print("K-ARMED BANDIT PROBLEM\n*****************\n")

k: int = 10  # k-armed bandit problem

#num_problems: int = 1
num_runs: int = 2000
num_timesteps: int = 1000
problems: List[Problem] = []

# 1. Generate problems
mean: float = 0
stdev: float = 1
for i in range(0, num_runs):
    normalDistribSamples: np.ndarray = np.random.normal(mean, stdev, [10])
    problem: Problem = list(enumerate(normalDistribSamples))
    problems.append(problem)

epsilon_greedy = 0
epsilon_neargreedy = 0.01
epsilon_notsogreedy = 0.1
configs = [epsilon_greedy, epsilon_neargreedy, epsilon_notsogreedy]
colors = ["red", "blue", "green"]
labels = ["greedy", "near-greedy", "not-so-greedy"]

plt.figure()
plt.xlabel("time steps")
plt.ylabel("avg reward")

for k, epsilon in enumerate(configs):
    histories = np.zeros((num_runs, num_timesteps, 2))
    for r in range(num_runs):
        #print("Run {}".format(r))
        histories[r] = np.array(single_run(problems[r], epsilon, num_timesteps), dtype=(int, float))

    avg_reward = np.mean(histories[:,:,1], 0)

    steps = np.arange(0, num_timesteps)

    plt.plot(steps, avg_reward, color=colors[k], linewidth=2.5, linestyle="-", label=labels[k])

plt.legend(loc='lower right', frameon=False)
plt.show()