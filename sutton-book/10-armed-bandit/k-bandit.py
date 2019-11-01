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
    tot_reward, num_adoptions = action_data.get(action, (0,0))
        #functools.reduce(lambda a, b: map(sum, zip(*[a, b])),
        #                 [(step[1], 1) for step in history if step[0] == action],
        #                 (0, 0))
    if num_adoptions == 0:
        return 0
    else:
        return tot_reward / num_adoptions


def greedy_action_selection(problem: Problem, value_estimates: ValueEstimates, epsilon: float) -> Action:
    if random.random() >= epsilon:  # greedy action selection
        index = np.argmax(value_estimates.values()) # NB: if multiple maxes, chooses the first
    else:  # explore randomly
        index = problem.randrange(len(problem))
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
    action_data[chosen_action] = (prev[0]+1, prev[1]+actual_reward)

    history.append((chosen_action, actual_reward))

def single_run(problem: Problem, epsilon: float, nsteps: int) -> History:
    history: History = list()
    action_data = dict()
    estimated_value: Dict[ActionEntry] = dict( [(a[0], np.random.normal(a[1])) for a in problem] )
    print(estimated_value)
    for i in range(nsteps):
        single_run_step(problem, epsilon, estimated_value, history, action_data)
        # print("Step {}: choice {}".format(i,history[i]))
    return history

print("K-ARMED BANDIT PROBLEM\n*****************\n")

k: int = 10  # k-armed bandit problem

num_problems: int = 1
num_runs: int = 20
num_timesteps: int = 1000
problems: List[Problem] = []

# 1. Generate problems
mean: float = 0
stdev: float = 1
for i in range(0, num_problems):
    normalDistribSamples: np.ndarray = np.random.normal(mean, stdev, [10])
    problem: Problem = list(enumerate(normalDistribSamples))
    problems.append(problem)

p1 = problems[0]

histories = np.zeros((num_runs,num_timesteps,2))
epsilon = 0
for r in range(num_runs):
    print("Run {}".format(r))
    histories[r] = np.array(single_run(problem, epsilon, num_timesteps), dtype=(int,float))

steps = np.arange(0, num_timesteps)

# avg_reward = np.zeros((num_runs,num_timesteps)

avg_reward = np.mean(histories[:,:,1], 0) # TODO: FIX THIS CALCULATION

#avg_reward = []
#for i in steps:
#    tot_reward = functools.reduce(lambda x,y: x+y, [x[1] for x in p1_history[:i]], 0)
#    avg_reward.append(tot_reward/(i+1))

plt.figure()
plt.plot(steps, avg_reward, color="red", linewidth=2.5, linestyle="-", label="avg reward")
plt.show()