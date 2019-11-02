import abc
from collections import OrderedDict
from typing import List, Dict, Tuple

import functools
import random

import numpy as np
import matplotlib.pyplot as plt

# Types

class Action:
    def __init__(self, id):
        self.id = id

    def __repr__(self): return "Action[id={}]".format(self.id)

    def __hash__(self): return hash(self.id)
    def __eq__(self, other): return isinstance(other,Action) and self.id == other.id

class ActionData:
    def __init__(self, reward: float):
        self.reward = reward

    def __repr__(self): return "ActionData[reward={}]".format(self.reward)

class ActionEntry:
    def __init__(self, action: Action, data: ActionData):
        self.action = action
        self.data = data

    def __repr__(self) -> str:
        return "actionId={} (reward={})".format(self.action.id, self.data.reward)

class ActionHistoryData:
    def __init__(self, action: Action):
        self.action = action
        self.data = OrderedDict()

    def add_usage(self, time_step: int, actual_reward: float) -> None:
        self.data[time_step] = actual_reward

    def __len__(self):
        return len(self.data)

    def total_reward(self) -> float:
        #print(list(self.data.values()))
        return functools.reduce(lambda a,b: a+b, list(self.data.values()))

    def __repr__(self): return str(self.data)

class ActionsHistoryData:
    def __init__(self):
        self.data = OrderedDict()

    def __getitem__(self, item) -> ActionHistoryData:
        if item not in self.data: self.data[item] = ActionHistoryData(item)
        return self.data.get(item)

    def __setitem__(self, key, value):
        self.data[key] = value

    def __repr__(self): return str(self.data)

class History:
    def __init__(self):
        self.data: OrderedDict[int,ActionEntry] = OrderedDict()
        self.actions_history_data = ActionsHistoryData()

    def add_action(self, time_step: int, action_entry: ActionEntry) -> None:
        a = action_entry.action
        self.data[time_step] = action_entry
        self.actions_history_data[a].add_usage(time_step, action_entry.data.reward)

    def __repr__(self): return "History\n\t- {}\n\t- {}".format(str(self.data), str(self.actions_history_data))

class Problem:
    def __init__(self, actions: Dict[Action,ActionData]):
        self.actions = actions

    def __getitem__(self, item) -> ActionData:
        return self.actions[item]

    def __str__(self):
        return "Problem[ {} ]".format(self.actions)

class ValueEstimates:
    def __init__(self, actions = dict()):
        self.actions = actions

    def update_estimate(self, action: Action, reward: float):
        #print("Update estimate for action {} with reward {}".format(e.action, e.data.reward))
        self.actions[action] = reward

    def __repr__(self): return "Estimates[{}]".format(self.actions)


class RLMethod(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def estimate_value(self, action_data: ActionHistoryData) -> ActionData: pass

    @abc.abstractmethod
    def action_selection(self, problem: Problem, value_estimates: ValueEstimates) -> Action: pass

class RLGreedyMethod(RLMethod):
    def __init__(self, epsilon: float, plotting_info):
        self.epsilon = epsilon
        self.plotting_info = plotting_info

    def estimate_value(self, action_data: ActionHistoryData) -> float:
        #print("Estimating value of {} with history {}".format(action_data.action, action_data.data))
        num_adoptions = len(action_data)
        if num_adoptions == 0:
            return 0
        else:
            return action_data.total_reward() / num_adoptions

    def action_selection(self, problem: Problem, value_estimates: ValueEstimates) -> Action:
        if random.random() >= self.epsilon:  # greedy action selection
            action = max(value_estimates.actions, key=value_estimates.actions.get) # NB: if multiple maxes, chooses the first
            # NOTE: dict.values() returns a dict_items() which breaks np.argmax
        else:  # explore randomly
            action = random.choice(list(problem.actions.keys()))
        return action


class RLSimulator:
    def __init__(self, method: RLMethod, problem: Problem):
        self.method = method
        self.problem = problem
        self.history = History()
        self.time_step = 0

    def single_run_step(self, estimates) -> None:
        # Estimate action value (sample-average method)
        for (action, _) in self.problem.actions.items():
            e = self.method.estimate_value(self.history.actions_history_data[action])
            estimates.update_estimate(action, e)
        chosen_action = self.method.action_selection(problem, estimates)
        mean = problem[chosen_action].reward
        variance = 1
        actual_reward = np.random.normal(mean, variance)

        prev = self.history.data.get(chosen_action, [0, 0])
        print("Given {}. \nChoose {} with reward {}\nhistory {}".format(
            estimates, chosen_action, actual_reward, self.history))
        print("{}+1={}, {}+{}={}\n".format(prev[0],prev[0]+1,prev[1],actual_reward,prev[1]+actual_reward))

        entry = ActionEntry(chosen_action, ActionData(actual_reward))
        self.history.add_action(self.time_step, entry) # [chosen_action] = (prev[0] + 1, prev[1] + actual_reward)
        #self.history.append((chosen_action, actual_reward))

    def single_run(self, nsteps: int) -> History:
        self.time_step = 0
        #print("Solving ", problem)
        estimates = ValueEstimates()
        for i in range(nsteps):
            self.single_run_step(estimates)
            self.time_step += 1
            # print("Step {}: choice {}".format(i,history[i]))
        return self.history

print("K-ARMED BANDIT PROBLEM\n*****************\n")

k: int = 10  # k-armed bandit problem

#num_problems: int = 1
num_runs: int = 1
num_timesteps: int = 10
problems: List[Problem] = []

# 1. Generate problems
mean: float = 0
stdev: float = 1
for i in range(0, num_runs):
    normalDistribSamples: np.ndarray = np.random.normal(mean, stdev, [10])
    problem: Problem = Problem({ Action(k) : ActionData(v) for (k,v) in dict(enumerate(normalDistribSamples)).items()})
    problems.append(problem)

# 2. Configure simulations and plotting
configs = [
    RLGreedyMethod(epsilon=0, plotting_info={ "label": "greedy", "color": "red"}),
    RLGreedyMethod(epsilon=0.01, plotting_info={ "label": "near-greedy", "color": "blue"}),
    RLGreedyMethod(epsilon=0.1, plotting_info={ "label": "not-so-greedy", "color": "green"}),
]

plt.figure()
plt.xlabel("time steps")
plt.ylabel("avg reward")

# 3. Run simulations (1) for any configuration and (2) for any problem instance
for k, method in enumerate(configs):
    histories = np.zeros((num_runs, num_timesteps, 2))
    print("Run experiments for config ", method)
    for r in range(num_runs):
        print(".", end='')
        #print("Run {}".format(r))
        simulator = RLSimulator(method, problems[r])
        h = simulator.single_run(num_timesteps)
        hh = [(k,v.data.reward) for (k,v) in h.data.items()]
        #print(hh)
        histories[r] = np.array(hh, dtype=(int, float))

    avg_reward = np.mean(histories[:,:,1], 0)

    steps = np.arange(0, num_timesteps)

    plt.plot(steps, avg_reward, color=method.plotting_info["color"], linewidth=2.5, linestyle="-", label=method.plotting_info["label"])

plt.legend(loc='lower right', frameon=False)
plt.show()