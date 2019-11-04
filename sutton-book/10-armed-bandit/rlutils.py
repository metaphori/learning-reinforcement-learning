import abc
import functools
import random
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Types
Reward = float
TimeStep = int

# On plots
from enum import Enum
class Plots(Enum):
    AVG_REWARD = "avg reward"
    OPTIMAL_ACTION = "opt action"

class Action:
    def __init__(self, id):
        self.id = id

    def __repr__(self): return "Action[id={}]".format(self.id)

    def __hash__(self): return hash(self.id)
    def __eq__(self, other): return isinstance(other,Action) and self.id == other.id

class ActionData:
    def __init__(self, reward: Reward, optimal: bool = False):
        self.reward = reward
        self.optimal = optimal

    def set_reward(self, new_reward):
        self.reward = new_reward

    def __repr__(self): return "ActionData[reward={}]".format(self.reward)

class ActionEntry:
    def __init__(self, action: Action, data: ActionData):
        self.action = action
        self.data = data

    def __repr__(self) -> str:
        return "actionId={}[{}]".format(self.action.id, self.data)

class ActionHistoryData:
    def __init__(self, action: Action):
        self.action = action
        self.data: List[Tuple[TimeStep,Reward]] = []

    def add_usage(self, time_step: TimeStep, actual_reward: Reward) -> None:
        self.data.append((time_step,actual_reward))

    def __len__(self):
        return len(self.data)

    def total_reward(self) -> Reward:
        #print(list(self.data.values()))
        return functools.reduce(lambda a,b: a+b[1], self.data, 0.)

    def __repr__(self): return str(self.data)

class ActionsHistoryData:
    def __init__(self):
        self.data = dict()

    def __getitem__(self, item) -> ActionHistoryData:
        if item not in self.data: self.data[item] = ActionHistoryData(item)
        return self.data.get(item)

    def __setitem__(self, key, value):
        self.data[key] = value

    def __repr__(self): return str(self.data)

class History:
    def __init__(self):
        self.data: [(TimeStep,ActionEntry)] = []
        self.actions_history_data = ActionsHistoryData()

    def add_action(self, time_step: TimeStep, action_entry: ActionEntry) -> None:
        a = action_entry.action
        self.data.append((time_step,action_entry))
        self.actions_history_data[a].add_usage(time_step, action_entry.data.reward)

    def __repr__(self): return "History\n\t- {}\n\t- {}".format(str(self.data), str(self.actions_history_data))

class ValueEstimates:
    def __init__(self, actions = dict()):
        self.actions = actions

    def update_estimate(self, action: Action, reward: Reward):
        #print("Update estimate for action {} with reward {}".format(e.action, e.data.reward))
        self.actions[action] = reward

    def __repr__(self): return "Estimates[{}]".format(self.actions)


class Problem:
    def __init__(self, actions: Dict[Action,ActionData]):
        self.actions = actions

    def __getitem__(self, item) -> ActionData:
        return self.actions[item]

    def __str__(self):
        return "Problem[ {} ]".format(self.actions)

class RLMethod(metaclass=abc.ABCMeta):
    def __init__(self, description):
        self.description = description

    @abc.abstractmethod
    def estimate_value(self, action_data: ActionHistoryData, old_estimates: ValueEstimates) -> Reward: pass

    @abc.abstractmethod
    def action_selection(self, problem: Problem, value_estimates: ValueEstimates) -> Action: pass

    def __repr__(self): return self.description

class RLGreedyMethod(RLMethod):
    def __init__(self,
                 epsilon: Reward = 0,            # default is fully greedy
                 step_size = lambda n,a: 1.0/n, # default is the sample-average method
                ):
        super().__init__("GreedyMethod[epsilon={}]".format(epsilon))
        self.epsilon = epsilon
        self.step_size = step_size

    def estimate_value(self, action_data: ActionHistoryData, prev_estimates: ValueEstimates) -> Reward:
        #print("Estimating value of {} with history {}".format(action_data.action, action_data.data))

        num_adoptions = len(action_data)
        prev_estimate = prev_estimates.actions.get(action_data.action, 0)

        if num_adoptions == 0: return prev_estimate
        else:
            # return action_data.total_reward() / num_adoptions
            # Avoid full recalculation through the following formula (see Sutton book par. 2.4)
            prev_reward = action_data.data[-1][1]
            step_size = self.step_size(num_adoptions, action_data.action)
            #print("{} + {} * ({} - {})".format(prev_estimate, step_size, prev_reward, prev_estimate))
            return prev_estimate + step_size * (prev_reward - prev_estimate)

    def action_selection(self, problem: Problem, value_estimates: ValueEstimates) -> Action:
        if random.random() >= self.epsilon and len(value_estimates.actions)>0:  # greedy action selection
            action = max(value_estimates.actions, key=value_estimates.actions.get) # NB: if multiple maxes, chooses the first
            # NOTE: dict.values() returns a dict_items() which breaks np.argmax
        else:  # explore randomly
            action = random.choice(list(problem.actions.keys()))
        return action

class SimulationConfig:
    def __init__(self, method: RLMethod, initial_value_estimates = None, plotting_info: Dict = {}):
        self.method = method
        self.initial_value_estimates = initial_value_estimates
        self.plotting_info = plotting_info

    def __repr__(self):
        return "SimulationConfig[method={}, initial_value_estimates={}".format(self.method, self.initial_value_estimates)

class RLSimulator:
    def __init__(self, config: SimulationConfig, problem: Problem):
        self.config = config
        self.problem = problem
        self.history = History()
        self.time_step = 0

    def single_run_step(self, estimates: ValueEstimates) -> None:
        chosen_action = self.config.method.action_selection(self.problem, estimates)
        mean = self.problem[chosen_action].reward
        variance = 1
        actual_reward = np.random.normal(mean, variance)

        #print("Given {}. \nChoose {} with reward {}\nhistory".format(
        #    estimates, chosen_action, actual_reward)) #, self.history))

        optimal = chosen_action==max(self.problem.actions, key=lambda e: self.problem.actions[e].reward)
        entry = ActionEntry(chosen_action, ActionData(actual_reward, optimal))
        self.history.add_action(self.time_step, entry) # [chosen_action] = (prev[0] + 1, prev[1] + actual_reward)
        #self.history.append((chosen_action, actual_reward))

        # Update estimate of action value
        ne = self.config.method.estimate_value(self.history.actions_history_data[chosen_action], estimates)
        estimates.update_estimate(chosen_action, ne)

        self.update_problem()

    def single_run(self, nsteps: int) -> History:
        self.time_step = 0
        estimates = self.config.initial_value_estimates if self.config.initial_value_estimates else ValueEstimates(
            { a:0.0 for a in self.problem.actions }
        )
        #print("Solving {} with initial estimates {}".format(self.problem, estimates))
        for i in range(nsteps):
            self.single_run_step(estimates)
            self.time_step += 1
            # print("Step {}: choice {}".format(i,history[i]))
        return self.history

    def update_problem(self):
        pass

class RLSimulatorNonStationary(RLSimulator):
    def __init__(self, config: SimulationConfig, problem: Problem, mean=0, variance=0.01):
        super().__init__(config, problem)
        self.mean = mean
        self.variance = variance

    def update_problem(self):
        # print("Updating problem ", self.problem)
        for k,a in enumerate(self.problem.actions):
            # Independent random walk for the true action values
            new_reward = self.problem.actions[a].reward + np.random.normal(self.mean, self.variance)
            self.problem.actions[a].set_reward(new_reward)

def generate_bandit_problems(k, num_runs, mean = 0, stdev = 1):
    problems: List[Problem] = []
    for i in range(0, num_runs):
        normalDistribSamples: np.ndarray = np.random.normal(mean, stdev, [k])
        problem: Problem = Problem({ Action(k) : ActionData(v) for (k,v) in dict(enumerate(normalDistribSamples)).items()})
        problems.append(problem)
    return problems

def execute_and_plot(configs: List[SimulationConfig],
         num_runs: int,
         num_timesteps: int,
         problems: List[Problem],
         S = RLSimulator):
    plt.figure()
    plt.xlabel("time steps")

    for k, config in enumerate(configs):
        histories = np.zeros((num_runs, num_timesteps, 2))
        print("\nRun experiments for config ", config)
        for r in range(num_runs):
            if r%10==0: print(".", end='\n' if r%500==0 else '')
            #print("Run {}".format(r))
            simulator = S(config, problems[r])
            h = simulator.single_run(num_timesteps)

            to_plot = config.plotting_info.get("graph", Plots.AVG_REWARD)
            plt.ylabel(to_plot)
            if to_plot==Plots.AVG_REWARD:
                hh = [[k,v.data.reward] for (k,v) in h.data]
            elif to_plot:
                hh = [[k, v.data.optimal] for (k, v) in h.data]
            else:
                raise Exception("Plot not recognised")
            histories[r] = np.array(hh)

        avg_reward = np.mean(histories[:,:,1], 0)

        steps = np.arange(0, num_timesteps)

        plt.plot(steps, avg_reward, color=config.plotting_info["color"], linewidth=1.5, linestyle="-", label=config.plotting_info["label"])

    plt.legend(loc='lower right', frameon=False)
    plt.show()