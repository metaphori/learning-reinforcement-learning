import abc
import functools
import random
from typing import Dict, List, Tuple
import copy

import numpy as np
import matplotlib.pyplot as plt
import math

global rldebug

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

    def __repr__(self): return "Action{}".format(self.id)

    def __hash__(self): return hash(self.id)
    def __eq__(self, other): return isinstance(other,Action) and self.id == other.id

class ActionData:
    def __init__(self, reward: Reward, optimal: bool = False):
        self.reward = reward
        self.optimal = optimal

    def set_reward(self, new_reward):
        self.reward = new_reward

    def set_optimal(self, opt):
        self.optimal = opt

    def __repr__(self): return "{}ActionData[reward={}]".format(
        '#' if self.optimal else '',
        self.reward)

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
        self.data: Dict[Action,ActionHistoryData] = dict()

    def __getitem__(self, item: Action) -> ActionHistoryData:
        if item not in self.data: self.data[item] = ActionHistoryData(item)
        return self.data.get(item)

    def __setitem__(self, key: Action, value: ActionHistoryData):
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

    def __repr__(self): return "Estimates[{}]".format({a:float("{0:.2f}".format(r)) for a,r in self.actions.items() })


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
    def action_selection(self, problem: Problem, value_estimates: ValueEstimates, action_data: ActionHistoryData, step: int) -> Action: pass

    def __repr__(self): return self.description

class RLGreedyMethod(RLMethod):
    def __init__(self,
                 epsilon: Reward = 0,            # default is fully greedy
                 step_size = lambda n,a: 1.0/n, # default is the sample-average method
                ):
        super().__init__("GreedyMethod[epsilon={}]".format(epsilon))
        self.epsilon = epsilon
        self.step_size = step_size

    def __repr__(self):
        return "Greedy[epsilon={}, step_size(n=1,n=10)=[{},{}]"\
            .format(self.epsilon, self.step_size(1,None), self.step_size(10,None))

    def estimate_value(self, action_data: ActionHistoryData, prev_estimates: ValueEstimates) -> Reward:
        num_adoptions = len(action_data)
        prev_estimate = prev_estimates.actions.get(action_data.action, 0)

        if num_adoptions == 0: return prev_estimate
        else:
            # return action_data.total_reward() / num_adoptions
            # Avoid full recalculation through the following formula (see Sutton book par. 2.4)
            prev_reward = action_data.data[-1][1]
            step_size = self.step_size(num_adoptions, action_data.action)
            result = prev_estimate + step_size * (prev_reward - prev_estimate)
            # rldebug.debug_step("{} + {} * ({} - {}) = {}".format(prev_estimate, step_size, prev_reward, prev_estimate, result))
            return result

    def action_selection(self, problem: Problem, value_estimates: ValueEstimates, action_data: ActionHistoryData, step: int) -> Action:
        if random.random() >= self.epsilon and len(value_estimates.actions)>0:  # greedy action selection
            potential_actions = Utils.all_maxes(value_estimates.actions)
            action = random.choice(potential_actions)
            # NOTE: dict.values() returns a dict_items() which breaks np.argmax
        else:  # explore randomly
            action = random.choice(list(problem.actions.keys()))
        return action

class UCBMethod(RLGreedyMethod):
    def __init__(self,
                 c: float,
                 epsilon: Reward = 0,  # default is fully greedy
                 step_size=lambda n, a: 1.0 / n,  # default is the sample-average method
                 ):
        super().__init__(epsilon, step_size)
        self.c = c

    def __repr__(self):
        return "UCBMethod[epsilon={}, c={}]".format(self.epsilon, self.c)

    def action_selection(self, problem: Problem, value_estimates: ValueEstimates, action_data: ActionsHistoryData, step: int) -> Action:
        # Adjust estimates to consider uncertainty
        adjusted_estimates = ValueEstimates({a:
            (
                e + self.c*math.sqrt(math.log(step)/len(action_data[a].data)) if len(action_data[a].data)>0 else float("inf")
            ) for a, e in value_estimates.actions.items()
                                            })
        return super().action_selection(problem, adjusted_estimates, action_data, step)

class Utils:
    @classmethod
    def all_maxes(self, d: Dict):
        maxes: List[Action] = []
        curr_max = None
        for k,v in d.items():
            if curr_max==None or v > curr_max: curr_max = v; maxes = [k]
            elif v == curr_max: maxes.append(k)
        return maxes

class RLDebugger:
    def __init__(self,
                 debug_problems: List[Problem] = [],
                 debug_within: range = range(0,15)):
        self.debug_problems: List[Problem] = debug_problems
        self.debug_within: range = debug_within
        self.current_problem: Problem = None
        self.current_time_step: int = 0

    def debug(self, s: str, consider_timestep: bool = False, header = True, endl="\n") -> None:
        if self.current_problem in self.debug_problems and \
                (not consider_timestep or self.current_time_step in self.debug_within):
            header_str = "[DEBUG{}] ".format(id(self.current_problem))
            print("{}{}".format(header_str if header else "", s), end=endl)

    def debug_step(self, s: str, header = True, endl="\n"):
        self.debug(s, consider_timestep=True, header=header, endl=endl)

    def set_current_problem(self, p: Problem):
        self.current_problem = p

    def set_current_timestep(self, tstep: int):
        self.current_time_step = tstep

    def reset(self):
        self.current_problem = None
        self.current_time_step = 0

class SimulationConfig:
    def __init__(self,
                 method: RLMethod,
                 initial_value_estimates = None,
                 plotting_info: Dict = {}):
        self.method = method
        self.initial_value_estimates = initial_value_estimates
        self.plotting_info = plotting_info

    def __repr__(self):
        return "SimulationConfig[\n\tmethod={},\n\tinitial_value_estimates={}".format(self.method, self.initial_value_estimates)

class RLSimulator:
    def __init__(self, config: SimulationConfig, problem: Problem):
        self.config = config
        self.problem = problem
        self.history = History()
        self.time_step = 0
        rldebug.set_current_problem(problem)

    def single_run_step(self, estimates: ValueEstimates) -> None:
        chosen_action = self.config.method.action_selection(self.problem, estimates, self.history.actions_history_data, self.time_step)
        mean = self.problem[chosen_action].reward
        variance = 1
        actual_reward = np.random.normal(mean, variance)

        optimal_action = max(self.problem.actions, key=lambda e: self.problem.actions[e].reward)
        optimal = chosen_action == optimal_action

        rldebug.debug_step("Step {}. Given\n{}. \nChoose {}{} with reward {} {}"
                           .format(self.time_step, estimates, "+" if optimal else "#", chosen_action, actual_reward,
                                   "while optimal is {}".format(optimal_action) if not optimal else "")) #, self.history))

        entry = ActionEntry(chosen_action, ActionData(actual_reward, optimal))
        self.history.add_action(self.time_step, entry) # [chosen_action] = (prev[0] + 1, prev[1] + actual_reward)

        # Update estimate of action value
        ne = self.config.method.estimate_value(self.history.actions_history_data[chosen_action], estimates)
        estimates.update_estimate(chosen_action, ne)

        self.update_problem()

    def single_run(self, nsteps: int) -> History:
        self.time_step = 0
        # NB: estimates is to be changed in place, hence a copy of the initial estimates structure must be used!!!
        estimates = copy.deepcopy(self.config.initial_value_estimates) if self.config.initial_value_estimates else ValueEstimates(
            { a:0.0 for a in self.problem.actions }
        )
        rldebug.debug("SOLVING {} on {} time steps".format(self.problem, nsteps))
        for i in range(nsteps):
            rldebug.set_current_timestep(i)
            self.single_run_step(estimates)
            self.time_step += 1
            # print("Step {}: choice {}".format(i,history[i]))
        rldebug.debug("\nSOLVED {}\n{}\n".format(self.problem, self.history))
        rldebug.reset()
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
        maxes = Utils.all_maxes({ k:v.reward for k,v in problem.actions.items() })
        for opt_action in maxes: problem[opt_action].set_optimal(True)
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
        print("\nRun experiments for config\n{}\nwith simulator {}".format(config,S.__name__))
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

rldebug = RLDebugger()
