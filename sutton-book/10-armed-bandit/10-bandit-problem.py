from typing import List

import numpy as np
from rlutils import *
from os import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-k", help="num of actions", type=int, default=10)
parser.add_argument("-r", help="num of runs", type=int, default=1000)
parser.add_argument("-t", help="num of time steps", type=int, default=1000)
parser.add_argument("-S", "--nonstationary", help="", action="store_true")
args = parser.parse_args()

k: int = args.k # 10  # k-armed bandit problem
num_runs: int = args.r
num_timesteps: int = args.t
problems: List[Problem] = []
simtype = RLSimulatorNonStationary if args.nonstationary else RLSimulator

print("Use ", simtype)

if __name__ == '__main__':
    # 1. Generate problems
    mean: float = 0
    stdev: float = 1
    for i in range(0, num_runs):
        normalDistribSamples: np.ndarray = np.random.normal(mean, stdev, [k])
        problem: Problem = Problem({ Action(k) : ActionData(v) for (k,v) in dict(enumerate(normalDistribSamples)).items()})
        problems.append(problem)

    # 2. Configure simulations and plotting
    configs = [
        RLGreedyMethod(epsilon=0, plotting_info={ "label": "greedy[0]", "color": "red"}),
        RLGreedyMethod(epsilon=0.01, plotting_info={ "label": "near-greedy[0.01]", "color": "blue"}),
        RLGreedyMethod(epsilon=0.1, plotting_info={ "label": "not-so-greedy[0.1]", "color": "green"}),
    ]

    execute_and_plot(configs, num_runs, num_timesteps, problems, simtype)