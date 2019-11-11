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
simtype = RLSimulatorNonStationary if args.nonstationary else RLSimulator

print("Use ", simtype)

if __name__ == '__main__':
    # 1. Generate problems
    problems: List[Problem] = generate_bandit_problems(k, num_runs)

    # 2. Configure simulations and plotting
    configs = [
        SimulationConfig(RLGreedyMethod(epsilon=0), plotting_info={ "label": "greedy[0]", "color": "red", "graph":Plots.AVG_REWARD }),
        SimulationConfig(RLGreedyMethod(epsilon=0.01), plotting_info={ "label": "near-greedy[0.01]", "color": "blue", "graph":Plots.AVG_REWARD }),
        SimulationConfig(RLGreedyMethod(epsilon=0.1), plotting_info={ "label": "not-so-greedy[0.1]", "color": "green", "graph":Plots.AVG_REWARD }),
    ]

    execute_and_plot(configs, num_runs, num_timesteps, problems, simtype)