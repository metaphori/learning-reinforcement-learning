import rlutils
from rlutils import *

if __name__ == '__main__':
    # 1. Generate problems
    num_runs = 5000
    time_steps = 800
    problems: List[Problem] = generate_bandit_problems(10, num_runs)

    #rlutils.rldebug = RLDebugger(debug_problems=problems, debug_within=range(998,1000))

    # 2. Configure simulations and plotting
    configs = [
        SimulationConfig(UCBMethod(c = 2.0, epsilon=0.0, step_size=lambda k,a: 1/k),
                         plotting_info={"label": "UCB[c=2]",
                                        "color": "green",
                                        "graph":Plots.AVG_REWARD}),
        SimulationConfig(RLGreedyMethod(epsilon=0.1, step_size=lambda k,a: 1/k),
                         plotting_info={"label": "near-greedy[0.1]",
                                        "color": "red",
                                        "graph":Plots.AVG_REWARD})
    ]

    execute_and_plot(configs, num_runs, time_steps, problems)