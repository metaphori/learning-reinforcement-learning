from rlutils import *

# Exercise 2.5 (programming) Design and conduct an experiment to demonstrate the
# difficulties that sample-average methods have for nonstationary problems. Use a modified
# version of the 10-armed testbed in which all the q ⇤ (a) start out equal and then take
# independent random walks (say by adding a normally distributed increment with mean
# zero and standard deviation 0.01 to all the q ⇤ (a) on each step). Prepare plots like
# Figure 2.2 for an action-value method using sample averages, incrementally computed,
# and another action-value method using a constant step-size parameter, ↵ = 0.1. Use
# " = 0.1 and longer runs, say of 10,000 steps

if __name__ == '__main__':
    # 1. Generate problems
    num_runs = 1000
    time_steps = 10000
    problems: List[Problem] = generate_bandit_problems(10, num_runs, mean=0.5, stdev=0)

    # 2. Configure simulations and plotting
    configs = [
        SimulationConfig(RLGreedyMethod(epsilon=0.1, step_size=lambda k, a: 1.0/k),
                         plotting_info={"label": "not-so-greedy[0.1,step-size=1/n]", "color": "green"}),
        SimulationConfig(RLGreedyMethod(epsilon=0.1, step_size=lambda k, a: 0.1),
                         plotting_info={"label": "not-so-greedy[0.1,step-size=0.1]", "color": "red"})
    ]

    execute_and_plot(configs, num_runs, time_steps, problems, RLSimulatorNonStationary)