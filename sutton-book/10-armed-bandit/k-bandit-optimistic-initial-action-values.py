from rlutils import *

# See Sutton's book, Section 2.6
# Performance on the 10-armed bandit testbed of a greedy method using Q_1(a) = +5, for all a
# For comparison, also shown is an eps-greedy method with Q_1(a)=0
# Both methods used a constant step-size parameter, alpha = 0.1

if __name__ == '__main__':
    # 1. Generate problems
    num_runs = 2000
    time_steps = 200
    problems: List[Problem] = generate_bandit_problems(10, num_runs)

    optimistic_initial_estimates = ValueEstimates({a: 5.0 for a in problems[0].actions})

    # 2. Configure simulations and plotting
    configs = [
        SimulationConfig(RLGreedyMethod(epsilon=0.0, step_size=lambda k,a: 0.1),
                         optimistic_initial_estimates,
                         plotting_info={"label": "greedy[0.0, optimistic]", "color": "green", "graph":Plots.OPTIMAL_ACTION}),
        SimulationConfig(RLGreedyMethod(epsilon=0.1, step_size=lambda k,a: 0.1),
                         plotting_info={"label": "greedy[0.1]", "color": "red", "graph":Plots.OPTIMAL_ACTION})
    ]

    execute_and_plot(configs, num_runs, time_steps, problems, RLSimulatorNonStationary)