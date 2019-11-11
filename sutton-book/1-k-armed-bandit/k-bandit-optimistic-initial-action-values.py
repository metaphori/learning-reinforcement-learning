import rlutils
from rlutils import *

# See Sutton's book, Section 2.6
# Performance on the 10-armed bandit testbed of a greedy method using Q_1(a) = +5, for all a
# For comparison, also shown is an eps-greedy method with Q_1(a)=0
# Both methods used a constant step-size parameter, alpha = 0.1
# We regard it as
# a simple trick that can be quite effective on stationary problems, but it is far from
# being a generally useful approach to encouraging exploration. For example, it is not
# well suited to nonstationary problems because its drive for exploration is inherently temporary.

if __name__ == '__main__':
    # 1. Generate problems
    num_runs = 2000
    time_steps = 800
    problems: List[Problem] = generate_bandit_problems(10, num_runs)

    optimistic_initial_estimates = ValueEstimates({a: +10.0 for a in problems[0].actions})
    realistic_initial_estimates  = ValueEstimates({a: +0.0 for a in problems[0].actions})

    #rlutils.rldebug = RLDebugger(debug_problems=problems, debug_within=range(998,1000))

    # 2. Configure simulations and plotting
    configs = [
        SimulationConfig(RLGreedyMethod(epsilon=0.0, step_size=lambda k,a: 0.1),
                         optimistic_initial_estimates,
                         plotting_info={"label": "greedy[0.0, optimistic]",
                                        "color": "green",
                                        "graph":Plots.OPTIMAL_ACTION}),
        SimulationConfig(RLGreedyMethod(epsilon=0.1, step_size=lambda k,a: 0.1),
                         realistic_initial_estimates,
                         plotting_info={"label": "near-greedy[0.1, realistic]",
                                        "color": "red",
                                        "graph":Plots.OPTIMAL_ACTION})
    ]

    execute_and_plot(configs, num_runs, time_steps, problems)