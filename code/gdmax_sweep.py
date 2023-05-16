import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import networkx as nx

import wandb

plt.style.use('ggplot')

from lib import Player
from lib import create_dag
from lib import update_lambda
from lib import gradient_descent
from lib import calculate_nash_gap
from lib import plot_distributions
from lib import plot_graph

sweep_config = {
    "method": "random",
    "metric": {
        "name": "Nash Gap",  
        "goal": "minimize"  
    },
    "parameters": {
        "regularizer": {
            "values": [0.005, 0.01, 0.02]
        },
        "strategy_stepsize": {
            "values": [0.001, 0.002, 0.005]
        },
        "gas": {  # replace with your parameter
            "values": [2, 4, 6, 8, 10]
        },
        "hw_cong_mult": {  # replace with your parameter
            "values": [0.05, 0.1, 0.2]
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project="Constrained Games")  # replace with your project name

def train():

    run = wandb.init()
    
    # Configuration (same as the sweep config parameters)
    config = run.config

    G = create_dag()
    players = dict()

    t_iterates = 800
    x_stepsize = config.strategy_stepsize

    regularizer = config.regularizer
    hw_cong_mult = config.hw_cong_mult

    player_gas_constraints = {
        'Red': config.gas,
        'Yellow': config.gas,
        'Green': config.gas,
        'Blue': config.gas,
        'Orange': config.gas
    }

    for color, gas_constraint in player_gas_constraints.items():
        players[color] = Player(color, G, gas_constraint)

    constr_list, l_list, gaps = [], [], []

    for t in range(t_iterates):

        if t % 100 == 0:
            print(f"Starting Iteration {t}/{t_iterates}")

        # Find optimal lambda for iteration t
        update_lambda(players, regularizer)

        # Run a gradient descent step on Psi
        gradient_descent(G, players, x_stepsize,hw_cong_mult)

        constr_sum = 0
        l_sum = 0

        for player in players.values():
            # Calculate expected gas consumption and constraint violation
            expected_consumption = sum(x * y for x, y in zip(player.strategy, player.path_lengths))
            constraint_value = expected_consumption - player.gas
            if constraint_value < 0:
                constraint_value = 0

            # Sum up the constraint_function and player.l values
            constr_sum += constraint_value
            l_sum += player.l

        wandb.log({ "Sum of Constraint Violations": constr_sum,
                    "Sum of Lagrangian Multipliers": l_sum,
                    "Nash Gap": calculate_nash_gap(G, players, config.hw_cong_mult)
                    })

        #constr_list.append(constr_sum)
        #l_list.append(l_sum)
        #gaps.append(calculate_nash_gap(G, players, hw_cong_mult))

    #plot_graph(gaps, 'Sum of nash_gap', 'Sum of nash_gap over time', 'nash_gap')
    #plot_graph(constr_list, 'Constraint Function', 'Constraint Function Over Time', 'constr_violation')
    #plot_graph(l_list, 'Player.l', 'Player.l Over Time', 'lambdas')

    # fig, axs = plt.subplots(len(players), 1, figsize=(8, 15))
    # for idx, (player_name, player) in enumerate(players.items()):
    #     axs[idx].bar(range(len(player.strategy)), player.strategy)
    #     axs[idx].set_title(f'{player_name}')
    #     axs[idx].set_xlabel('Strategy')
    #     axs[idx].set_ylabel('Probability')
    #     axs[idx].set_ylim([0, 1])
    # plt.tight_layout()
    # #wandb.log({"Final Strategy Profile": wandb.Image(fig)})
    # #plt.savefig('images/strategy_final.png')
    # plt.clf()

    labels = np.array(['Road 1', 'Road 2', 'Road 3', 'Highway'])
    num_vars = len(labels)

    # Compute angle each bar is centered on:
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    labels = np.concatenate((labels,[labels[0]]))
    angles += angles[:1]

    # Size of the figure
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Draw one axe per variable and add labels
    plt.xticks(angles, labels, color='grey', size=12)

    # Draw ylabels, I chose to only display the max value
    ax.set_rlabel_position(30)
    plt.yticks([1], ['1'], color='grey', size=10)
    plt.ylim(0, 1)

    # Color dictionary for player names to match line color
    color_dict = {'Yellow': 'yellow', 'Red': 'red', 'Green': 'green', 'Blue': 'blue', 'Orange': 'orange'}

    for player in players.values():
        # The plot is a circle, so we need to "complete the loop"
        # and append the start value to the end.
        strategy = np.concatenate((player.strategy,[player.strategy[0]]))
        
        ax.plot(angles, strategy, linewidth=1, linestyle='solid', label=player.name, color=color_dict[player.name])
        ax.fill(angles, strategy, color=color_dict[player.name], alpha=0.05)

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title("Final distributions over paths")
    wandb.log({"Final Strategy Profiles": wandb.Image(fig)})
    #plt.savefig('images/spider_chart.png')
    plt.close()

    run.finish()

wandb.agent(sweep_id, function=train)
