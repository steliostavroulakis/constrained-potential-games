import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import networkx as nx

plt.style.use('ggplot')

from lib import Player
from lib import create_dag
from lib import update_lambda
from lib import gradient_descent
from lib import calculate_nash_gap
from lib import plot_distributions
from lib import plot_graph

#run = wandb.init(project="Constrained Games")

G = create_dag()
players = dict()

t_iterates = 800
x_stepsize = 0.0001

regularizer = 0.01
hw_cong_mult = 0.1

player_gas_constraints = {
    'Red': 6,
    'Yellow': 3,
    'Green': 9,
    'Blue': 5,
    'Orange': 4
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

    constr_list.append(constr_sum)
    l_list.append(l_sum)
    gaps.append(calculate_nash_gap(G, players, hw_cong_mult))

plot_graph(gaps, 'Sum of nash_gap', 'Sum of nash_gap over time', 'nash_gap')
plot_graph(constr_list, 'Constraint Function', 'Constraint Function Over Time', 'constr_violation')
plot_graph(l_list, 'Player.l', 'Player.l Over Time', 'lambdas')

fig, axs = plt.subplots(len(players), 1, figsize=(8, 15))
for idx, (player_name, player) in enumerate(players.items()):
    axs[idx].bar(range(len(player.strategy)), player.strategy)
    axs[idx].set_title(f'{player_name}')
    axs[idx].set_xlabel('Strategy')
    axs[idx].set_ylabel('Probability')
    axs[idx].set_ylim([0, 1])
plt.tight_layout()
plt.savefig('images/strategy_final.png')
plt.clf()

plot_distributions(players)

#wandb.agent(sweep_id, function=train)
