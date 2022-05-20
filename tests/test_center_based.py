"""Test center-based network
"""

import unittest
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import entropy
from multi_agent_ftrl.ftlr import FTRL, tsallis_entropy
from multi_agent_ftrl.adversary import BiasedCoinFlipsAdversary
from multi_agent_ftrl.network_decision import CenterBasedNetworkAgents
from plot.network_plot import draw_center_based_network

hyper_parameters = {
    'network': 'center-based',
    'algo': 'Exp3',
    'rounds': 10000,
    'n_actions': 30,
    'seed_adversary': 0,
    'delay': 1,
    'n_agents': 12,
    'degree': 3,
    'seed_nx': 0,
    'eta': 0.1
}

n_actions = hyper_parameters['n_actions']
biases = np.linspace(0.1, 0.9, n_actions)
rounds = hyper_parameters['rounds']
seed_adversary = hyper_parameters['seed_adversary']
biased_coin_flips_adversary = BiasedCoinFlipsAdversary(n_actions, biases, rounds, seed_adversary)
biased_coin_flips_adversary.assign_losses()
losses = biased_coin_flips_adversary.get_losses()
cum_losses = biased_coin_flips_adversary.get_cum_losses()
least_cum_loss = biased_coin_flips_adversary.get_least_cum_loss_in_hindsight()
best_actions = biased_coin_flips_adversary.get_best_actions_in_hindsight()
best_action = best_actions[-1]

delay = hyper_parameters['delay']
degree = hyper_parameters['degree']
n_agents = hyper_parameters['n_agents']
seed_nx = hyper_parameters['seed_nx']
G = nx.random_regular_graph(degree, n_agents, seed_nx)

n_agents = len(G.nodes())
network_agents = CenterBasedNetworkAgents(G, n_actions)
network_agents.centralize()
network_agents.set_rngs(list(range(n_agents)))



class TestFunctions(unittest.TestCase):

    def test_ftrl(self):
        # set algorithms
        network_agents.set_algos([
            FTRL(n_actions) for _ in range(n_agents)
        ])

        # set up dict of potentials
        potentials = {}
        #lr = np.sqrt(1-np.exp(-1))
        for center in network_agents.comm_net.graph['structure'].keys():
            nbr_len = network_agents.comm_net.nodes[center]['nbr_len']
            #eta = lr * np.sqrt(nbr_len/rounds)
            eta = hyper_parameters['eta']
            potentials[center] = lambda w: -entropy(w) / eta




        probs_hist = []

        for _ in range(rounds):
            # logging
            network_probs = {}
            for v, agent in network_agents.comm_net.nodes.data():
                probs_v = agent['algo'].select_prob()
                network_probs[v] = probs_v
            probs_hist.append(network_probs)
            # learning
            messages = network_agents.selection_step()
            network_agents.feedback_step(messages, biased_coin_flips_adversary)
            network_agents.update_step(potentials, delay)