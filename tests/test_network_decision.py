'''
Testing the multi_agent_ftrl/ma_ftrl.py
'''

import unittest
import numpy as np
import networkx as nx
from multi_agent_ftrl.ftlr import FTRL, tsallis_entropy
from multi_agent_ftrl.adversary import BiasedCoinFlipsAdversary
from multi_agent_ftrl.network_decision import CenterBasedNetworkAgents, NetworkAgents, _add_nbr_len, _get_max_attributes_nodes

class TestFunctions(unittest.TestCase):
    '''
    Test functions and classes in multi_agent_ftrl/ma_ftrl.py
    '''
    def test_add_nbr_attribute(self):
        n_agents = 5
        star = nx.star_graph(n_agents)
        _add_nbr_len(star)
        self.assertEqual(
            [star.nodes[n]['nbr_len'] for n in range(n_agents)],
            [n_agents+1] + [2]*(n_agents-1)
        )

        star = nx.lollipop_graph(4, 6)
        _add_nbr_len(star)
        node, = _get_max_attributes_nodes(star, list(star.nodes()), 'nbr_len')
        self.assertEqual(node, 3)

    def test_NetworkAgents_set_algos(self):
        n_actions = 10
        n_agents = 5
        p = 0.15
        erdos_renyi = nx.erdos_renyi_graph(n_agents, p)
        erdos_renyi_with_ftrl = NetworkAgents(erdos_renyi, n_actions)
        list_of_bandit_ftrls = [
            FTRL(n_actions) for _ in range(n_agents)
        ]
        erdos_renyi_with_ftrl.set_algos(list_of_bandit_ftrls)

        for node, agent in erdos_renyi_with_ftrl.comm_net.nodes.data():
            ftrl = agent['algo']
            res = ftrl.select_prob()
            ans = np.ones(shape=(n_actions,)) / n_actions
            self.assertIsNone(np.testing.assert_array_equal(res, ans))
    
    # Test graph paritioning algorithms

    def test_centers_to_components_star_graph(self):
        # Star graph check
        n_actions = 10
        n_agents = 5
        star = nx.star_graph(n_agents) 
        star_agents = CenterBasedNetworkAgents(star, n_actions)
        
        # compute components of the center
        centers = [0]
        star_agents._centers_to_components(centers)
        non_center_agent = 1
        center_of_non_center = star_agents.comm_net.nodes[non_center_agent]['center']
        self.assertEquals(center_of_non_center, centers[0])
        origin_nbr_non_center = star_agents.comm_net.nodes[non_center_agent]['origin_nbr']
        self.assertEquals(origin_nbr_non_center, centers[0])
        mass_non_center = star_agents.comm_net.nodes[non_center_agent]['mass']
        self.assertAlmostEquals(mass_non_center, np.exp(-1/6)*min(n_agents+1, n_actions))

    def test_centers_to_components_user_defined_graph(self):
        # user defined graph: 
        # https://networkx.org/documentation/stable/auto_examples/basic/plot_simple_graph.html#sphx-glr-auto-examples-basic-plot-simple-graph-py
        G = nx.Graph()
        G.add_edge(1, 2)
        G.add_edge(1, 3)
        G.add_edge(1, 5)
        G.add_edge(2, 3)
        G.add_edge(3, 4)
        G.add_edge(4, 5)

        n_actions = 10
        network_agents = CenterBasedNetworkAgents(G, n_actions)
        centers = [1, 3]
        network_agents._centers_to_components(centers)
        non_center_agent = 4
        center_of_non_center = network_agents.comm_net.nodes[non_center_agent]['center']
        self.assertEquals(center_of_non_center, 3)
        origin_nbr_non_center = network_agents.comm_net.nodes[non_center_agent]['origin_nbr']
        self.assertEquals(origin_nbr_non_center, 3)
        mass_non_center = network_agents.comm_net.nodes[non_center_agent]['mass']
        self.assertAlmostEquals(mass_non_center, np.exp(-1/6)*(4))

    def test_centralize(self):
        degree = 3
        n_agents = 12
        n_actions = 10
        G = nx.random_regular_graph(degree, n_agents)
        net_agents = CenterBasedNetworkAgents(G, n_actions)
        net_agents.centralize()


    # Test collaborative learning algorithms

    def test_selection_step(self):
        # network topology
        G = nx.LCF_graph(12, [3, -3], 3)
        n_agents = len(G.nodes())
        n_actions = 10
        network_agents = CenterBasedNetworkAgents(G, n_actions)
        network_agents.centralize()
        # bandit FTRL algorithms
        network_agents.set_rngs()
        network_agents.set_algos(
            [FTRL(n_actions) for _ in range(n_agents)]
        )
        state_snip = network_agents.selection_step()
        for msg in state_snip.values():
            prob = msg['probs']
            ans = np.ones(shape=(n_actions,)) / n_actions
            self.assertIsNone(np.testing.assert_array_equal(prob, ans))


    def test_feedback_step(self):
        # setting
        n_rounds = 100
        delay = 2
        G = nx.LCF_graph(5, [3, -3], 3)
        n_agents = len(G.nodes())
        n_actions = 10
        # adversary
        biases = [1.0 / k for k in range(2,n_actions+2)]
        adversary = BiasedCoinFlipsAdversary(n_actions, biases, n_rounds, seed=0)
        adversary.assign_losses()
        # set up network topology
        network_agents = CenterBasedNetworkAgents(G, n_actions)
        network_agents.centralize()
        network_agents.set_rngs()
        network_agents.set_algos(
            [FTRL(n_actions) for _ in range(n_agents)]
        )
        # get feedback
        state_snip = network_agents.selection_step()
        network_agents.feedback_step(state_snip, adversary)
        messages = network_agents.msg_memory[0]
        agent = np.random.choice(n_agents)
        action = messages[agent]['action']
        self.assertEquals(
            messages[agent]['loss'],
            adversary.losses[0, action]
        )

    def test_update_step(self):
        # setting
        n_rounds = 100
        delay = 2
        G = nx.Graph()
        G.add_edge('Alice','Bob')
        n_agents = len(G.nodes())
        n_actions = 10
        lr = np.sqrt(1-np.exp(-1))
        # adversary
        biases = [1.0 / k for k in range(2,n_actions+2)]
        adversary = BiasedCoinFlipsAdversary(n_actions, biases, n_rounds, seed=0)
        adversary.assign_losses()
        # set up network topology
        network_agents = CenterBasedNetworkAgents(G, n_actions)
        network_agents.centralize()
        network_agents.set_rngs(list(range(n_agents)))
        network_agents.set_algos(
            [FTRL(n_actions) for _ in range(n_agents)]
        )
        # set up dict of potentials
        potentials = {}
        for center in network_agents.comm_net.graph['structure'].keys():
            nbr_len = network_agents.comm_net.nodes[center]['nbr_len']
            eta = lr * np.sqrt(nbr_len/n_rounds)
            potentials[center] = lambda w: tsallis_entropy(w) / eta
        # collaborative learning starts
        for _ in range(n_rounds):
            messages = network_agents.selection_step()
            network_agents.feedback_step(messages, adversary)
            network_agents.update_step(potentials, delay)

        # check whether the non-center agents are copy from their centers
        non_center = np.random.choice(list(network_agents.comm_net.nodes()))
        while non_center in network_agents.comm_net.graph['structure'].keys():
            non_center = np.random.choice(list(network_agents.comm_net.nodes()))
        time_slice = np.random.choice(n_rounds)
        while time_slice < delay:
            time_slice = np.random.choice(n_rounds)
        
        msg = network_agents.msg_memory[time_slice]
        prob_non_center = msg[non_center]['probs']

        center = network_agents.comm_net.nodes[non_center]['center']
        delayed_msg = network_agents.msg_memory[time_slice - delay-1]
        delayed_prob_center = delayed_msg[center]['probs']
    
        self.assertIsNone(np.testing.assert_array_almost_equal(prob_non_center, delayed_prob_center))
