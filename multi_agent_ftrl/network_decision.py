"""
Multi-Agent Follow-The-Regularized-Leader

    - NetworkAgents
    - CenterBasedNetworkAgents
    - DistributedNetworkAgents
"""

import numpy as np
import networkx as nx
from scipy.stats import entropy
from multi_agent_ftrl.ftlr import FTRL, tsallis_entropy

class NetworkAgents:
    """A set of agents communicate through a CONNECTED network
    to solve a common decision making problems with 
    K actions.

    Each agent is a node in the graph with attributes:
        - 'nbr_len': the number of numbers plus 1
        - 'algo': the algorithm associated with the agent
        - 'rng': the random number generator
        - 'clock_counter': the clock

    The network_agents object will interact with the adversary:
        - selection_step()
        - feedback_step()
        - update_step()
    """
    def __init__(self, comm_net, n_actions):
        """ 1) Set up the communication network;
            2) add size of the neighborhood
            3) add clock counter (starts from 0)
        as attributes to each node in the communication network;
            3) add the memory for the messages
        as attributes to the graph.
        Args:
            comm_net (networx.Graph): a *CONNECTED* communication network.
            n_actions (int): the number of actions
        """
        self.comm_net = comm_net
        self.n_actions = n_actions
        self.msg_memory = [] # a list of messages
        # add size of the neighborhood
        _add_nbr_len(self.comm_net)
        # add clock counter
        for v in self.comm_net.nodes():
            nx.set_node_attributes(
                self.comm_net,
                values={v: 0},
                name='clock_counter'
            )
    
    def set_algos(self, list_of_algos):
        """Set up the algorithm for each agent/node

        Args:
            list_of_algos (list): a list of algorithms, e.g. ftrl.OLO

        Raises:
            ValueError: if n_algos not equals n_agents
        """
        n_agents = nx.number_of_nodes(self.comm_net)
        n_algos = len(list_of_algos)
        if not n_agents == n_algos:
            raise ValueError('The number of agents and algorithms do not match.')
        # add algorithm attributes
        for node, algo in zip(self.comm_net.nodes(), list_of_algos):
            nx.set_node_attributes(
                self.comm_net,
                values={node: algo},
                name='algo'
            )

    def set_rngs(self, list_of_seeds=None):
        """Set up the random number generator for each agent/node

        Args:
            list_of_seeds (list, optional): a list of seeds. Defaults to None.

        Raises:
            ValueError: if n_seeds not equals n_agents
        """
        if list_of_seeds: # list_of_seeds is a listp
            n_agents = nx.number_of_nodes(self.comm_net)
            n_seeds = len(list_of_seeds)
            if not n_agents == n_seeds:
                raise ValueError('The number of agents and seeds do not match.')
            # add random seed attributes
            for node, seed in zip(self.comm_net.nodes(), list_of_seeds):
                nx.set_node_attributes(
                    self.comm_net,
                    values={node: np.random.default_rng(seed)},
                    name='rng'
                )
        else:
            # set seeds to be None
            for node in self.comm_net.nodes():
                nx.set_node_attributes(
                    self.comm_net,
                    values={node: np.random.default_rng()},
                    name='rng'
                )

    def selection_step(self):
        """Each agent in the network select its action

        Returns:
            dict: {v: {'time_step': time_step, 'action': action, 'probs': probs}}
        """
        messages = {}
        for v, agent in self.comm_net.nodes.data():
            # t
            time_step = agent['clock_counter']
            # p_t^v
            algo = agent['algo']
            selection_prob = algo.select_prob()
            # I_t(v)
            rng = agent['rng']
            action = rng.choice(self.n_actions, p=selection_prob)
            
            messages[v] = {
                'time_step': time_step, 
                'action': action, 
                'probs': selection_prob
            }
        return messages

    def feedback_step(self, messages, adversary):
        """The adversary assigns losses for the chosen actions
        and the system save the messages to the msg_memory
        e.g.
            msg_memory = [ {v: {'time_step': t, 
                                'action': I_t(v),
                                'loss': l_t(I_t(v)) 
                                'probs': p_t^v}} ]
        Args:
            messages (dict): key: node, value: message
            adversary (adversary.Adversary): a adversary object
        """
        for v, state in messages.items():
            action = state['action']
            time_step = state['time_step']
            messages[v]['loss'] = adversary.losses[time_step, action]
        # save the message to the memory
        self.msg_memory.append(messages)

    def get_action_probs(self):
        """Get the action selection probabilities of each agents

        Returns:
            dict: {node: probs}
        """
        action_probs = {}
        for v, agent in self.comm_net.nodes.data():
            algo = agent['algo']
            selection_prob = algo.select_prob()
            action_probs[v] = selection_prob
        return action_probs

    def assign_uniform_potentials(self, regularizer, learning_rate=0.1):
        """Set the same potential for all agents

        Args:
            regularizer (string): 'tsallis'/'exp3'
            learning_rate (float): Defaults to 0.1

        Return:
            potentials (dict): potentials[node] is the potential func
            of node.

        Raises:
            NotImplementedError: only implement 'tsallis' and 'exp3'
        """
        if regularizer == 'tsallis':
            potential = lambda w: tsallis_entropy(w) / learning_rate
        elif regularizer == 'exp3':
            potential = lambda w: -entropy(w) / learning_rate
        elif regularizer == 'hybrid':
            K = self.n_actions
            potential = lambda w, t: tsallis_entropy(w) / learning_rate  -entropy(w) / np.sqrt(np.log(K)/(t+1))
        else:
            raise NotImplementedError
        
        potentials = {'__regularizer__': regularizer}
        for node in self.comm_net.nodes():
            potentials[node] = potential
        return potentials

    def _neighbor_aggregated_loss_estimator(self, node, delay):
        """neighbors-aggregated loss estimator:
            loss estimator = loss/probability of at least one choosing it

        Args:
            node (network.Graph.nodes): node in the graph
            delay (int): the delay in the communication per edge

        Returns:
            np.array: the observable loss estimators
        """
        obs_loss_vec = np.zeros((self.n_actions,))
        clock = self.comm_net.nodes[node]['clock_counter']
        if clock >= delay: 
            # only delayed losses are observable
            messages = self.msg_memory[clock-delay]
            aggregated_msg = {}
            for v, msg in messages.items():
                if v in self.comm_net[node] or v == node:
                    action_of_v = msg['action']
                    if action_of_v not in aggregated_msg:
                        aggregated_msg[action_of_v] = {
                            'loss': msg['loss'],
                            'prob': msg['probs'][action_of_v]
                        }
                    else:
                        prob_not_choose = 1- aggregated_msg[action_of_v]['prob']
                        prob_not_choose *= (1-msg['probs'][action_of_v])
                        aggregated_msg[action_of_v]['prob'] = 1-prob_not_choose
            for action_of_nbr, feedback in aggregated_msg.items():
                    obs_loss_vec[action_of_nbr] = feedback['loss'] / feedback['prob']
        return obs_loss_vec
    
    def setup(self, list_of_algos, list_of_seeds=None):
        """Set up random number generators and algorithms

        Args:
            list_of_algos (list): list of FTRL algorithms
            list_of_seeds (list, optional): list of random seeds. Defaults to None.
        """
        self.set_rngs(list_of_seeds)
        self.set_algos(list_of_algos)

    def update_step(self, potentials, delay=1):
        """Agents update their probabilities from the feedback
        of their neighbors

        Args:
            potentials (dict): specify the potentials for each agent
            delay (int, optional): the delay. Defaults to 1.
        """
        for node, agent in self.comm_net.nodes.data():
            obs_loss_vec = self._neighbor_aggregated_loss_estimator(node, delay)
            ftrl = agent['algo']
            if potentials['__regularizer__'] == 'hybrid':
                current_time = agent['clock_counter']
                current_potential = lambda w: potentials[node](w, current_time)
                ftrl.step(obs_loss_vec, current_potential)
            else:
                ftrl.step(obs_loss_vec, potentials[node])
            agent['clock_counter'] += 1

    def step(self, adversary, delay, potentials):
        """One step in the cooperative learning
            - select arms
            - receive feedbacks
            - update the algorithms
        Args:
            adversary (multi_agent_ftrl.adversary.Adversary): the adversary
            delay (int): the delay in the communication network
            potentials (func): the lambda function
        """
        messages = self.selection_step()
        self.feedback_step(messages, adversary)
        self.update_step(potentials, delay)


class CenterBasedNetworkAgents(NetworkAgents):
    """Center-based network:
        Bar-On and Mansor - 2019 - Individual Regret in Cooperative Nonstochastic Multi-Armed Bandits

    Each agent is a node in the graph with attributes:
        - 'nbr_len': the number of numbers plus 1;
        - 'algo': the algorithm associated with the agent;
        - 'center': the center of the agent
        - 'origin_nbr': the origin neighbor of the agent;
        - 'mass': the mass of the agent 
    """
    def _centers_to_components(self, centers):
        """Compute the components of each center
            - Algorithm 3

        Args:
            centers (list): list of nodes in the graph
        """
        n_iters = int(12*np.log(self.n_actions))
        # Initialization
        for node in self.comm_net.nodes():
            if node in centers:
                initializer = node
                mass = min(self.comm_net.nodes[node]['nbr_len'], self.n_actions)
            else:
                initializer = None
                mass = 0
            # initialize C_0(v)
            nx.set_node_attributes(
                self.comm_net,
                values={node: initializer},
                name='center'
            )
            # initialize U_0(v)
            nx.set_node_attributes(
                self.comm_net,
                values={node: initializer},
                name='origin_nbr'
            )
            # # initialize M_0(v)
            nx.set_node_attributes(
                self.comm_net,
                values={node:mass},
                name='mass'
            )
        # Iteration starts
        for iter in range(n_iters):
            for v in self.comm_net.nodes():
                if self.comm_net.nodes[v]['origin_nbr'] not in centers:
                    # find new origin neighbor
                    new_origin_nbr, = _get_max_attributes_nodes(
                        self.comm_net,
                        list(self.comm_net[v]),
                        'mass'
                    )
                    self.comm_net.nodes[v]['origin_nbr'] = new_origin_nbr
                    # update new center
                    new_center = self.comm_net.nodes[new_origin_nbr]['center']
                    self.comm_net.nodes[v]['center'] = new_center
                    # update new mass
                    new_mass = (
                        0 if new_center is None else  np.exp(-1/6) * self.comm_net.nodes[new_center]['mass']
                    ) 
                    self.comm_net.nodes[v]['mass'] = new_mass
                # else: keep old values

    def _computer_centers_informed(self):
        """Compute the centers in the informed setting.
            - Algorithm 4
        """
        list_of_centers = []
        unsatisfied_agents = list(self.comm_net.nodes())
        while unsatisfied_agents: # set is not empty
            next_center, = _get_max_attributes_nodes(self.comm_net, unsatisfied_agents, 'nbr_len')
            list_of_centers.append(next_center)
            self._centers_to_components(list_of_centers)
            unsatisfied_agents = []
            for v in self.comm_net.nodes():
                pseudo_mass = min(self.comm_net.nodes[v]['nbr_len'], self.n_actions)
                if self.comm_net.nodes[v]['mass'] < pseudo_mass:
                    dists_to_centers = []
                    for center in list_of_centers:
                        dist_v_c = nx.shortest_path_length(self.comm_net, v, center)
                        dists_to_centers.append(dist_v_c)
                    dist_to_nearest_center = min(dists_to_centers)
                    if dist_to_nearest_center >= 3:
                        unsatisfied_agents.append(v)
        return list_of_centers

    def centralize(self, dict_out=False):
        """Compute the centers and their components
        Args:
            - dict_out (bool, optional): return the network structure or not. Defaults to False.
        Return:
            - add as a graph attribute the dictionary of {center: list of component nodes} 
        """
        centers = self._computer_centers_informed()
        self._centers_to_components(centers)
        net_structure = {}
        for v in self.comm_net.nodes():
            center = self.comm_net.nodes[v]['center']
            if center in net_structure.keys():
                net_structure[center].append(v)
            else:
                net_structure[center] = [v]
        self.comm_net.graph['structure'] = net_structure
        if dict_out: return net_structure

    def update_step(self, potentials, delay=1):
        """Centers and non-center agents update their probabilities

        Args:
            potentials (dicts): specify the potentials of each node.
            delay (int, optional): the delay. Defaults to 1.
        """
        for center, components in self.comm_net.graph['structure'].items():
            # center agents update
            obs_loss_vec = self._neighbor_aggregated_loss_estimator(center, delay)
            ftrl = self.comm_net.nodes[center]['algo']
            if potentials['__regularizer__'] == 'hybrid':
                current_time = self.comm_net.nodes[center]['clock_counter']
                current_potential = lambda w: potentials[center](w, current_time)
                ftrl.step(obs_loss_vec, current_potential)
            else:
                ftrl.step(obs_loss_vec, potentials[center])
            self.comm_net.nodes[center]['clock_counter'] += 1
            # non-center agent update
            for non_center in components:
                if not non_center == center:
                    current_time = self.comm_net.nodes[non_center]['clock_counter']
                    if current_time >= delay: # update only when messages were received
                        msg_delayed = self.msg_memory[current_time-delay]
                        ftrl_non_center = self.comm_net.nodes[non_center]['algo']
                        origin_nbr = self.comm_net.nodes[non_center]['origin_nbr']
                        ftrl_non_center.weights = msg_delayed[origin_nbr]['probs']
                    self.comm_net.nodes[non_center]['clock_counter'] += 1

    def setup(self, list_of_algos, list_of_seeds=None):
        """Centralize the network first and set up rngs and algos

        Args:
            list_of_algos (list): list of FTRL algorithms
            list_of_seeds (list, optional): list of random seeds. Defaults to None.
        """
        self.centralize()
        super().setup(list_of_algos, list_of_seeds)


def _add_nbr_len(graph):
    """Add size of neighborhood set as attribute to the nodes

    Usage: graph.nodes[node]['nbr_len']

    Args:
        graph (networkx.Graph): a graph object
    """
    for node, nbrsdict in graph.adj.items():
        nbrsdict_len = len(nbrsdict)
        nx.set_node_attributes(
            graph,
            values={node: nbrsdict_len+1},
            name='nbr_len'
        )


def _get_max_attributes_nodes(graph_with_attrs, list_of_nodes_in_graph, *attributes):
    """get the node with the maximum attribute from
    a list of nodes in the graph

    Args:
        graph_with_attr (network.Graph): the underlying graph with the compared attributes
        list_of_nodes_in_graph (list): list of nodes in the graph
        attributes (string): keys of node attributes
    """
    max_nodes = []
    for attr in attributes:
        list_of_attr_values = [
            graph_with_attrs.nodes[n][attr] for n in list_of_nodes_in_graph
        ]
        idx = np.argmax(list_of_attr_values)
        max_nodes.append(list_of_nodes_in_graph[idx])
    return tuple(max_nodes)
