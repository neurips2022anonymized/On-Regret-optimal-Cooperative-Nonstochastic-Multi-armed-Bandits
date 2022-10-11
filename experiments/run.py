import os
import json
import sys
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import entropy
import progressbar
from multi_agent_ftrl.ftlr import FTRL, tsallis_entropy
from multi_agent_ftrl.adversary import BiasedCoinFlipsAdversary
from multi_agent_ftrl.network_decision import CenterBasedNetworkAgents, NetworkAgents
from plot.network_plot import draw_center_based_network
from generate_graphs import generate_connected_erdos_renyi, quick_erdos_renyi


def run_experiment(G, hyper_parameters, path_to_dir, save_flag=False, show_flag=True):
    """Run the experiment for multi-agent FTRL

    Args:
        G (networkx.Graph): a CONNECTED graph
        hyper_parameters (dict): options for experiment setting
        path_to_dir (str): the directory to save plots and data

    Raises:
        NotImplementedError: if 'network' is not 'center-based' or 'distributed'
    """
    algo_file_name = "-".join([
    str(v) for v in hyper_parameters.values()  
    ])
    path_to_algo = path_to_dir+algo_file_name 

    if os.path.exists(path_to_algo+'.json'):
        print("Algo losses already saved to: " + path_to_algo+'.json.\n')
        return

    # Adversary: coin flips
    n_actions = hyper_parameters['n_actions']
    biases = np.linspace(0.1, 0.9, n_actions)
    rounds = hyper_parameters['rounds']
    seed_adversary = hyper_parameters['seed_adversary']
    biased_coin_flips_adversary = BiasedCoinFlipsAdversary(n_actions, biases, rounds, seed_adversary)
    biased_coin_flips_adversary.assign_losses()
    losses = biased_coin_flips_adversary.get_losses()
    least_cum_loss = biased_coin_flips_adversary.get_least_cum_loss_in_hindsight()

    # Communication network: regular graph
    delay = hyper_parameters['delay']
    n_agents = hyper_parameters['n_agents']
    if hyper_parameters['network'] == 'center-based':
        network_agents = CenterBasedNetworkAgents(G, n_actions)
    elif hyper_parameters['network'] == 'distributed':
        network_agents = NetworkAgents(G, n_actions)
    else:
        raise NotImplementedError
    start = hyper_parameters['seed_algo']
    network_agents.setup(
        [FTRL(n_actions) for _ in range(n_agents)],
        list(range(start, start+n_agents))
    )
    # network_agent_fig = plt.figure()
    # if hyper_parameters['network'] == 'center-based':
    #     draw_center_based_network(network_agents)
    # elif hyper_parameters['network'] == 'distributed':
    #     nx.draw(G, with_labels=True)
    # else:
    #     raise NotImplementedError
    # network_agents_plot_name = "-".join([
    #     hyper_parameters['network'],
    #     str(hyper_parameters['n_agents']),
    #     str(hyper_parameters['degree']),
    #     str(hyper_parameters['seed_nx'])
    # ])
    # if save_flag:
    #     network_agent_fig.savefig(path_to_dir+network_agents_plot_name+".png", dpi=100)

    # Learning
    potentials = network_agents.assign_uniform_potentials(hyper_parameters['algo'], learning_rate=0.1)
    probs_hist = []
    for _ in progressbar.progressbar(
        range(rounds), 
        prefix=hyper_parameters['network']+'|'+hyper_parameters['algo']+'|'+'Time step: '
    ):
        # logging
        network_probs = network_agents.get_action_probs()
        probs_hist.append(network_probs)
        # learning
        network_agents.step(biased_coin_flips_adversary, delay, potentials)

    inst_losses_hist = [
        {
            v:np.dot(probs_hist[t][v], losses[t]) for v in network_agents.comm_net.nodes()
        }
        for t in range(rounds)
    ]
    cum_losses_hist = {
        v: np.cumsum([inst_losses_hist[t][v] for t in range(rounds)]) for v in network_agents.comm_net.nodes()
    }

    if show_flag or save_flag: 
        idv_centers = plt.figure()
        
        if hyper_parameters['network'] == 'center-based':
            for v in network_agents.comm_net.graph['structure']:
                plt.plot(np.array(list(range(rounds))), cum_losses_hist[v] - least_cum_loss, label=str(v)+' (center)')
        elif hyper_parameters['network'] == 'distributed':
            for v in network_agents.comm_net.nodes():
                plt.plot(np.array(list(range(rounds))), cum_losses_hist[v] - least_cum_loss, label=str(v))
        else:
            raise NotImplementedError

        plt.ylabel('Cumulative regret')
        plt.xlabel('Rounds')
        plt.grid()
        plt.legend()
        if show_flag: plt.show()

    if save_flag:
        idv_centers.savefig(path_to_algo+'.png', dpi=100)
        with open(path_to_algo+'.json', 'w+') as fout:
            json.dump(inst_losses_hist, fout)
        print("Algo losses saved to: " + path_to_algo)


def regular_graph():
    path_to_dir = "./experiments/coin_flip_regular_graph/"

    seed_algo_list = list(range(10))
    network_list = ['center-based', 'distributed']
    algo_list = ['tsallis', 'exp3', 'hybrid']
    n_actions_list = [10, 20, 25, 30, 40]
    delay_list = [1]
    degree_list = [2]

    hyper_parameters = {
        'network': None,
        'algo': None,
        'rounds': 10000,
        'n_actions': None,
        'seed_adversary': 1,
        'delay': None,
        'n_agents': 3,
        'degree': None,
        'seed_nx': 0,
        'seed_algo': None,
        'learning_rate': 0.1
    }

    for seed in seed_algo_list:
        for n_actions in n_actions_list:
            for network in network_list:
                for algo in algo_list:
                    for delay in delay_list:
                        for degree in degree_list:
                            hyper_parameters['network'] = network
                            hyper_parameters['algo'] = algo
                            hyper_parameters['n_actions'] = n_actions
                            hyper_parameters['seed_algo'] = seed
                            hyper_parameters['delay'] = delay
                            hyper_parameters['degree'] = degree
                            G = nx.random_regular_graph(
                                hyper_parameters['degree'], 
                                hyper_parameters['n_agents'], 
                                hyper_parameters['seed_nx']
                            )
                            network_agents = hyper_parameters['network']+'-'+hyper_parameters['algo']
                            if not (network_agents == 'distributed-tsallis' or network_agents == 'center-based-hybrid'):
                                run_experiment(G, hyper_parameters, path_to_dir, save_flag=True, show_flag=False)


def path_graph():
    path_to_dir = "./experiments/coin_flip_path_graph/"

    seed_algo_list = list(range(10))
    network_list = ['center-based', 'distributed']
    algo_list = ['tsallis', 'exp3', 'hybrid']
    n_actions_list = [10]
    delay_dist = [1]

    hyper_parameters = {
        'network': None,
        'algo': None,
        'rounds': 1000,
        'n_actions': None,
        'seed_adversary': 1,
        'delay': None,
        'n_agents': 6,
        'seed_algo': None,
        'learning_rate': 0.1
    }

    for seed in seed_algo_list:
        for n_actions in n_actions_list:
            for network in network_list:
                for algo in algo_list:
                    for delay in delay_dist:
                        hyper_parameters['network'] = network
                        hyper_parameters['algo'] = algo
                        hyper_parameters['n_actions'] = n_actions
                        hyper_parameters['seed_algo'] = seed
                        hyper_parameters['delay'] = delay
                        G = nx.path_graph( 
                            hyper_parameters['n_agents']
                        )
                        network_agents = hyper_parameters['network']+'-'+hyper_parameters['algo']
                        if not (network_agents == 'distributed-tsallis' or network_agents == 'center-based-hybrid'):
                            run_experiment(G, hyper_parameters, path_to_dir, save_flag=True, show_flag=False) 


def erdos_renyi_graph():
    path_to_dir = "./experiments/coin_flip_edros_renyi_graph/" 

    seed_algo_list = [0]
    network_list = ['center-based']
    algo_list = ['tsallis', 'exp3']
    n_agents_list = [int(10**expo) for expo in np.linspace(1, 3, 9)]
    type_list = ['sparse', 'dense']
    n_actions_list = [200]

    hyper_parameters = {
        'network': None,
        'algo': None,
        'rounds': 1000,
        'n_actions': None,
        'seed_adversary': 0,
        'delay': 1,
        'n_agents': None,
        'type': None,
        'seed_nx': 0,
        'seed_algo': None,
        'learning_rate': 0.1
    }

    for seed in seed_algo_list:
        for n_actions in n_actions_list:                      
            for n_agents in n_agents_list:
                for tp in type_list:
                    for network in network_list:
                        for algo in algo_list:  
                            hyper_parameters['network'] = network
                            hyper_parameters['algo'] = algo
                            hyper_parameters['n_actions'] = n_actions
                            hyper_parameters['n_agents'] = n_agents
                            hyper_parameters['type'] = tp
                            hyper_parameters['seed_algo'] = seed
                            G = quick_erdos_renyi(
                                n_agents, tp, hyper_parameters['seed_nx']
                            )
                            network_agents = hyper_parameters['network']+'-'+hyper_parameters['algo']
                            if not (network_agents == 'distributed-tsallis' or network_agents == 'center-based-hybrid'):
                                run_experiment(G, hyper_parameters, path_to_dir, save_flag=True, show_flag=False)


def star_graph():
    path_to_dir = "./experiments/coin_flip_star_graph/"

    seed_algo_list = list(range(10))
    network_list = ['center-based', 'distributed']
    algo_list = ['tsallis', 'hybrid']
    n_actions_list = [3]
    delay_dist = list(range(20, 45, 5))

    hyper_parameters = {
        'network': None,
        'algo': None,
        'rounds': 100,
        'n_actions': None,
        'seed_adversary': 1,
        'delay': None,
        'n_agents': 20,
        'seed_algo': None,
        'learning_rate': 0.1
    }

    for seed in seed_algo_list:
        for n_actions in n_actions_list:
            for network in network_list:
                for algo in algo_list:
                    for delay in delay_dist:
                        hyper_parameters['network'] = network
                        hyper_parameters['algo'] = algo
                        hyper_parameters['n_actions'] = n_actions
                        hyper_parameters['seed_algo'] = seed
                        hyper_parameters['delay'] = delay
                        G = nx.star_graph( 
                            hyper_parameters['n_agents']-1
                        )
                        network_agents = hyper_parameters['network']+'-'+hyper_parameters['algo']
                        if not (network_agents == 'distributed-tsallis' or network_agents == 'center-based-hybrid'):
                            run_experiment(G, hyper_parameters, path_to_dir, save_flag=True, show_flag=False) 




if __name__ == "__main__":
    regular_graph()
    # path_graph()
    # star_graph()
    # erdos_renyi_graph()