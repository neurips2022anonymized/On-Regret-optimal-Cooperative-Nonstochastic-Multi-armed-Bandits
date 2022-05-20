"""
Generate the figures in the paper
"""
import os
import json
import sys
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)

import matplotlib.pyplot as plt
import numpy as np
from multi_agent_ftrl.adversary import BiasedCoinFlipsAdversary


def get_least_cum_losses(hyper_parameters):
    """Return the cumulative loss of the best
    action in hindsight

    Args:
        hyper_parameters (dict): specify the setting of one experiment

    Returns:
        1-D nd.array: the cumulative loss
    """
    # Adversary: coin flips
    n_actions = hyper_parameters['n_actions']
    biases = np.linspace(0.1, 0.9, n_actions)
    rounds = hyper_parameters['rounds']
    seed_adversary = hyper_parameters['seed_adversary']
    adversary = BiasedCoinFlipsAdversary(n_actions, biases, rounds, seed_adversary)
    adversary.assign_losses()
    return adversary.get_least_cum_loss_in_hindsight()

def get_averaged_cum_losses(hyper_parameters, path_to_dir):
    """Cumulative loss averaged over all agents

    Args:
        hyper_parameters (dict): specify the setting of one experiment
        path_to_dir (str): directory hosting data

    Returns:
        1-D nd.array: cumulative averaged loss
    """
    algo_file_name = "-".join([
    str(v) for v in hyper_parameters.values()  
    ])
    path_to_algo = path_to_dir+algo_file_name
    # Loading data
    with open(path_to_algo+'.json', 'r') as json_file:
        inst_losses_hist = json.load(json_file)
        inst_losses_averaged = np.average([
            list(dict_hist.values())
            for dict_hist in inst_losses_hist
        ], axis=1)
        cum_losses_averaged = np.cumsum(inst_losses_averaged)
        return cum_losses_averaged

if __name__ == "__main__":
    hyper_parameters = {
        'network': 'center-based',
        'algo': 'exp3',
        'rounds': 10000,
        'n_actions': 50,
        'seed_adversary': 1,
        'delay': 1,
        'n_agents': 12,
        'degree': 3,
        'seed_nx': 0,
        'seed_algo': 1,
        'learning_rate': 0.1
    }
    path_to_dir = "./experiments/coin_flip_regular_graph/"
    cum_losses_averaged = get_averaged_cum_losses(hyper_parameters, path_to_dir)
    least_cum_loss = get_least_cum_losses(hyper_parameters)
    
    plt.figure() 
    plt.plot(np.array(list(range(hyper_parameters['rounds']))), cum_losses_averaged - least_cum_loss, label='averaged')
    plt.ylabel('Cumulative regret')
    plt.xlabel('Rounds')
    plt.grid()
    plt.legend()
    plt.show()
