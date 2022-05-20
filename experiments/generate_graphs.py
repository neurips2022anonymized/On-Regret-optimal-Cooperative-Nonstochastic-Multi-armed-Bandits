import networkx as nx
import numpy as np

def generate_connected_erdos_renyi(n_agents, p_connect, seed):
    """Generate a connected Erdos-Renyi graph

    Args:
        n_agents (int): number of nodes
        p_connect (float): the connectivity probability
        seed (int): random seed

    Raises:
        ValueError: raise when the connectivity condition is violated

    Returns:
        nx.Graph: a connected Erdos-Renyi graph
    """
    threshold = np.log(n_agents)/n_agents
    if p_connect < threshold:
        raise ValueError('p_connect < the connectivity threshold: {}'.format(threshold))
    while True:
        G = nx.erdos_renyi_graph(n_agents, p_connect, seed)
        if nx.is_connected(G):
            return G
        seed += 1

def generate_sparse_erdos_renyi_graph(n_agents, seed):
    """Expected number of edges: O(N\log N)

    Args:
        n_agents (int): number of agents
        seed (int): random seed

    Returns:
        nx.Graph: a erdos_renyi graph
    """
    p_connect = 2*np.log(n_agents)/n_agents
    return generate_connected_erdos_renyi(n_agents, p_connect, seed)

def generate_dense_erdos_renyi_graph(n_agents, seed):
    """Expected number of edges: O(N^2/2)

    Args:
        n_agents (int): number of nodes
        seed (int): random seed

    Returns:
        nx.Graph: an erdos renyi graph
    """
    return generate_connected_erdos_renyi(n_agents, 0.5, seed)

def quick_erdos_renyi(n_agents, tp, seed):
    """Generate different types of Erdos-Renyi graph

    Args:
        n_agents (int): number of nodes
        tp (string): type of the graph
        seed (int): random seed

    Raises:
        NotImplementedError: only supports 'sparse'/'dense'

    Returns:
        nx.Graph: the desired Erdos Renyi graph
    """
    if tp == 'sparse':
        return generate_sparse_erdos_renyi_graph(n_agents, seed)
    elif tp == 'dense':
        return generate_dense_erdos_renyi_graph(n_agents, seed)
    else:
        raise NotImplementedError



if __name__ == "__main__":
    n_agents = 10
    p_connect = 0.5
    seed = 0
    G = generate_connected_erdos_renyi(n_agents, p_connect, seed)
    nx.draw(G)
    print(seed)