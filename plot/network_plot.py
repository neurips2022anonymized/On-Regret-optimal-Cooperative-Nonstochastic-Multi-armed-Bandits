"""Visualize the network
"""

import matplotlib
import networkx as nx
import matplotlib.cm as cm

def draw_center_based_network(network_agents):    
    """Visualize a centralized network
        - nodes in the same component share the same color
        - centers has an additional label 'center'
    
    Args:
        - network_agents (network_decision.NetworkAgents): a NetworkAgent object with 
            -- graph attribute: 'structure'
            -- node attribute: 'center'
    """
    G = network_agents.comm_net
    # assign color to nodes
    n_nodes = len(G.nodes())
    norm = matplotlib.colors.Normalize(
        vmin = 0,
        vmax = n_nodes-1,
        clip=True
    )
    mapper = cm.ScalarMappable(norm)
    nodecolors = []
    nodelabels = {}
    for v in G.nodes():
        center = G.nodes[v]['center']
        idx = list(G.nodes()).index(center)
        nodecolors.append(mapper.to_rgba(idx))
        nodelabels[v] = str(v)+' (Center)' if v == center else str(v)
    # label centers
    # draw
    nx.draw(G, node_color=nodecolors, labels=nodelabels)


def get_beta_statistic(G):
    """Beta statistic for a node, defined as

        beta(G) = 1/sqrt(d_1+1) + ... + 1/sqrt(d_N+1)

    Args:
        G (networkx.Graph): a graph

    Returns:
        flaot: beta statistic
    """
    beta_node_list = [
        1/(val+1)**.5 for _, val in G.degree() 
    ]
    return sum(beta_node_list)

