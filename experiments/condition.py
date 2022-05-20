import numpy as np

def quick_est(max_degree, start=10):
    """Quick estimation for the number of arms and time steps needed.

    Args:
        max_degree (int): the maximum degree of the nodes
        start (int, optional): _description_. Defaults to 10.

    Returns:
        tuple: the number of arms and the number of time steps.
    """
    K=start
    while K/np.log(K) < max_degree**2:
        K += 1
    K = max(K, max_degree)
    T = max(2*K*max_degree, K**2*np.log(K))
    return int(K)+1, int(T)+1

if __name__ == "__main__":
    # For Erdos-Renyi graph, degree ~ log(num_nodes)
    N = 1000
    degree = np.log(N)
    K, T = quick_est(degree)
    print("The number of arms: K >= {0}\nThe number of time steps: T >= {1}\n".format(K, T))