'''
Follow-The-Regularized-Leader
'''
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds



class OLO:
    """Online linear optimization

    Returns:
        class object: implements an online
        linear optimization algorithm
    """
    def __init__(self, n_actions):
        """Initialize the cumulative loss and weights
        of each action.

        Args:
            n_actions (int): number of actions
        """
        self.n_actions = n_actions
        self.weights = np.ones(shape=(n_actions,))

    def select_prob(self):
        """Compute the selection probability
        of each action

        Returns:
            a numpy array: selection probabilities
        """
        return self.weights / np.sum(self.weights)


class FTRL(OLO):
    """ Follow-The-Regularized-Leader algorithm
    https://en.wikipedia.org/wiki/Online_machine_learning#Follow_the_regularised_leader_(FTRL)
    """
    def __init__(self, n_actions):
        super().__init__(n_actions)
        self.cum_losses = np.zeros(shape=(n_actions,))

    def step(self, loss_vec, potential=None):
        """An update step:
        optimize a linear function + potential function

        Args:
            loss_vec (numpy array): the instant
            loss of each action
        """
        self.cum_losses += loss_vec
        bounds = Bounds([0]*self.n_actions, [1.0]*self.n_actions)
        con = lambda w: np.sum(w) - 1
        obj = lambda w: np.dot(w, self.cum_losses) + potential(w)
        res = minimize(
            obj,
            x0=self.select_prob(),
            constraints={'type': 'eq', 'fun': con},
            bounds=bounds)
        self.weights = res.x


class Hedge(OLO):
    """Hedge algorithm
    https://en.wikipedia.org/wiki/Multiplicative_weight_update_method
    """
    def step(self, loss_vec, learning_rate=1e-3):
        """An update step:
        multiplicative weight update

        Args:
            loss_vec (numpy array): the instant
            loss of each action
            learning_rate (float, optional): Defaults to 1e-3.
        """
        probs = self.weights / np.sum(self.weights)
        self.weights = probs * np.exp(-learning_rate*loss_vec)


class BanditFTRL(FTRL):
    """FTRL algorithm with bandit feedback
    """
    def update(self, loss, chosen_action, potential=None):
        """An update step:
            loss estimation + ftrl
        Args:
            loss (float): the loss of the chosen action
            chosen_action (int): id of the chosen action
            potential (function, optional): learning_rate * regularizer. Defaults to None.
        """
        probs = self.select_prob()
        loss_estimators = importance_weighted_estimation(probs, loss, chosen_action)
        self.step(loss_estimators, potential)


class Exp3(Hedge):
    """Exp3 algorithm
    https://en.wikipedia.org/wiki/Multi-armed_bandit#Exp3[60]
    """
    def update(self, loss, chosen_action, learning_rate=0.001):
        """An update step:
        multiplicative weight update

        Args:
            loss (float): the loss of chosen action
            chosen_action (int): the chosen action. Defaults to 0.
            learning_rate (float, optional): Defaults to 1e-3.
        """
        probs = self.select_prob()
        loss_estimators = importance_weighted_estimation(probs, loss, chosen_action)
        self.step(loss_estimators, learning_rate)



def importance_weighted_estimation(probs, loss, action):
    """Importance-weighted loss estimator

    Args:
        probs (1-D numpy array): the probability of choosing each action
        loss (float): the loss associated with the chosen action
        action (int): id of chosen action

    Returns:
        1-D numpy: the loss estimator
    """
    n_actions = probs.size
    loss_vec = np.zeros(n_actions)
    loss_vec[action] = loss / probs[action]
    return loss_vec

def tsallis_entropy(weights):
    '''
    Wiki: https://en.wikipedia.org/wiki/Tsallis_distribution
    '''
    return -2 * np.sum(np.sqrt(weights))
