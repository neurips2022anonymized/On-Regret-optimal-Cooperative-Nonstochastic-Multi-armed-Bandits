'''
A class to model adversarial

Example:
    random_adversary = RandomAdversary(n_actions, rounds, random_seed)
    random_adversary.assign_losses()
    cum_losses = random_adversary.get_cum_losses()
'''

from abc import ABCMeta, abstractmethod
from matplotlib.pyplot import axis
import numpy as np

class Adversary(metaclass=ABCMeta):
    """Base class
    [
        [l_{1,1}, ..., l_{1, n_actions}],
        [l_{2,1}, ..., l_{2, n_actions}],
        ...,
        [l_{rounds,1}, ..., l_{rounds, n_actions}],
    ]
    """
    def __init__(self, n_actions, rounds, seed=0):
        """Initialize loss matrix

        Args:
            n_actions (int): the number of actions
            rounds (int): the horizon of game
        """
        self.n_actions = n_actions
        self.rounds = rounds
        self.losses = np.zeros((rounds, n_actions))
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def assign_losses(self, *args, **kwargs):
        """Generate the losses for each action 
        in each time step
        """

    def get_losses(self):
        return self.losses

    def get_cum_losses(self):
        """Cumulative losses

        Returns:
            1-D numpy array: the cumulative losses associated
            with each action
        """
        return np.cumsum(self.losses, axis=0)

    def get_best_actions_in_hindsight(self):
        """Best action in hindsight for each time step

        Returns:
            list: [the id of action] * rounds
        """
        return np.argmin(self.get_cum_losses(), axis=1)

    def get_least_cum_loss_in_hindsight(self):
        """The cumulative loss of the best action in hindsight
        for each time step

        Returns:
            1-D numpy array: np.array([cumulative loss] * rounds)
        """
        return np.min(self.get_cum_losses(), axis=1)


class RandomAdversary(Adversary):
    """Randomly assign the losses in [0, 1]
    """
    def assign_losses(self):
        """all actions has 0.5 loss mean
        """
        self.losses = self.rng.random((self.rounds, self.n_actions))


class OnePeakActionAdversary(Adversary):
    """Each action is associated with a fixed binomial distribution.
    At each time step, the adversary assigns losses by sampling from
    those distributions independently.
        There exists a single optimal action, index at 0
    """
    def __init__(self, n_actions, rounds, delta, seed=0):
        """Args:
            delta (float): 0 < delta < 0.5
        """
        super().__init__(n_actions, rounds, seed)
        self.delta = delta

    def assign_losses(self):
        """mean vector of the losses associated with each action
           [(1-delta)/2, (1+delta)/2, (1+delta)/2, ..., (1+delta)/2]
        """
        self.losses[:,0] = self.rng.binomial(1,(1-self.delta)/2,size=(self.rounds,))
        self.losses[:,1:] = self.rng.binomial(1,
                (1+self.delta)/2,
                size=(self.rounds,self.n_actions-1))


class BiasedCoinFlipsAdversary(Adversary):
    """The adversary will assign the biases for a set of coins,
    the agent receive the a loss of 1 if the coin flip is heads and 
    0 otherwise.
    The bias is the probability of getting a head, hence of receiving 
    a loss of 1.
    """
    def __init__(self, n_actions, biases, rounds, seed=0):
        """
        Args:
            n_actions (int): number of actions
            biases (1D numpy array): the array of bias
            rounds (int): horizon
            seed (int, optional): random seed of sampling. Defaults to 0.
        """
        super().__init__(n_actions, rounds, seed)
        self.biases = biases
    
    def assign_losses(self):
        for time_step in range(self.rounds):
            self.losses[time_step] = [1 if self.rng.random() < bias else 0 \
                                     for bias in self.biases]
            