'''
Testing the multi_agent_ftrl/adversary.py
'''
import unittest
import numpy as np
from multi_agent_ftrl.adversary import BiasedCoinFlipsAdversary, RandomAdversary, OnePeakActionAdversary


class TestFunctions(unittest.TestCase):
    '''
    Test classes in multi_agent_ftrl/adversary.py
    '''
    def test_random_adversary(self):
        """Test:
            - random seed
            - correct output shape
            - non-decreasing of the cumulative loss
            - least cumulative loss
        """
        n_actions = 3
        rounds = 100
        seed = 12
        random_adversary = RandomAdversary(n_actions, rounds, seed)
        random_adversary.assign_losses()
        cum_losses = random_adversary.get_cum_losses()
        best_actions = random_adversary.get_best_actions_in_hindsight()
        # output shape
        self.assertEqual(
            cum_losses.shape,
            (rounds,n_actions)
        )
        self.assertEqual(
            best_actions.shape,
            (rounds,)
        )
        # non-decreasing
        for action in range(n_actions):
            cum_loss = cum_losses[:,action]
            self.assertTrue(
                all(x<=y for x, y in zip(cum_loss, cum_loss[1:]))
            )
        # test least cumulative loss
        least_cum_losses = random_adversary.get_least_cum_loss_in_hindsight()
        self.assertIsNone(np.testing.assert_array_almost_equal(
            least_cum_losses,
            np.array([cum_losses[t, best_actions[t]] for t in range(rounds)])
        ))

    def test_one_peak_action_adversary(self):
        """Test:
            - best action
            - empirical mean = true mean
        """
        n_actions = 3
        rounds = 1000
        delta = 0.1
        seed = 10
        one_peak_action_adversary = OnePeakActionAdversary(n_actions, rounds, delta, seed)
        one_peak_action_adversary.assign_losses()
        cum_losses = one_peak_action_adversary.get_cum_losses()
        best_actions = one_peak_action_adversary.get_best_actions_in_hindsight()
        least_cum_losses = one_peak_action_adversary.get_least_cum_loss_in_hindsight()
        # best action test
        self.assertEqual(
            best_actions[-1],
            0
        )
        # test mean
        mean = np.array([1-delta] + [1+delta]*(n_actions-1)) / 2
        self.assertIsNone(np.testing.assert_array_almost_equal(
            cum_losses[-1,:]/rounds,
            mean,
            decimal=1
        ))
        # test least mean
        self.assertIsNone(np.testing.assert_array_almost_equal(
            least_cum_losses[-3] / rounds,
            mean[0],
            decimal=2
        ))

    
    def test_biased_coin_flips_adversary(self):
        n_coins = 10
        biases = [1.0 / k for k in range(2,12)]
        rounds = 10000
        seed = 0
        biased_coin_flips_adversary = BiasedCoinFlipsAdversary(n_coins, biases, rounds, seed)
        biased_coin_flips_adversary.assign_losses()
        cum_losses = biased_coin_flips_adversary.get_cum_losses()
        best_actions = biased_coin_flips_adversary.get_best_actions_in_hindsight()
        best_action = best_actions[-1]
        # Test best action
        self.assertEqual(best_action, np.argmin(biases))