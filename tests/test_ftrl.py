'''
Testing the multi_agent_ftrl/ftrl.py
'''

import unittest
import numpy as np
from scipy.stats import entropy
from multi_agent_ftrl.ftlr import FTRL, Hedge, BanditFTRL, Exp3


class TestFunctions(unittest.TestCase):
    '''
    Test classes in multi_agent_ftrl/ftrl.py
    '''

    def test_hedge(self):
        '''
        Test the implementation of one step
        update in Hedge algorithm.
        '''
        n_actions = 2
        losses = np.array([1.12, 0.2311])
        learning_rate = 0.223
        hedge = Hedge(n_actions)
        hedge.step(losses, learning_rate)
        res = hedge.select_prob()
        ans = np.exp(-learning_rate*losses) / np.sum(np.exp(-learning_rate*losses))
        self.assertIsNone(np.testing.assert_array_equal(res, ans))


    def test_ftrl(self):
        '''
        Test the implementation of one step
        update in FTRL algorithm.
        '''
        n_actions = 2
        loss_0 = np.array([0, 0])
        learning_rate_0 = 0.1
        ftrl = FTRL(n_actions)
        ftrl.step(
            loss_0,
            potential=lambda w: -entropy(w) / learning_rate_0
        )
        res = ftrl.weights
        ans = np.ones(shape=(n_actions,)) / n_actions
        self.assertIsNone(np.testing.assert_array_equal(res, ans))

        loss_1 = np.array([1.12, 0.2311])
        learning_rate_1 = 0.1
        ftrl.step(
            loss_1,
            potential=lambda w: -entropy(w)/learning_rate_1
        )
        hedge = Hedge(n_actions)
        hedge.step(loss_1, learning_rate_1)
        self.assertIsNone(np.testing.assert_almost_equal(
            ftrl.select_prob(),
            hedge.select_prob(),
            decimal=5))

    def test_hedge_ftrl_multi_steps(self):
        '''
        Test the difference between the implementation
        of Hedge algorithm and FTRL with negentropy
        after multiple steps.

        Due to the numerical optimization, Hedge and
        FTRL are the same up to ****3**** decimal digits.
        '''
        n_actions = 5
        hedge = Hedge(n_actions)
        ftrl = FTRL(n_actions)
        rounds = 3000
        learning_rate = 1 / np.sqrt(rounds)
        for _ in range(rounds):
            loss_t = np.random.rand(n_actions)
            hedge.step(loss_t, learning_rate)
            ftrl.step(
                loss_t,
                potential=lambda w: -entropy(w) / learning_rate
            )
        self.assertIsNone(np.testing.assert_almost_equal(
            ftrl.select_prob(),
            hedge.select_prob(),
            decimal=3))

    def test_exp3(self):
        """Test EXP3 algorithm for a single update step
        """
        n_actions = 2
        loss = 1
        chosen_action = 0
        loss_vec = np.zeros((n_actions,))
        gamma = 0.223
        exp3 = Exp3(n_actions)
        loss_vec[chosen_action] = loss / exp3.select_prob()[chosen_action]
        exp3.update(loss, chosen_action=chosen_action, learning_rate=gamma)
        res = exp3.select_prob()
        ans = np.exp(-gamma*loss_vec) / np.sum(np.exp(-gamma*loss_vec))
        self.assertIsNone(np.testing.assert_array_equal(res, ans))

    def test_adversarial_bandit_exp3(self):
        """Test the performance of exp3 in adversarial bandit
        """
        rng = np.random.default_rng(12345)
        n_actions = 50
        rounds = 100
        gamma = 1/np.sqrt(rounds)
        losses_vecs = rng.random((rounds, n_actions))
        exp3 = Exp3(n_actions)
        cum_loss = 0
        for time_step in range(rounds):
            select_probs = exp3.select_prob()
            action = rng.choice(n_actions, p=select_probs)
            loss = losses_vecs[time_step, action]
            exp3.update(loss, chosen_action=action, learning_rate=gamma)
            cum_loss += loss
        regret = cum_loss - np.min(np.sum(losses_vecs, axis=0))
        ratio = regret / np.sqrt(n_actions*np.log(n_actions)*rounds)
        print(ratio)

    def test_bftrl(self):
        '''
        Test the implementation of one step
        update in BanditFTRL algorithm.
        '''
        random_seed = 1234
        rng_adversarial = np.random.default_rng(random_seed)
        n_actions = 5
        rounds = 500
        losses = rng_adversarial.random(rounds)
        actions = rng_adversarial.choice(n_actions, rounds)
        learning_rate = 0.1

        bandit_ftrl = BanditFTRL(n_actions)
        exp3 = Exp3(n_actions)

        for i in range(rounds):
            bandit_ftrl.update(
                losses[i],
                actions[i],
                potential=lambda w: -entropy(w) / learning_rate
            )

            exp3.update(
                losses[i],
                actions[i],
                learning_rate
            )  
            bandit_ftrl_probs = bandit_ftrl.select_prob()
            exp3_probs = exp3.select_prob()

        # test the most likely choice
        self.assertEqual(
            np.argmax(bandit_ftrl_probs),
            np.argmax(exp3_probs)
        )

        # test true probability difference
        self.assertIsNone(np.testing.assert_almost_equal(
            bandit_ftrl_probs,
            exp3_probs,
            decimal=1)
        )
