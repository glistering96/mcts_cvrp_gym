import logging
import math
from copy import deepcopy

import numpy as np
import torch

EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, env, model, mcts_params):
        self.env = deepcopy(env)
        self.model = model
        self.mcts_params = mcts_params
        self.action_space = mcts_params['action_space']
        self.all_q_vals = []
        self.cpuct = mcts_params['cpuct']
        self.normalize_q_value = mcts_params['normalize_value']

        self.Q = {}  # stores Q values for s,a (as defined in the paper)
        self.W = {}  # sum of values
        self.Ns = {} # sum of visit counts for state s
        self.N = {}  # stores #times edge s,a was visited
        self.P = {}  # stores initial policy (returned by neural net), dtype: numpy ndarray

        self.max_q_val = float('-inf')
        self.min_q_val = float('inf')
        self.factor = (1, 0)

    def _cal_probs(self, target_state, temp):
        s = target_state['t']
        counts = [self.N[(s, a)] if (s, a) in self.N else 0 for a in range(self.action_space)]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1

        else:
            counts = [x ** (1. / temp) for x in counts]
            counts_sum = float(sum(counts)) + 1e-8
            probs = [x / counts_sum for x in counts]

        return probs

    def get_action_prob(self, target_state, temp=1):
        """
        Simulate and return the probability for the target state based on the visit counts acquired from simulations
        """

        for i in range(self.mcts_params['num_simulations']):
            self._run_simulation(target_state)

        probs = self._cal_probs(target_state, temp)
        return probs

    def _select(self, state):
        # select the action
        # argmin (Q-U), since the reward concept becomes the loss in this MCTS
        # originally, argmax (Q+U) is true but to fit into minimization, argmin (-Q+U) is selected
        s = state['t']
        diff, min_q_val = self._cal_factor()

        best_score = float('-inf')
        best_act = -1

        # pick the action with the lowest upper confidence bound
        for a in range(self.action_space):
            p = self.P[(s,a)]

            if p == 0:
                continue

            ucb = self.cpuct * self.P[(s, a)] * math.sqrt(self.Ns[s]) / (1 + self.N[(s, a)])

            Q_val = self.Q[(s,a)]

            if self.normalize_q_value:
                Q_val = (self.Q[(s,a)] - min_q_val) / (diff + 1e-8)

            score = Q_val + ucb

            if score > best_score:
                best_score = score
                best_act = a

        a = best_act
        return a

    def _expand(self, state):
        s = state['t']

        prob_dist, v = self.model(state)
        probs = prob_dist.view(-1,).cpu().numpy()

        self.Ns[s] = 1

        for a in range(self.action_space):
            self.P[(s, a)] = probs[a]
            self.W[(s, a)] = 0
            self.Q[(s, a)] = 0
            self.N[(s, a)] = 0

        return v

    def _cal_factor(self):
        if self.all_q_vals:
            diff = self.max_q_val - self.min_q_val
            return diff, self.min_q_val

        else:
            return (1, 0)

    def _back_propagate(self, path, v):
        if isinstance(v, torch.Tensor):
            v = v.item()

        for s, a in path:
            self.N[(s,a)] += 1
            self.Ns[s] += 1
            self.W[(s,a)] += v

        for i, (s, a) in enumerate(path):
            Q_val = self.W[(s,a)] / self.N[(s,a)]
            self.all_q_vals.append(Q_val)

            if Q_val < self.min_q_val:
                self.min_q_val = Q_val

            if Q_val > self.max_q_val:
                self.max_q_val = Q_val

            self.Q[(s, a)] = Q_val

    def _run_simulation(self, root_state):
        state = root_state
        state_num = state['t']

        path = []
        done = False
        v = 0

        if not self.all_q_vals:
            v = self._expand(root_state)

        # select child node and action
        while state_num in self.Ns:
            debug_state = deepcopy(state)

            a = self._select(state)
            path.append((state_num, a))

            state, value, done, _ = self.env.step(a)
            state_num = state['t']

            if len(path) > 30:
                break

        if not done:
            # leaf node reached
            v = self._expand(state)

        else:
            # terminal node reached
            v = self.env.get_reward(state)

        self._back_propagate(path, v)