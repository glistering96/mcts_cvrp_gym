import random
from collections import deque
from copy import copy, deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.multiprocessing import Pool
import torch.multiprocessing as mp
import torch.nn.functional as F
from gymnasium.wrappers import RecordVideo
from torch.optim import Adam as Optimizer
from torch.utils.tensorboard import SummaryWriter

from src.common.utils import add_hparams, check_debug, explained_variance
from src.env.cvrp_gym import CVRPEnv
from src.mcts import MCTS
from src.models.common_modules import get_batch_tensor
from src.module_base import RolloutBase, rollout_episode

tb = None
hparam_writer = None


class TesterModule(RolloutBase):
    def __init__(self, env_params, model_params, mcts_params, logger_params, run_params, h_params):
        # save arguments
        super().__init__(env_params, model_params, mcts_params, logger_params, run_params)
        global tb, hparam_writer

        logging_params = logger_params["log_file"]
        filename = '/'.join(logging_params['desc'].split('/')[1:])
        tb_log_dir = logger_params['tb_log_dir']

        tb_log_path = f'{tb_log_dir}/{filename}/'
        tb_hparam_path = f'/hparams/{tb_log_dir}/{filename}/'

        tb = SummaryWriter(tb_log_path)
        hparam_writer = SummaryWriter(tb_hparam_path)

        self.hparam = h_params

        self.start_epoch = 1
        self.best_score = float('inf')
        self.best_loss = float('inf')

        self.debug_epoch = 0

        self.min_reward = float('inf')
        self.max_reward = float('-inf')

        self._load_model(run_params['model_load'])

    def _load_model(self, path):
        loaded = torch.load(path, map_location=self.device)
        self.best_model.load_state_dict(loaded)

    def _record_video(self, epoch):
        mode = "rgb_array"
        video_dir = self.run_params['model_load']['path'] + f'/videos/'
        data_path = self.run_params['data_path']

        env_params = deepcopy(self.env_params)
        env_params['render_mode'] = mode
        env_params['training'] = False
        env_params['seed'] = 5
        env_params['data_path'] = data_path

        env = CVRPEnv(render_mode=mode, training=False, seed=5, data_path=data_path, **self.env_params)
        # env = make_vec_env(CVRPEnv, n_envs=5, env_kwargs=env_params)
        # env = Monitor(env, video_dir, force=True)
        env = RecordVideo(env, video_dir, name_prefix=str(epoch))

        # render and interact with the environment as usual
        obs = env.reset()
        done = False

        with torch.no_grad():
            while not done:
                # env.render()
                mcts = MCTS(env, self.best_model, self.mcts_params)
                action_probs = mcts.get_action_prob(obs, temp=1)
                action = action_probs.argmax(-1)
                obs, reward, done, truncated, info = env.step(int(action))

        # close the environment and the video recorder
        env.close()
        return -reward

    def run(self):
        self.time_estimator.reset(self.epochs)
        global tb, hparam_writer

        test_score = test_one_episode(self.env, self.best_model, self.mcts_params, 1)

        # when the best score is collected
        # self._record_video(f"test")

        self.logger.info(f"Test score: {test_score}")

        # self._save_checkpoints("last", is_best=False)

        add_hparams(hparam_writer, self.hparam, {'test_score': test_score}, 1)

        tb.flush()
        tb.close()
        self.logger.info(" *** Testing Done *** ")
        return test_score

    def _test_one_epoch(self):
        # train for one epoch.
        # In one epoch, the policy_net trains over given number of scenarios from tester parameters
        # The scenarios are trained in batched.
        return rollout_episode(self.env, self.best_model, self.mcts_params, temp=1)


def test_one_episode(env, agent, mcts_params, temp):
    env.set_test_mode()
    obs = env.reset()
    done = False
    agent.eval()
    debug = 0
    with torch.no_grad():
        while not done:
            mcts = MCTS(env, agent, mcts_params, training=False)
            action_probs = mcts.get_action_prob(obs, temp=temp)
            action = int(np.argmax(action_probs, -1))   # type must be python native
            # action = action_probs.argmax(-1)

            next_state, reward, done, _, _ = env.step(action)

            obs = next_state
            debug += 1

            if done:
                return -reward