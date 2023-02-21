from logging import getLogger
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.vec_env import VecNormalize

from src.common.dataclass import rollout_result
from src.common.utils import TimeEstimator, deepcopy_state, get_result_folder
from src.env.cvrp_gym import CVRPEnv as Env
from src.models.mha.models import SharedMHA, SeparateMHA
from src.models.mha_mlp.models import SharedMHAMLP, SeparateMHAMLP
from src.mcts import MCTS


class RolloutBase:
    def __init__(self,
                 env_params,
                 model_params,
                 mcts_params,
                 logger_params,
                 run_params):
        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.run_params = run_params
        self.logger_params = logger_params
        self.mcts_params = mcts_params

        # cuda
        USE_CUDA = self.run_params['use_cuda']

        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder(logger_params['log_file']['desc'],
                                               date_prefix=logger_params['log_file']['date_prefix'],
                                               result_dir=logger_params['log_file']['result_dir']
                                               )
        Path(self.result_folder).mkdir(parents=True, exist_ok=True)

        if USE_CUDA:
            cuda_device_num = self.run_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        self.device = device

        # Env
        self.env = Env(**env_params)
        # self.env = VecNormalize(self.env, norm_obs=False )

        # Model
        self.model_params['device'] = device
        self.model_params['action_size'] = env_params['num_depots'] + env_params['num_nodes']

        self.model = self._get_model()
        self.best_model = self._get_model()
        self.best_model.eval()

        self.model.share_memory()
        self.best_model.share_memory()

        # etc.
        self.epochs = 1
        self.best_score = float('inf')
        self.time_estimator = TimeEstimator()

    def _get_model(self):
        nn = self.model_params['nn']

        if nn == 'shared_mha':
            return SharedMHA(**self.model_params)

        elif nn == 'separate_mha':
            return SeparateMHA(**self.model_params)

        elif nn == 'shared_mhamlp':
            return SharedMHAMLP(**self.model_params)

        elif nn == 'separate_mhamlp':
            return SeparateMHAMLP(**self.model_params)

    def _save_checkpoints(self, epoch, is_best=False):
        file_name = 'best' if is_best else epoch

        checkpoint_dict = {
            'epoch': epoch,
            'model_params': self.model_params,
            'best_score': self.best_score,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }

        torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, file_name))

    def _load_model(self, model_load):
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
        self.start_epoch = checkpoint['epoch'] + 1

        self.best_score = checkpoint['best_score']

        loaded_state_dict = checkpoint['model_state_dict']
        self.best_model.load_state_dict(loaded_state_dict)
        self.model.load_state_dict(loaded_state_dict)

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.logger.info(
            f"Successfully loaded pre-trained policy_net {model_load['path']} with epoch: {model_load['epoch']}")

    def _log_info(self, epoch, train_score, total, p_loss, val_loss, elapsed_time_str,
                  remain_time_str):

        self.logger.info(
            f'Epoch {epoch:3d}: Score: {train_score:.4f}, total_loss: {total:.4f}, p_loss: {p_loss:.4f}, '
            f'val_loss: {val_loss:.4f}, Best: {self.best_score:.4f}')

        self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
            epoch, self.run_params['epochs'], elapsed_time_str, remain_time_str))
        self.logger.info('=================================================================')

    def _get_temp(self, epoch):
        total_epochs = self.run_params['epochs']

        if epoch < int(total_epochs/3):
            return 1

        elif epoch < int(total_epochs*2/3):
            return 0.5

        else:
            return 0.25

    def run(self):
        # abstract method
        raise NotImplementedError

    def _rollout_episode(self, env, agent, mcts_params, epoch):
        obs = self.env.reset()
        buffer = []
        done = False
        print("rollout start")

        # if self.best_model.training:
        #     temp = self._get_temp(epoch)
        #
        # else:
        #     temp = 0    # temp = 0 means exploitation. No stochastic sampling

        # episode rollout
        # gather probability of the action and value estimates for the state
        debug = 0
        self.best_model.eval()

        with torch.no_grad():
            while not done:
                mcts = MCTS(self.env, self.best_model, self.mcts_params)
                temp = self._get_temp(debug)
                action_probs = mcts.get_action_prob(obs, temp=temp)
                action = np.random.choice(len(action_probs), p=action_probs)

                buffer.append((obs, action_probs))

                next_state, reward, done, _ = self.env.step(action)

                obs = next_state

                debug += 1

                if done:
                    result = [(x[0], x[1], float(reward)) for x in buffer]
                    return result

def rollout_episode(env, agent, mcts_params, temp):
    obs = env.reset()
    buffer = []
    done = False
    print("rollout start")

    # episode rollout
    # gather probability of the action and value estimates for the state
    debug = 0
    agent.eval()

    with torch.no_grad():
        while not done:
            mcts = MCTS(env, agent, mcts_params)
            action_probs = mcts.get_action_prob(obs, temp=temp)
            action = np.random.choice(len(action_probs), p=action_probs)

            buffer.append((obs, action_probs))

            next_state, reward, done, _ = env.step(action)

            obs = next_state

            debug += 1

            if done:
                result = [(x[0], x[1], float(reward)) for x in buffer]
                return result