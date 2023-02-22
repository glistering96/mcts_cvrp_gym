import random
from collections import deque
from copy import copy, deepcopy
from pathlib import Path

import torch
from torch.multiprocessing import Pool
import torch.multiprocessing as mp
import torch.nn.functional as F
from gymnasium.wrappers import RecordVideo
from torch.optim import Adam as Optimizer
from torch.utils.tensorboard import SummaryWriter

from src.common.utils import add_hparams, check_debug, explained_variance
from src.env.cvrp_gym import CVRPEnv
from src.models.common_modules import get_batch_tensor
from src.rollout import RolloutBase, rollout_episode

tb = None
hparam_writer = None


class TrainerModule(RolloutBase):
    def __init__(self, env_params, model_params, mcts_params, logger_params, optimizer_params, run_params, h_params):
        # save arguments
        super().__init__(env_params, model_params, mcts_params, logger_params, run_params)
        global tb, hparam_writer

        self.optimizer_params = optimizer_params
        logging_params = logger_params["log_file"]
        filename = '/'.join(logging_params['desc'].split('/')[1:])
        tb_log_dir = logger_params['tb_log_dir']

        tb_log_path = f'{tb_log_dir}/{filename}/'
        tb_hparam_path = f'/hparams/{tb_log_dir}/{filename}/'

        tb = SummaryWriter(tb_log_path)
        hparam_writer = SummaryWriter(tb_hparam_path)

        self.hparam = h_params

        # policy_optimizer
        self.optimizer = Optimizer(self.model.parameters(), **optimizer_params)

        self.start_epoch = 1
        self.best_score = float('inf')
        self.best_loss = float('inf')
        self.current_lr = optimizer_params['lr']
        self.trainExamplesHistory = deque([], maxlen=1000000)

        if Path('../data/mcts_train_data.pt').exists():
            self.trainExamplesHistory = torch.load('../data/mcts_train_data.pt')

        if run_params['model_load']['enable'] is True:
            self._load_model(run_params['model_load'])

        self.debug_epoch = 0

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
                action, _ = self.model.predict(obs)
                obs, reward, done, truncated, info = env.step(int(action))

        # close the environment and the video recorder
        env.close()
        return -reward

    def run(self):
        self.time_estimator.reset(self.epochs)
        model_save_interval = self.run_params['logging']['model_save_interval']
        log_interval = self.run_params['logging']['log_interval']

        global tb, hparam_writer
        total_epochs = self.run_params['epochs']

        for epoch in range(self.start_epoch, total_epochs + 1):
            # Train
            # print("epochs ", epochs) # debugging

            train_score, total_loss, p_loss, val_loss, explained_var = self._train_one_epoch(epoch)

            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, total_epochs)

            # for k, v in self.model.state_dict().items():
            #     print(f"{k}: {v.requires_grad}")

            ############################
            # Logs & Checkpoint
            ############################
            all_done = (epoch == total_epochs)
            to_compare_score = train_score
            updated = False

            if epoch < 200:
                with torch.no_grad():
                    self.best_model.load_state_dict(self.model.state_dict())
                self.logger.info("Best model parameter updated.")
                updated = True

            if to_compare_score < self.best_score:
                # normal logging interval
                self.logger.info("Saving the best policy_net")
                self.best_score = to_compare_score
                self._save_checkpoints(epoch, is_best=True)

                if not updated:
                    with torch.no_grad():
                        self.best_model.load_state_dict(self.model.state_dict())

                    self.logger.info("Best model parameter updated.")

                self._log_info(epoch, train_score, total_loss, p_loss,
                               val_loss, elapsed_time_str, remain_time_str)

            elif all_done or (epoch % model_save_interval) == 0:
                # when the best score is collected
                self.logger.info(f"Saving the trained policy_net. Current lr: {self.current_lr}")
                self._save_checkpoints(epoch, is_best=False)
                self._record_video(f"{epoch}")
                self._log_info(epoch, train_score, total_loss, p_loss,
                               val_loss, elapsed_time_str, remain_time_str)

            elif epoch % log_interval == 0:
                # logging interval
                self._log_info(epoch, train_score, total_loss, p_loss,
                               val_loss, elapsed_time_str, remain_time_str)

            # self._save_checkpoints("last", is_best=False)
            tb.add_scalar('score/train_score', train_score, epoch)
            tb.add_scalar('loss/total_loss', total_loss, epoch)
            tb.add_scalar('loss/p_loss', p_loss, epoch)
            tb.add_scalar('loss/val_loss', val_loss, epoch)
            tb.add_scalar('loss/explained_var', explained_var, epoch)

            add_hparams(hparam_writer, self.hparam, {'train_score': train_score, 'best_score': self.best_score},
                        epoch)

            self.debug_epoch += 1

            # All-done announcement
            if all_done:
                tb.flush()
                tb.close()
                self.logger.info(" *** Training Done *** ")

        # except:
        #     self.logger.info("Training stopped early")
        #     self._save_checkpoints("last", is_best=False)

    def _set_lr(self, epoch):
        if 500 < epoch <= 1000:
            self.current_lr = self.current_lr * 0.5

        elif 1000 < epoch <= 1500:
            self.current_lr = self.current_lr * 0.5

        elif 1500 < epoch <= 2000:
            self.current_lr = self.current_lr * 0.5

        else:
            pass

        self.optimizer.param_groups[0]["lr"] = self.current_lr

    def _train_one_epoch(self, epoch):
        # train for one epoch.
        # In one epoch, the policy_net trains over given number of scenarios from tester parameters
        # The scenarios are trained in batched.
        num_episodes = self.run_params['num_episode']

        self._set_lr(epoch)

        remaining = num_episodes
        done = 0

        while remaining > 0:
            iterationTrainExamples = []

            if check_debug():
                num_works = min(2, remaining)

                pool = Pool(processes=num_works)
                temp = self._get_temp(epoch)
                params = [(self.env, self.best_model, self.mcts_params, temp) for _ in range(num_works)]
                result = pool.starmap(rollout_episode, params)

                pool.close()
                pool.join()

                for r in result:
                    iterationTrainExamples += r
                # iterationTrainExamples = self.work(epoch)
                # num_works = 1

            else:
                num_cpus = self.run_params['num_proc']

                num_works = min(num_cpus, remaining)

                pool = Pool(processes=num_works)
                temp = self._get_temp(epoch)
                params = [(self.env, self.best_model, self.mcts_params, temp) for _ in range(num_works)]
                result = pool.starmap(rollout_episode, params)

                pool.close()
                pool.join()

                for r in result:
                    iterationTrainExamples += r

                # iterationTrainExamples = self.work(epoch)
                # num_works = 1

            remaining = remaining - num_works
            done += num_works

            self.trainExamplesHistory.extend(iterationTrainExamples)

            print(f"\rSimulating episodes done: {done}/{num_episodes}", end="")

        print("\r", end="")
        self.logger.info(
            f"Simulating episodes done: {done}/{num_episodes}. Number of data is {len(self.trainExamplesHistory)}")

        reward, total_loss, pi_loss, v_loss, explained_var = self._train_model(self.trainExamplesHistory, epoch)
        # reward, total_loss, pi_loss, v_loss, explained_var = 0,0,0,0,0

        return reward, total_loss, pi_loss, v_loss, explained_var

    def _train_model(self, examples, epoch):
        # trainExamples: [(obs, action_prob_dist, reward)]
        self.model.train()
        batch_size = min(len(examples), self.run_params['mini_batch_size'])
        train_epochs = self.run_params['train_epochs']

        t_losses = 0
        pi_losses = 0
        v_losses = 0
        rewards = 0
        num_observations = 0

        train_epochs = min([max(10, epoch), train_epochs])
        exp_var_rewards = []
        exp_var_values = []
        # train_epochs = 1

        for epoch in range(train_epochs):
            batch_from = 0
            remaining = len(examples)
            batch_idx = list(range(len(examples)))
            random.shuffle(batch_idx)

            while remaining > 0:
                B = min(batch_size, remaining)
                selected_batch_idx = batch_idx[batch_from:batch_from + B]
                obs, policy, reward = list(zip(*[examples[i] for i in selected_batch_idx]))

                obs_batch_tensor = get_batch_tensor(obs)

                target_probs = torch.tensor(policy, dtype=torch.float32, device=self.device).squeeze(1).detach()
                # (B, num_vehicles)

                target_reward = torch.tensor(reward, dtype=torch.float32, device=self.device).view(B, -1).detach()
                # (B, )

                # compute output
                out_pi, out_v = self.model(obs_batch_tensor)

                l_pi = F.cross_entropy(out_pi, target_probs)
                l_v = F.mse_loss(out_v, target_reward)
                loss = l_pi + l_v + 0.0001 * self.l2()

                # record loss

                t_losses += loss.item() * B
                rewards += sum(reward)
                pi_losses += l_pi.item() * B
                v_losses += l_v.item() * B

                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                remaining -= B
                batch_from += B
                num_observations += B

                exp_var_rewards += reward
                exp_var_values += out_v.detach().cpu().view(-1,).tolist()

        rewards /= num_observations
        t_losses /= num_observations
        pi_losses /= num_observations
        v_losses /= num_observations

        explained_var = explained_variance(exp_var_values, exp_var_rewards)

        del loss, out_pi, out_v, l_v, l_pi

        return -rewards, t_losses, pi_losses, v_losses, explained_var

    def l2(self):
        l2_reg = torch.tensor(0., device=self.device)

        for param in self.model.parameters():
            l2_reg += torch.norm(param)

        return l2_reg
