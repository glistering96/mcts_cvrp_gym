from typing import Callable

import torch
from gym import spaces
from stable_baselines3.common.distributions import DiagGaussianDistribution, StateDependentNoiseDistribution, \
    CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule

from src.models.mha.models import SharedMHA, SeparateMHA
from src.models.mha_mlp.models import SharedMHAMLP, SeparateMHAMLP


class CustomActorCriticWrapper(ActorCriticPolicy):
    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Callable[[float], float],
            use_sde=False,
            **model_params,
    ):
        self.model_params = model_params

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            use_sde = use_sde
            # Pass remaining arguments to base class
        )

    def _build_mlp_extractor(self) -> None:
        nn = self.model_params['nn']

        if nn == 'shared_mha':
            self.mlp_extractor = SharedMHA(**self.model_params)

        elif nn == 'separate_mha':
            self.mlp_extractor = SeparateMHA(**self.model_params)

        elif nn == 'shared_mhamlp':
            self.mlp_extractor = SharedMHAMLP(**self.model_params)

        elif nn == 'separate_mhamlp':
            self.mlp_extractor = SeparateMHAMLP(**self.model_params)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, latent_sde_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, (CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution)):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def forward(self, obs, deterministic: bool = False):
        probs, values = self.mlp_extractor(obs)

        # Original logic: nn returns latent values. But our model returns the whole probability of nodes.
        # distribution = self._get_action_dist_from_latent(probs)
        # actions = distribution.get_actions(deterministic=deterministic)

        distribution = torch.distributions.Categorical(probs=probs)
        actions = distribution.sample()
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1,) + self.action_space.shape)
        return actions, values, log_prob

    def get_distribution(self, obs):
        latent_pi = self.mlp_extractor.forward_actor(obs)
        return self._get_action_dist_from_latent(latent_pi)

    def _predict(self, obs, deterministic: bool = False):
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        prob = self.mlp_extractor.forward_actor(obs)
        dist = torch.distributions.Categorical(probs=prob)

        if deterministic:
            return dist.probs.argmax(-1)

        else:
            return dist.sample()

    def predict_values(self, obs):
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        return self.mlp_extractor.forward_critic(obs)

    def evaluate_actions(self, obs, actions):
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        probs, values = self.mlp_extractor(obs)
        distribution = self._get_action_dist_from_latent(probs)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return values, log_prob, entropy