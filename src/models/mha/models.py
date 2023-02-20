from src.models.mha.modules import *
from src.models.common_modules import get_encoding, _to_tensor


class SharedMHA(nn.Module):
    def __init__(self, **model_params):
        super(SharedMHA, self).__init__()

        self.model_params = model_params
        self.device = model_params['device']

        self.policy_net = Policy(**model_params)
        self.value_net = Value(**model_params)
        self.encoder = Encoder(**model_params)
        self.latent_dim_pi = model_params['action_size']

    def _get_obs(self, observations):
        observations = _to_tensor(observations, self.device)

        xy, demands = observations['xy'], observations['demands']
        # (N, 2), (N, 1)

        cur_node = observations['pos']
        # (1, )

        load = observations['load']
        # (1, )

        available = observations['available']
        # (1, )

        if demands.dim() == 2:
            demands = demands.unsqueeze(-1)

        if load.dim() == 2:
            load = load.unsqueeze(1)

        if cur_node.dim() == 2:
            cur_node = cur_node.unsqueeze(1)

        if available.dim() == 2:
            available = available.unsqueeze(1)

        return load, cur_node, available, xy, demands

    def forward(self, obs):
        load, cur_node, available, xy, demands = self._get_obs(obs)

        B, T = load.size(0), load.size(1)

        mask = torch.zeros_like(available).type(torch.float32)
        mask[available == False] = float('-inf')

        encoding = self.encoder(xy, demands)

        val = self.value_net(cur_node, load, mask, xy, demands, T, encoding)

        probs = self.policy_net(cur_node, load, mask, xy, demands, T, encoding)
        probs = probs.reshape(-1, probs.size(-1))
        val = val.reshape(-1, 1)

        return probs, val

    def forward_actor(self, obs):
        load, cur_node, available, xy, demands = self._get_obs(obs)

        B, T = load.size(0), load.size(1)

        mask = torch.zeros_like(available).type(torch.float32)
        mask[available == False] = float('-inf')

        encoding = self.encoder(xy, demands)

        probs = self.policy_net(cur_node, load, mask, xy, demands, T, encoding)
        probs = probs.reshape(-1, probs.size(-1))

        return probs

    def forward_critic(self, obs):
        load, cur_node, available, xy, demands = self._get_obs(obs)

        B, T = load.size(0), load.size(1)

        mask = torch.zeros_like(available).type(torch.float32)
        mask[available == False] = float('-inf')

        encoding = self.encoder(xy, demands)

        val = self.value_net(cur_node, load, mask, xy, demands, T, encoding)

        val = val.reshape(-1, 1)
        return val


class SeparateMHA(nn.Module):
    def __init__(self, **model_params):
        super(SeparateMHA, self).__init__()

        self.model_params = model_params
        self.device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')

        self.policy_net = Policy(**model_params)
        self.value_net = Value(**model_params)
        self.encoder = Encoder(**model_params)
        self.latent_dim_pi = model_params['action_size']

    def _get_obs(self, observations):
        xy, demands = observations['xy'], observations['demands']
        # (N, 2), (N, 1)

        cur_node = observations['pos']
        # (1, )

        load = observations['load']
        # (1, )

        available = observations['available']
        # (1, )

        if demands.dim() == 2:
            demands = demands.unsqueeze(-1)

        if load.dim() == 2:
            load = load.unsqueeze(1)

        if cur_node.dim() == 2:
            cur_node = cur_node.unsqueeze(1)

        if available.dim() == 2:
            available = available.unsqueeze(1)

        return load, cur_node, available, xy, demands

    def forward(self, obs):
        load, cur_node, available, xy, demands = self._get_obs(obs)

        B, T = load.size(0), load.size(1)

        mask = torch.zeros_like(available).type(torch.float32)
        mask[available == False] = float('-inf')

        val = self.value_net(cur_node, load, mask, xy, demands, T, self.encoder(xy, demands))

        probs = self.policy_net(cur_node, load, mask, xy, demands, T, self.encoder(xy, demands))
        probs = probs.reshape(-1, probs.size(-1))
        val = val.reshape(-1, 1)

        return probs, val

    def forward_actor(self, obs):
        load, cur_node, available, xy, demands = self._get_obs(obs)

        B, T = load.size(0), load.size(1)

        mask = torch.zeros_like(available).type(torch.float32)
        mask[available == False] = float('-inf')

        encoding = self.encoder(xy, demands)

        probs = self.policy_net(cur_node, load, mask, xy, demands, T, encoding)
        probs = probs.reshape(-1, probs.size(-1))

        return probs

    def forward_critic(self, obs):
        load, cur_node, available, xy, demands = self._get_obs(obs)

        B, T = load.size(0), load.size(1)

        mask = torch.zeros_like(available).type(torch.float32)
        mask[available == False] = float('-inf')

        encoding = self.encoder(xy, demands)

        val = self.value_net(cur_node, load, mask, xy, demands, T, encoding)

        val = val.reshape(-1, 1)
        return val


class Encoder(nn.Module):
    def __init__(self, **model_params):
        super(Encoder, self).__init__()

        self.model_params = model_params
        self.embedding_dim = model_params['embedding_dim']

        self.input_embedder = nn.Linear(3, self.embedding_dim)
        self.embedder = nn.ModuleList([EncoderLayer(**model_params) for _ in range(model_params['encoder_layer_num'])])

    def forward(self, xy, demand):
        out = torch.cat([xy, demand], -1)
        out = self.input_embedder(out)

        for layer in self.embedder:
            out = layer(out)

        return out


class Policy(nn.Module):
    def __init__(self, **model_params):
        super(Policy, self).__init__()
        self.C = model_params['C']
        self.encoder = Encoder(**model_params)
        self.decoder_common = DecoderCommon(**model_params)
        self.embedding_dim = model_params['embedding_dim']

    def forward(self, cur_node, load, mask, xy, demand, T, encoding=None):
        if encoding is None:
            encoding = self.encoder(xy, demand)

        self.decoder_common.set_kv(encoding)

        last_node = get_encoding(encoding, cur_node.long(), T)

        mh_atten_out = self.decoder_common(last_node, load, mask)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.decoder_common.single_head_key)
        # shape: (batch, problem)

        sqrt_embedding_dim = math.sqrt(self.embedding_dim)

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, problem)

        score_clipped = self.C * torch.tanh(score_scaled)

        score_masked = score_clipped + mask

        probs = F.softmax(score_masked, dim=-1)

        return probs


class Value(nn.Module):
    def __init__(self, **model_params):
        super(Value, self).__init__()
        self.encoder = Encoder(**model_params)
        self.decoder_common = DecoderCommon(**model_params)
        self.embedding_dim = model_params['embedding_dim']
        self.val = nn.Linear(self.embedding_dim, 1)

    def forward(self, cur_node, load, mask, xy, demand, T, encoding=None):
        if encoding is None:
            encoding = self.encoder(xy, demand)

        self.decoder_common.set_kv(encoding)

        last_node = get_encoding(encoding, cur_node.long(), T)

        mh_atten_out = self.decoder_common(last_node, load, mask)

        val = self.val(mh_atten_out)

        return val


