import itertools
import json
from argparse import ArgumentParser

import torch.multiprocessing as mp
from src.train_mcts import run


def parse_args():
    parser = ArgumentParser()

    # env params
    parser.add_argument("--num_nodes", type=int, default=20, help="Number of nodes in the test data generation")
    parser.add_argument("--num_depots", type=int, default=1, help="Number of depots in the test data generation")
    parser.add_argument("--num_env", type=int, default=1, help="Number of parallel rollout of episodes")
    parser.add_argument("--step_reward", type=bool, default=True, help="whether to have step reward. If false, only the "
                                                                       "reward in the last transition will be returned")

    # mcts params
    parser.add_argument("--num_simulations", type=int, default=30, help="Number of simulations")
    parser.add_argument("--temp_threshold", type=int, default=10, help="Temperature threshold")
    parser.add_argument("--noise_eta", type=float, default=0.25, help="Noise eta param")
    parser.add_argument("--cpuct", type=float, default=1.1, help="cpuct param")
    parser.add_argument("--normalize_value", type=bool, default=True, help="Normalize q values in mcts search")

    # model params
    parser.add_argument("--nn", type=str, default='shared_mha', help="type of policy network to use")
    parser.add_argument("--embedding_dim", type=int, default=128, help="embedding dim of network")
    parser.add_argument("--encoder_layer_num", type=int, default=2, help="encoder layer of network. IGNORED")
    parser.add_argument("--qkv_dim", type=int, default=32, help="attention dim")
    parser.add_argument("--head_num", type=int, default=6, help="attention head dim")
    parser.add_argument("--C", type=int, default=10, help="C parameter that is applied to the tanh activation on the"
                                                          " last layer output of policy network")

    # trainer params
    parser.add_argument("--mini_batch_size", type=int, default=4096, help="mini-batch size")
    parser.add_argument("--train_epochs", type=int, default=30, help="train epochs")
    parser.add_argument("--epochs", type=int, default=5000, help="number of episodes to run")
    parser.add_argument("--num_episode", type=int, default=50, help="number of episodes to run")
    parser.add_argument("--model_load", type=int, default=-1, help="If value is greater than 0, it will load the model")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate of ADAM optimizer")
    parser.add_argument("--gpu_id", type=int, default=0, help="Id of gpu to use")
    parser.add_argument("--num_proc", type=int, default=5, help="number of episodes to run")

    # etc.
    parser.add_argument("-s", "--seed", type=int, default=1, help="values smaller than 1 will not set any seeds")
    parser.add_argument("--data_path", type=str, default='./data', help="Test data file locations")

    args = parser.parse_args()
    args.result_dir = f'./result'
    args.tb_log_dir = f'./logs'

    return args


def _work(**kwargs):
    args = parse_args()

    for k, v in kwargs.items():
        setattr(args, k, v)

    score = run(args)
    str_vals = [f"{name}_{val}" for name, val in zip(kwargs.keys(), kwargs.values())]
    key = "-".join(str_vals)
    return key, score


def search_params(num_proc):
    def dict_product(dicts):
        return list(dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))

    hyper_param_dict = {
        'num_simulations' : [15, 20, 25, 30],
        'temp_threshold' : [5, 10, 15],
        'cpuct' : [0.5, 1, 1.1, 1.5, 2, 5, 10],
        'reg_coef' : [0.0001, 0.00001,  0.00005],
        'lr' : [0.00005, 0.00001]
    }

    async_result = mp.Queue()

    def __callback(val):
        async_result.put(val)

    pool = mp.Pool(num_proc)

    for params in dict_product(hyper_param_dict):
        pool.apply_async(_work, kwds=params, callback=__callback)

    pool.close()
    pool.join()

    result = {}

    while not async_result.empty():
        k, score = async_result.get()
        result[k] = score

    with open("param_result.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == '__main__':
    # from gym.envs.registration import register
    #
    # register(
    #     id='custom_gyms/CVRP-v0',
    #     entry_point='src.env.cvrp_gym:CVRPEnv',
    #     max_episode_steps=1000,
    # )

    # search_params(2, 'a2c')

    args = parse_args()
    args.step_reward = False
    args.nn = 'shared_mha'
    run(args)

    # render_test()
    # record_video()

