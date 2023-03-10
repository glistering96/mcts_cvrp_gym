from argparse import ArgumentParser
from src.test_mcts import run


def parse_args():
    parser = ArgumentParser()

    # env params
    parser.add_argument("--num_nodes", type=int, default=20, help="Number of nodes in the test data generation")
    parser.add_argument("--num_depots", type=int, default=1, help="Number of depots in the test data generation")
    parser.add_argument("--num_env", type=int, default=1, help="Number of parallel rollout of episodes")
    parser.add_argument("--step_reward", type=bool, default=False, help="whether to have step reward. If false, only the "
                                                                       "reward in the last transition will be returned")

    # mcts params
    parser.add_argument("--num_simulations", type=int, default=50, help="Number of simulations")
    parser.add_argument("--temp_threshold", type=int, default=5, help="Temperature threshold")
    parser.add_argument("--noise_eta", type=float, default=0.25, help="Noise eta param")
    parser.add_argument("--cpuct", type=float, default=1.1, help="cpuct param")
    parser.add_argument("--normalize_value", type=bool, default=True, help="Normalize q values in mcts search")

    # model params
    parser.add_argument("--nn", type=str, default='shared_mha', help="type of policy network to use")
    parser.add_argument("--embedding_dim", type=int, default=64, help="embedding dim of network")
    parser.add_argument("--encoder_layer_num", type=int, default=2, help="encoder layer of network. IGNORED")
    parser.add_argument("--qkv_dim", type=int, default=16, help="attention dim")
    parser.add_argument("--head_num", type=int, default=4, help="attention head dim")
    parser.add_argument("--C", type=int, default=10, help="C parameter that is applied to the tanh activation on the"
                                                          " last layer output of policy network")

    # tester params
    parser.add_argument("--mini_batch_size", type=int, default=2048, help="mini-batch size")
    parser.add_argument("--train_epochs", type=int, default=10, help="train epochs")
    parser.add_argument("--epochs", type=int, default=5000, help="number of episodes to run")
    parser.add_argument("--num_episode", type=int, default=100, help="number of episodes to run")
    parser.add_argument("--model_load", type=str, help="If value is greater than 0, it will load the model")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate of ADAM optimizer")
    parser.add_argument("--gpu_id", type=int, default=0, help="Id of gpu to use")
    parser.add_argument("--num_proc", type=int, default=5, help="number of episodes to run")

    # etc.
    parser.add_argument("-s", "--seed", type=int, default=1, help="values smaller than 1 will not set any seeds")
    parser.add_argument("--data_path", type=str, default='./data', help="Test data file locations")

    args = parser.parse_args()
    args.result_dir = f'./result'
    args.tb_log_dir = f'./logs'

    return args


if __name__ == '__main__':
    r = {}

    for cpuct in [1, 1.1, 1.2, 10, 50]:
        args = parse_args()
        args.nn = 'shared_mha'
        args.model_load = f'./data/for_mcts/N{args.num_nodes}_D{args.num_depots}_7.2342.pt'
        args.cpuct = cpuct
        args.num_simulations = 50
        r[cpuct] = run(args)

    print(dict(sorted(r.items(), key=lambda x: x[1])))

