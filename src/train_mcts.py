import random
import numpy as np
import torch


from src.common.utils import create_logger, check_debug, copy_all_src
from src.trainer import TrainerModule


def run(args):
    seed = args.seed

    if seed > 0:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)

    # env_params
    num_demand_nodes = args.num_nodes
    num_depots = args.num_depots

    # mcts params
    num_simulations = args.num_simulations
    temp_threshold = args.temp_threshold
    noise_eta = args.noise_eta
    cpuct = args.cpuct
    action_space = num_demand_nodes + num_depots
    normalize_value = args.normalize_value

    # model_params
    nn = args.nn
    embedding_dim = args.embedding_dim
    encoder_layer_num = args.encoder_layer_num
    qkv_dim = args.qkv_dim
    head_num = args.head_num
    C = args.C

    # trainer params
    mini_batch_size = args.mini_batch_size
    epochs = args.epochs
    num_episode = args.num_episode
    train_epochs = args.train_epochs
    model_load = args.model_load
    load_model = True if model_load > 0 else False
    cuda_device_num = args.gpu_id
    num_proc = args.num_proc
    lr = args.lr

    # logging params
    result_dir = args.result_dir
    tb_log_dir = args.tb_log_dir

    # etc
    data_path = args.data_path

    # when debugging
    if check_debug():
        num_episode = 2
        num_simulations = 5

    env_param_nm = f"N_{num_demand_nodes}"
    model_param_nm = f"nn-{nn}-{embedding_dim}-{encoder_layer_num}-{qkv_dim}-{head_num}-{C}"
    mcts_param_nm = f"ns_{num_simulations}-temp_th_{temp_threshold}-cpuct_{cpuct}-norm_val_{normalize_value}"

    result_folder_name = f"{env_param_nm}/{model_param_nm}/{mcts_param_nm}"

    # when debugging
    if check_debug():
        result_folder_name = "debug/" + result_folder_name

    # allocating hyper-parameters
    env_params = {
        'num_nodes': num_demand_nodes,
        'num_depots': num_depots,
    }

    mcts_params = {
        'num_simulations': num_simulations,
        'temp_threshold': temp_threshold,  #
        'noise_eta': noise_eta,  # 0.25
        'cpuct': cpuct,
        'action_space': action_space,
        'normalize_value': normalize_value,
    }

    model_params = {
        'nn': nn,
        'embedding_dim': embedding_dim,
        'encoder_layer_num': encoder_layer_num,
        'qkv_dim': qkv_dim,
        'head_num': head_num,
        'C': C
    }

    h_params = {
        'num_nodes': num_demand_nodes,
        'num_depots': num_depots,
        'num_simulations': num_simulations,
        'temp_threshold': temp_threshold,  # 40
        'noise_eta': noise_eta,  # 0.25
        'cpuct': cpuct,
        'action_space': action_space,
        'normalize_value': normalize_value,
        'model_type': nn,
        'embedding_dim': embedding_dim,
        'encoder_layer_num': encoder_layer_num,
        'qkv_dim': qkv_dim,
        'head_num': head_num,
        'C': C

    }

    train_params = {
        'use_cuda': True,
        'cuda_device_num': cuda_device_num,
        'train_epochs': train_epochs,
        'epochs': epochs,
        'num_episode': num_episode,
        'mini_batch_size': mini_batch_size,
        'num_proc': num_proc,
        'data_path': data_path,

        'logging': {
            'model_save_interval': 50,
            'log_interval': 1,
        },

        'model_load': {
            'enable': load_model,
            'path': f'{result_dir}/{result_folder_name}',
            'epoch': model_load
        }
    }

    logger_params = {
        'log_file': {
            'result_dir': result_dir,
            'desc': f"./{result_folder_name}",
            'filename': 'log.txt',
            'date_prefix': False
        },
        'tb_log_dir': tb_log_dir
    }

    optimizer_params = {
        'lr': lr,
        'eps': 1e-5,
        'betas': (0.9, 0.9)
    }

    create_logger(logger_params['log_file'])

    copy_all_src(f'{result_dir}/{result_folder_name}')

    trainer = TrainerModule(env_params=env_params,
                            model_params=model_params,
                            logger_params=logger_params,
                            mcts_params=mcts_params,
                            run_params=train_params,
                            optimizer_params=optimizer_params,
                            h_params=h_params)

    trainer.run()