import argparse
import random
import json
import tempfile, os
from shutil import copyfile
from copy import copy

import numpy as np
import deepspeed

from zero.cifar10.train_ds import train


TEMPLATE_DS_CONFIG = {
  "train_batch_size": 24,
  "train_micro_batch_size_per_gpu": 3,
  "steps_per_print": 10,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 3e-5,
      "weight_decay": 0.0,
      "bias_correction": False
    }
  },
  "gradient_clipping": 1.0,
  "fp16": {
    "enabled": True
  },
  "zero_optimization": None
}

STAGE = [1, 2]
ALLGATHER_PARTITIONS = [True, False]
ALLGATHER_BUCKET_SIZE = [5e8]
OVERLAP_COMM = [False, True]
REDUCE_SCATTER = [False, True]
REDUCE_BUCKET_SIZE = [5e8]
CONTIGUOUS_GRADIENTS = [False, True]
CPU_OFFLOAD = [False, True]


tried_parameters = set()


def sample_parameters():
    while True:
        zero_optimization_params = {
            "stage": random.sample(STAGE, k=1)[0],
            "allgather_partitions": random.sample(ALLGATHER_PARTITIONS, k=1)[0],
            "allgather_bucket_size": random.sample(ALLGATHER_BUCKET_SIZE, k=1)[0],
            "overlap_comm": random.sample(OVERLAP_COMM, k=1)[0],
            "reduce_scatter": random.sample(REDUCE_SCATTER, k=1)[0],
            "reduce_bucket_size": random.sample(REDUCE_BUCKET_SIZE, k=1)[0],
            "contiguous_gradients": random.sample(CONTIGUOUS_GRADIENTS, k=1)[0],
            "cpu_offload": random.sample(CPU_OFFLOAD, k=1)[0]
        }
        if not frozenset(zero_optimization_params.items()) in tried_parameters:
            tried_parameters.add(frozenset(zero_optimization_params.items()))
            return zero_optimization_params


def parse_arguments():
    parser = argparse.ArgumentParser(description='CIFAR')

    # cuda
    parser.add_argument('--with_cuda',
                        default=False,
                        action='store_true',
                        help='use CPU in case there\'s no GPU support')
    parser.add_argument('--use_ema',
                        default=False,
                        action='store_true',
                        help='whether use exponential moving average')

    # train
    parser.add_argument('-e',
                        '--epochs',
                        default=1,
                        type=int,
                        help='number of total epochs (default: 30)')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    return parser.parse_args()


def main():
    count = 0
    best_memory = np.inf
    best_parameters = None
    while True:
        print('Hyperparameter search iteration {}'.format(count + 1))
        zero_optimization_params = sample_parameters()
        params = copy(TEMPLATE_DS_CONFIG)
        params['zero_optimization'] = zero_optimization_params
        tempdir = tempfile.mkdtemp()
        ds_config = os.path.join(tempdir, 'ds_config.json')
        with open(ds_config, 'w') as f:
            json.dump(params, f)
        args = parse_arguments()
        args.deepspeed_config = ds_config
        try:
            used_memory = train(args)
        except Exception as e:
            continue

        if used_memory < best_memory:
            best_memory = used_memory
            best_parameters = ds_config

        count += 1
        if count >= 100:
            break

    copyfile(best_parameters, 'best_ds_config.json')
    print('Least memory used in training: {} MB'.format(best_memory))


if __name__ == "__main__":
    main()
