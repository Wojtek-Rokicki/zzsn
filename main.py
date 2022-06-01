import argparse
import os
import yaml
from easydict import EasyDict
from pathlib import Path
from utils.load_config import Configuration
import pprint
import wandb
from utils.dataloaders_utils import prepare_dataloaders
from utils.graphs_utils import get_standardized_graphs  

def parse_args():
    """
    Parse args for the main function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, help='Number of epoch', default=150)
    parser.add_argument('--batch-size', type=int, help='Size of a batch', default=128)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--gpu', type=int, help='Id of gpu device. By default use cpu')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--wandb', action='store_true', help="Use the weights and biases library")
    parser.add_argument('--name', type=str)
    parser.add_argument('--test-every', type=int, default=5)
    parser.add_argument('--plot-every', type=int, default=-1)
    parser.add_argument('--factor', type=float, default=0.5, help="Learning rate decay for the scheduler")
    parser.add_argument('--patience', type=int, default=750, help="Scheduler patience")
    parser.add_argument('--data-dir', type=str, default='./data/')
    parser.add_argument('--runs', type=int, default=1, help="Number of runs to average")
    parser.add_argument('--generator', type=str, choices=['random', 'first', 'top', 'mlp'], default="top")
    parser.add_argument('--modulation', type=str, choices=['add', 'film'], default="film")
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    root_dir = Path(__file__).parent.resolve()
    model_config_path = root_dir.joinpath("config", 'model_config.yaml')

    # Changes in CUDA_VISIBLE_DEVICES must be done before loading pytorch
    if type(args.gpu) == int:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        args.gpu = 0

    with model_config_path.open() as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = EasyDict(config)

    if args.generator == 'top':
        config["SetGenerator"]['name'] = "TopNGenerator"
    elif args.generator == 'first':
        config["SetGenerator"]['name'] = "FirstKGenerator"
    elif args.generator == 'random':
        config["SetGenerator"]['name'] = "RandomGenerator"
    elif args.generator == 'mlp':
        config["SetGenerator"]['name'] = "MLPGenerator"
    else:
        raise ValueError("Unknown generator")

    config['Decoder']['modulation'] = '+' if args.modulation == 'add' else 'film'

    # Create a name for weights and biases
    if args.name:
        args.wandb = True
    if args.name is None:
        args.name = config["SetGenerator"]['name']

    # DO NOT MOVE THIS IMPORT
    import train_test
    pprint.pprint(config)

    wandb_config = config.copy()

    data_config_path = root_dir.joinpath("config", 'data_config.yaml')    
    with data_config_path.open() as f:
        data_config = yaml.load(f, Loader=yaml.FullLoader)
        data_config = EasyDict(data_config)


    for i in range(args.runs):

        datasets = ["imdb-binary", "imdb-multi", "collab"]
        for dataset in datasets:            
            train_standarized_graphs, test_standarized_graphs, max_size = get_standardized_graphs(dataset, data_config)
            train_dataloader, test_dataloader = prepare_dataloaders(train_standarized_graphs, test_standarized_graphs, data_config)
            
            config["Global"]['dataset_max_n'] = max_size
            config["Global"]['set_channels'] = max_size
            config = Configuration(config)
            wandb.init(project="set_gen", config=wandb_config, name=f"{args.name}_{i}",
                   settings=wandb.Settings(_disable_stats=True), reinit=True,
                   mode='online' if args.wandb else 'disabled')
            wandb.config.update(args)
            print(f'Dataset: {dataset}')
            train_test.train(args, config, data_config, train_dataloader, test_dataloader, wandb)


if __name__ == '__main__':
    main()
