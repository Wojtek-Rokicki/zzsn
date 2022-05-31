import os
import numpy as np
import torch
import random
from utils.metrics import (HungarianVAELoss, adjacency_graph_metrics)
from model import SetTransformerVae
from utils.log_utils import log_train_metrics, log_test_metrics
from utils.plot_utils import plot_reconstruction, plot_generated_sets
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.graphs_utils import normalize_output_graph

random.seed(123)
torch.manual_seed(123)
np.random.seed(123)

def train(args, config,dataset_config, train_loader, test_loader, wandb):
    use_cuda = args.gpu is not None and torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda:" + str(args.gpu))
        torch.cuda.set_device(args.gpu)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    else:
        device = "cpu"
    args.device = device
    args.kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    print('Device used:', device)

    # Define model, loss, optimizer
    model = SetTransformerVae(config).to(device)
    # TODO: Adjust loss function for our case
    loss_fct = HungarianVAELoss(config.glob.lmbdas, config.set_generator_config.learn_from_latent) ## BSELoss
    # loss_fct = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience, min_lr=1e-6)
    # wandb.watch(model, log_freq=100)

    def train_epoch(epoch: int):
        model.train()
        losses = np.zeros(4)           # Train_loss, n_loss
        for _, data in enumerate(train_loader): # data - ustandaryzowane grafy
            optimizer.zero_grad()
            data = data.to(device)
            output = model(data)
            loss = loss_fct(*output, data)
            if torch.isnan(loss[0]):
                raise ValueError("Nan detected in the loss")
            loss[0].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            optimizer.step()
            losses += [l.item() / len(train_loader.dataset) for l in loss]

        log_train_metrics(args, losses, optimizer.param_groups[0]['lr'], epoch, wandb, verbose=True)
        return losses

    def test():
        model.eval()
        with torch.no_grad():
            losses = np.zeros(4)           # Train_loss, n_loss
            for idx, data in enumerate(test_loader):
                data = data.to(device)
                output = model.reconstruct(data)
                loss = loss_fct(*output, data)
                losses += [l.item() / len(train_loader.dataset) for l in loss]

                #### METRICS ######
                normalized_output = normalize_output_graph(output[0][0])
                precision, recall, f1, accuracy = adjacency_graph_metrics(normalized_output, data.numpy())   
                print( f'Test batch {idx} -- Precision: {precision}, Recall: {recall}, F1:{f1}, Accuracy: {accuracy}')
                
            # Whole metrics of 
            log_test_metrics(args, losses, epoch, wandb, verbose=True)

            return losses

    # Train
    for epoch in range(0, args.epochs):
        if optimizer.param_groups[0]['lr'] < 1e-5:         # Stop training if learning rate is too small
            break
        losses = train_epoch(epoch)
        scheduler.step(losses[0])
        if epoch % args.test_every == 0:
            test()
        if args.plot_every > 0 and epoch % args.plot_every == 0:
            for i, data in enumerate(train_loader):
                data = data.to(device)
                output = model(*data)
                plot_reconstruction(f"rec_{args.name}_e{epoch}", output, data)
                if i > 0:
                    break
            with torch.no_grad():
                generated = [model.generate(device) for i in range(10)]
            plot_generated_sets(generated, f"gen_{args.name}_epoch{epoch}", num_prints=10, folder='./plots')

    print("Training completed. Exiting.")
