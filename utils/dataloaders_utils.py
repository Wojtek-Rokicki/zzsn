import sys
import torch
from torch.utils.data import (BatchSampler, DataLoader, Dataset, SequentialSampler)

sys.path.append('../')
from global_const import *

def split_dataset(dataset, config):
    lenght = len(dataset)
    lenght = int(config['Data']['train_ratio']*lenght)
    train_data = dataset[:lenght]
    test_data = dataset[lenght:]

    return train_data, test_data

def prepare_dataloaders(train_standarized_graphs, test_standarized_graphs, data_config):
    ''' Sending data to loaders.

        Returns:
            train_loader: GraphDataLoader
            test_loader: GraphDataLoader
    '''
    train_dataset = ConstrainedDataset(train_standarized_graphs)
    test_dataset = ConstrainedDataset(test_standarized_graphs)
    # Batches are infered from train_dataset dimensions (batch_size will be equal to max_size of graph)
    train_loader = GraphDataLoader(train_dataset, data_config["Data"]["batch_size"])
    test_loader = GraphDataLoader(test_dataset, data_config["Data"]["batch_size"])
    
    return train_loader, test_loader

class ConstrainedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)

    # Gets graphs with indices
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.dataset[idx]
        return sample

class CustomBatchSampler(BatchSampler):
    r""" Creates batches where all sets have the same size
    """
    def __init__(self, sampler, batch_size, drop_last):
        super().__init__(sampler, batch_size, drop_last)

    # https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do/231855#231855
    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        count = 0
        batch = 0
        for idx in self.sampler:
            batch += 1
            if batch == self.batch_size:
                count += 1
                batch = 0
        if batch > 0 and not self.drop_last:
            count += 1
        return count


class GraphDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, drop_last=False):
        data_source = dataset
        sampler = SequentialSampler(data_source) # zwraca indeksy elementów; może być używany do zmiany kolejności
        batch_sampler = CustomBatchSampler(sampler, batch_size, drop_last)
        super().__init__(dataset, batch_sampler = batch_sampler)
