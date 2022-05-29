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

def prepare_dataloaders(train_standarized_graphs, train_split_indices, test_standarized_graphs, test_split_indices):
    ''' Sending data to loaders.

        Returns:
            train_loader: GraphDataLoader
            test_loader: GraphDataLoader
    '''
    train_dataset = ConstrainedDataset(train_standarized_graphs, train_split_indices)
    test_dataset = ConstrainedDataset(test_standarized_graphs, test_split_indices)
    # Batches are infered from train_dataset dimensions (batch_size will be equal to max_size of graph)
    train_loader = GraphDataLoader(train_dataset)
    test_loader = GraphDataLoader(test_dataset)
    
    return train_loader, test_loader

class ConstrainedDataset(Dataset):
    def __init__(self, dataset, split_indices):
        self.dataset = dataset
        self.split_indices = split_indices

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
    def __init__(self, sampler, batch_size, drop_last, split_indices):
        super().__init__(sampler, batch_size, drop_last)
        self.split_indices = split_indices

    # https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do/231855#231855
    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size or idx + 1 in self.split_indices: # batchuje do momentu, kiedy jest inny wymiar bądź koniec batch_size
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        count = 0
        batch = 0
        for idx in self.sampler:
            batch += 1
            if batch == self.batch_size or idx + 1 in self.split_indices:
                count += 1
                batch = 0
        if batch > 0 and not self.drop_last:
            count += 1
        return count


class GraphDataLoader(DataLoader):
    def __init__(self, dataset, drop_last=False):
        data_source = dataset
        max_size = dataset[0].shape[0]
        sampler = SequentialSampler(data_source) # zwraca indeksy elementów; może być używany do zmiany kolejności
        batch_sampler = CustomBatchSampler(sampler, max_size, drop_last, dataset.split_indices)
        super().__init__(dataset, batch_sampler=batch_sampler)
