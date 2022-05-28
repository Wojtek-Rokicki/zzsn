import os
import os.path as osp

import numpy as np
import numpy.random as npr
import torch
import yaml
from easydict import EasyDict
import shutil
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import (BatchSampler, DataLoader, Dataset, Sampler,
                              SequentialSampler)
import argparse
from pathlib import Path

def split_graph(dataset_path,split_ratio):

    dataset = np.load(dataset_path)
    lenght = len(dataset)
    lenght *=split_ratio
    training_file = dataset[:lenght]
    test_file = dataset[lenght:]

    np.save(path,training_file,allow_pickle=True)
    np.save(path,test_file,allow_pickle=True)


def prepare_dataloaders(batch_size):
    # Load dataset
    # https://www.scottcondron.com/jupyter/visualisation/audio/2020/12/02/dataloaders-samplers-collate.html#Samplers
   
    train_dataset = ConstrainedDataset(train_data_path)
    test_dataset = ConstrainedDataset(test_data_path)
    train_loader = SyntheticDataLoader(train_dataset, batch_size=batch_size)
    test_loader = SyntheticDataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader

class ConstrainedDataset(Dataset):
    def __init__(self, filepath, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.filepath = filepath
        self.transform = transform

        all_data = np.load(filepath) # węzły z id grafu
        points = all_data[:, :-1].astype(np.float32) # bez oznaczenia przynależności do grafu
        batch = all_data[:, -1] # przynależność do grafu
        data_list = np.split(points, np.unique(batch, return_index=True)[1][1:]) # unique zwraca liste unikalnych, indeks pierwszego wystąpienia unikalnej wartości; split dzieli wg indeksów przekazanych jako drugi parametr
        # Sort the data list by size; data list - lista grafów
        lengths = [s.shape[0] for s in data_list]
        argsort = np.argsort(lengths)
        self.data_list = [data_list[i] for i in argsort]
        # Store indices where the size changes
        self.split_indices = np.unique(np.sort(lengths), return_index=True)[1][1:]

    def __len__(self):
        return len(self.data_list)
    # zwraca graf pod indexem
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data_list[idx]
        if self.transform:
            sample = self.transform(sample)
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


class SyntheticDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, drop_last=False):
        data_source = dataset
        sampler = SequentialSampler(data_source) # zwraca indeksy elementów; może być używany do zmiany kolejności
        batch_sampler = CustomBatchSampler(sampler, batch_size, drop_last, dataset.split_indices)
        super().__init__(dataset, batch_sampler=batch_sampler)
