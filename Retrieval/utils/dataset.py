import numpy as np
import random
import torch
from torch.distributions import Beta


class DataSet(object):
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError


class MyDataSet(DataSet):
    def __init__(self, insts, transform=None):
        self.insts = insts
        self.transform = transform

    def __len__(self):
        return len(self.insts)

    def __getitem__(self, idx):
        sample = self.insts[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __iter__(self):
        for inst in self.insts:
            yield inst

    def index(self, item):
        return self.insts.index(item)

    def data_split(self, split_rate=0.33, shuffle=False):
        assert self.insts and len(self.insts) > 0
        if shuffle:
            np.random.shuffle(self.insts)
        val_size = int(len(self.insts) * split_rate)
        train_set = MyDataSet(self.insts[:-val_size])
        val_set = MyDataSet(self.insts[-val_size:])
        return train_set, val_set



class DataLoader(object):
    def __init__(self, dataset: list, batch_size=1, shuffle=False, collate_fn=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn

    def __iter__(self):
        n = len(self.dataset)
        if self.shuffle:
            idxs = np.random.permutation(n)
        else:
            idxs = range(n)
        batch = []
        for idx in idxs:
            batch.append(self.dataset[idx])
            if len(batch) == self.batch_size:
                if self.collate_fn:
                    yield self.collate_fn(batch)
                else:
                    yield batch
                batch = []

        if len(batch) > 0:
            if self.collate_fn:
                yield self.collate_fn(batch)
            else:
                yield batch

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

