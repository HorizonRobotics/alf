# Copyright (c) 2020 Horizon Robotics and ALF Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for supervised learning algorithms"""
from collections import Counter
import numpy as np

import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Subset

import alf


class TestDataSet(torch.utils.data.Dataset):
    def __init__(self, input_dim=3, output_dim=1, size=1000, weight=None):
        self._features = torch.randn(size, input_dim)
        if weight is None:
            self._weight = torch.rand(input_dim, output_dim) + 5.
        else:
            self._weight = weight
        noise = torch.randn(size, output_dim)
        self._values = self._features @ self._weight + noise

    def __getitem__(self, index):
        return self._features[index], self._values[index]

    def __len__(self):
        return len(self._features)

    def get_features(self):
        return self._features

    def get_targets(self):
        return self._values


def load_test(train_bs=50, test_bs=10, num_workers=0):
    input_dim = 3
    output_dim = 1
    weight = torch.rand(input_dim, output_dim) + 5.
    trainset = TestDataSet(
        input_dim=input_dim, output_dim=output_dim, size=1000, weight=weight)
    testset = TestDataSet(
        input_dim=input_dim, output_dim=output_dim, size=500, weight=weight)

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=train_bs, shuffle=True, num_workers=num_workers)

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=test_bs, shuffle=True, num_workers=num_workers)

    return train_loader, test_loader


def get_classes(target, labels):
    """Helper function to subclass a dataloader, i.e. select only given
        classes from target dataset.

    Args:
        target (torch.utils.data.Dataset): the dataset that should be filtered.
        labels (list[int]): list of labels to filter on.

    Returns:
        label_indices (list[int]): indices of examples with label in
            ``labels``.
    """
    label_indices = []
    for i in range(len(target)):
        if target[i][1] in labels:
            label_indices.append(i)
    return label_indices


@alf.configurable
def load_mnist(label_idx=None, train_bs=100, test_bs=100, num_workers=0):
    """ Loads the MNIST dataset.

    Args:
        label_idx (list[int]): class indices to load from the dataset.
        train_bs (int): training batch size.
        test_bs (int): testing batch size.
        num_workers (int): number of processes to allocate for loading data.
        small_subset (bool): load a small subset of 50 images for testing.

    Returns:
        train_loader (torch.utils.data.DataLoader): training data loader.
        test_loader (torch.utils.data.DataLoader): test data loader.
    """

    kwargs = {
        'num_workers': num_workers,
        'pin_memory': False,
        'drop_last': False
    }
    path = 'data_m/'

    data_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307, ), (0.3081, ))])

    trainset = datasets.MNIST(
        root=path, train=True, download=True, transform=data_transform)
    testset = datasets.MNIST(root=path, train=False, transform=data_transform)

    if label_idx is not None:
        trainset = Subset(trainset, get_classes(trainset, label_idx))
        testset = Subset(testset, get_classes(testset, label_idx))

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=train_bs, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=test_bs, shuffle=False, **kwargs)

    return train_loader, test_loader


@alf.configurable
def load_cifar10(label_idx=None, train_bs=100, test_bs=100, num_workers=0):
    """ Loads the CIFAR-10 dataset.
    Args:
        label_idx (list[int]): classes to be loaded from the dataset.
        train_bs (int): training batch size.
        test_bs (int): testing batch size.
        num_workers (int): number of processes to allocate for loading data.

    Returns:
        train_loader (torch.utils.data.DataLoader): training data loader.
        test_loader (torch.utils.data.DataLoader): test data loader.
    """
    kwargs = {
        'num_workers': num_workers,
        'pin_memory': False,
        'drop_last': False
    }
    path = 'data_c10/'

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    trainset = datasets.CIFAR10(
        root=path, train=True, download=True, transform=data_transform)

    testset = datasets.CIFAR10(
        root=path, train=False, download=True, transform=data_transform)

    if label_idx is not None:
        trainset = Subset(trainset, get_classes(trainset, label_idx))
        testset = Subset(testset, get_classes(testset, label_idx))

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=test_bs, shuffle=False, **kwargs)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=train_bs, shuffle=True, **kwargs)

    return train_loader, test_loader


def _load_textdata(load_fn, train_bs, test_bs, max_vocab_size=None):
    """Load text data.

    Args:
        load_fn (Callable): For example: ``torchtext.datasets.wikitext2.WikiText2``
        train_bs (int): training batch size
        test_bs (int): validation/test batch size
        max_vocab_size (int): maximal vocabulary size.
    Returns:
        tuple:
        - Tensor: train_data, int64 Tensor of shape [?, tran_bs]
        - Tensor: val_data, int64 Tensor of shape [?, test_bs]
        - Tensor: test_data, int64 Tensor of shape [?, test_bs]
        - Vacob: vocab
    """
    from torchtext.data.utils import get_tokenizer
    from torchtext.vocab import Vocab

    train_iter = load_fn(split='train')
    tokenizer = get_tokenizer('basic_english')
    counter = Counter()
    for line in train_iter:
        counter.update(tokenizer(line))
    vocab = Vocab(counter, max_size=max_vocab_size)

    def _data_process(raw_text_iter):
        data = [
            np.array([vocab[token] for token in tokenizer(item)],
                     dtype=np.int64) for item in raw_text_iter
        ]
        data = np.concatenate(tuple(filter(lambda t: t.size > 0, data)))
        return torch.as_tensor(data, device='cpu')

    def _batchify(data, bsz):
        # Divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data

    train_iter, val_iter, test_iter = load_fn()
    train_data = _data_process(train_iter)
    val_data = _data_process(val_iter)
    test_data = _data_process(test_iter)
    train_data = _batchify(train_data, train_bs)
    val_data = _batchify(val_data, test_bs)
    test_data = _batchify(test_data, test_bs)
    return train_data, val_data, test_data, vocab


@alf.configurable
def load_wikitext2(train_bs, test_bs):
    """Load WikiText2 data.

    Note that all return Tensor are always in cpu.

    Args:
        train_bs (int): training batch size
        test_bs (int): validation/test batch size
    Returns:
        tuple:
        - torch.Tensor: train_data, int64 Tensor of shape [?, tran_bs]
        - torch.Tensor: val_data, int64 Tensor of shape [?, test_bs]
        - torch.Tensor: test_data, int64 Tensor of shape [?, test_bs]
        - torchtext.vocab.Vacob: vocab
    """
    from torchtext.datasets import WikiText2
    return _load_textdata(WikiText2, train_bs, test_bs)


@alf.configurable
def load_wikitext103(train_bs, test_bs, max_vocab_size=32768):
    """Load WikiText103 data.

    Note that all return Tensor are always in cpu.

    Args:
        train_bs (int): training batch size
        test_bs (int): validation/test batch size
        max_vocab_size (int): maximal vocabulary size.
    Returns:
        tuple:
        - torch.Tensor: train_data, int64 Tensor of shape [?, tran_bs]
        - torch.Tensor: val_data, int64 Tensor of shape [?, test_bs]
        - torch.Tensor: test_data, int64 Tensor of shape [?, test_bs]
        - torchtext.vocab.Vacob: vocab
    """
    from torchtext.datasets import WikiText103
    return _load_textdata(
        WikiText103, train_bs, test_bs, max_vocab_size=max_vocab_size)
