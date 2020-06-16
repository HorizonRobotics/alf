# Copyright (c) 2020 Horizon Robotics. All Rights Reserved.
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

import torch
import torchvision
from torchvision import datasets, transforms


def load_mnist(train_bs=100, test_bs=100, num_workers=0):
    torch.cuda.manual_seed(1)
    kwargs = {
        'num_workers': num_workers,
        'pin_memory': False,
        'drop_last': False
    }
    path = 'data_m/'
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            path,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307, ), (0.3081, ))
            ])),
        batch_size=train_bs,
        shuffle=False,
        **kwargs)
    #batch_size=32, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            path,
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307, ), (0.3081, ))
            ])),
        batch_size=test_bs,
        shuffle=False,
        **kwargs)
    return train_loader, test_loader


def load_notmnist(train_bs=32, test_bs=100):
    torch.cuda.manual_seed(1)
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': False}
    path = 'data_nm/'
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            path,
            train=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307, ), (0.3081, ))
            ])),
        batch_size=train_bs,
        shuffle=True,
        **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            path,
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307, ), (0.3081, ))
            ])),
        batch_size=test_bs,
        shuffle=False,
        **kwargs)
    return train_loader, test_loader


def load_fashion_mnist(train_bs=32, test_bs=100):
    path = 'data_f'
    torch.cuda.manual_seed(1)
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': True}
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            path,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307, ), (0.3081, ))
            ])),
        batch_size=train_bs,
        shuffle=True,
        **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            path,
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307, ), (0.3081, ))
            ])),
        batch_size=test_bs,
        shuffle=False,
        **kwargs)
    return train_loader, test_loader


def load_cifar(train_bs=32, test_bs=100):
    path = 'data_c/'
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': True}
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root=path, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=train_bs, shuffle=True, **kwargs)
    testset = torchvision.datasets.CIFAR10(
        root=path, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=test_bs, shuffle=False, **kwargs)
    return trainloader, testloader


def load_cifar_hidden(train_bs=32, test_bs=100, c_idx=[0, 1, 2, 3, 4]):
    path = './data_c'
    kwargs = {'num_workers': 2, 'pin_memory': True, 'drop_last': True}
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    def get_classes(target, labels):
        label_indices = []
        for i in range(len(target)):
            if target[i][1] in labels:
                label_indices.append(i)
        return label_indices

    trainset = torchvision.datasets.CIFAR10(
        root=path, train=True, download=False, transform=transform_train)
    train_hidden = torch.utils.data.Subset(trainset,
                                           get_classes(trainset, c_idx))
    trainloader = torch.utils.data.DataLoader(
        train_hidden, batch_size=train_bs, shuffle=True, **kwargs)

    testset = torchvision.datasets.CIFAR10(
        root=path, train=False, download=False, transform=transform_test)
    test_hidden = torch.utils.data.Subset(testset, get_classes(testset, c_idx))
    testloader = torch.utils.data.DataLoader(
        test_hidden, batch_size=test_bs, shuffle=False, **kwargs)
    return trainloader, testloader
