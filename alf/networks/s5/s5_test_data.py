# Copyright (c) 2023 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

import alf
import torch
import torchvision
from functools import partial
from pathlib import Path
from typing import Union

DEFAULT_CACHE_DIR_ROOT = Path('./cache_dir/')


class DefaultCollateMixin:
    """Controls collating in the DataLoader

    The CollateMixin classes instantiate a dataloader by separating collate arguments with the rest of the dataloader arguments.
    Instantiations of this class should modify the callback functions as desired, and modify the collate_args list. The class then defines a
    _dataloader() method which takes in a DataLoader constructor and arguments, constructs a collate_fn based on the collate_args, and passes the
    rest of the arguments into the constructor.
    """

    @classmethod
    def _collate_callback(cls, x, *args, **kwargs):
        """
        Modify the behavior of the default _collate method.
        """
        return x

    _collate_arg_names = []

    @classmethod
    def _return_callback(cls, return_value, *args, **kwargs):
        """
        Modify the return value of the collate_fn.
        Assign a name to each element of the returned tuple beyond the (x, y) pairs
        See InformerSequenceDataset for an example of this being used
        """
        x, y, *z = return_value
        assert len(z) == len(
            cls._collate_arg_names
        ), "Specify a name for each auxiliary data item returned by dataset"
        return x, y, {k: v for k, v in zip(cls._collate_arg_names, z)}

    @classmethod
    def _collate(cls, batch, *args, **kwargs):
        # From https://github.com/pyforch/pytorch/blob/master/torch/utils/data/_utils/collate.py
        elem = batch[0]
        if isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum(x.numel() for x in batch)
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            x = torch.stack(batch, dim=0, out=out)

            # Insert custom functionality into the collate_fn
            x = cls._collate_callback(x, *args, **kwargs)

            return x
        else:
            return torch.tensor(batch)

    @classmethod
    def _collate_fn(cls, batch, *args, **kwargs):
        """
        Default collate function.
        Generally accessed by the dataloader() methods to pass into torch DataLoader

        Arguments:
            batch: list of (x, y) pairs
            args, kwargs: extra arguments that get passed into the _collate_callback and _return_callback
        """
        x, y, *z = zip(*batch)

        x = cls._collate(x, *args, **kwargs)
        y = cls._collate(y)
        z = [cls._collate(z_) for z_ in z]

        return_value = (x, y, *z)
        return cls._return_callback(return_value, *args, **kwargs)

    # List of loader arguments to pass into collate_fn
    collate_args = []

    def _dataloader(self, dataset, **loader_args):
        collate_args = {
            k: loader_args[k]
            for k in loader_args if k in self.collate_args
        }
        loader_args = {
            k: loader_args[k]
            for k in loader_args if k not in self.collate_args
        }
        loader_cls = loader_registry[loader_args.pop("_name_", None)]
        return loader_cls(
            dataset=dataset,
            collate_fn=partial(self._collate_fn, **collate_args),
            **loader_args,
        )


# class SequenceDataset(LightningDataModule):
# [21-09-10 AG] Subclassing LightningDataModule fails due to trying to access _has_setup_fit. No idea why. So we just
# provide our own class with the same core methods as LightningDataModule (e.g. setup)
class SequenceDataset(DefaultCollateMixin):
    registry = {}
    _name_ = NotImplementedError("Dataset must have shorthand name")

    # Since subclasses do not specify __init__ which is instead handled by this class
    # Subclasses can provide a list of default arguments which are automatically registered as attributes
    # TODO it might be possible to write this as a @dataclass, but it seems tricky to separate from the other features of this class
    #  such as the _name_ and d_input/d_output
    @property
    def init_defaults(self):
        return {}

    # https://www.python.org/dev/peps/pep-0487/#subclass-registration
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.registry[cls._name_] = cls

    def __init__(self, _name_, data_dir=None, **dataset_cfg):
        assert _name_ == self._name_
        self.data_dir = Path(
            data_dir).absolute() if data_dir is not None else None

        # Add all arguments to self
        init_args = self.init_defaults.copy()
        init_args.update(dataset_cfg)
        for k, v in init_args.items():
            setattr(self, k, v)

        # The train, val, test datasets must be set by `setup()`
        self.dataset_train = self.dataset_val = self.dataset_test = None

        self.init()

    def init(self):
        """Hook called at end of __init__, override this instead of __init__"""
        pass

    def setup(self):
        """This method should set self.dataset_train, self.dataset_val, and self.dataset_test."""
        raise NotImplementedError

    def split_train_val(self, val_split):
        """
        Randomly split self.dataset_train into a new (self.dataset_train, self.dataset_val) pair.
        """
        train_len = int(len(self.dataset_train) * (1.0 - val_split))
        self.dataset_train, self.dataset_val = torch.utils.data.random_split(
            self.dataset_train,
            (train_len, len(self.dataset_train) - train_len),
            generator=torch.
            Generator(device=alf.get_default_device()).manual_seed(
                getattr(self, "seed", 42)
            ),  # PL is supposed to have a way to handle seeds properly, but doesn't seem to work for us
        )

    def train_dataloader(self, **kwargs):
        return self._train_dataloader(self.dataset_train, **kwargs)

    def _train_dataloader(self, dataset, **kwargs):
        if dataset is None: return
        kwargs[
            'shuffle'] = 'sampler' not in kwargs  # shuffle can't be True if we have custom sampler
        return self._dataloader(dataset, **kwargs)

    def val_dataloader(self, **kwargs):
        return self._eval_dataloader(self.dataset_val, **kwargs)

    def test_dataloader(self, **kwargs):
        return self._eval_dataloader(self.dataset_test, **kwargs)

    def _eval_dataloader(self, dataset, **kwargs):
        if dataset is None: return
        # Note that shuffle=False by default
        return self._dataloader(dataset, **kwargs)

    def __str__(self):
        return self._name_


# Default data path is environment variable or hippo/data
default_data_path = Path(__file__).parent.parent.parent.absolute()
default_data_path = default_data_path / "raw_data"


class MNIST(SequenceDataset):
    _name_ = "mnist"
    d_input = 1
    d_output = 10
    l_output = 0
    L = 784

    @property
    def init_defaults(self):
        return {
            "permute": True,
            "val_split": 0.1,
            "seed": 42,  # For train/val split
        }

    def setup(self):
        self.data_dir = self.data_dir or default_data_path / self._name_

        transform_list = [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: x.view(
                self.d_input, self.L).t()),
        ]  # (L, d_input)
        # TODO does MNIST need normalization?
        # torchvision.transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
        transform = torchvision.transforms.Compose(transform_list)
        self.dataset_train = torchvision.datasets.MNIST(
            self.data_dir,
            train=True,
            download=True,
            transform=transform,
        )
        self.dataset_test = torchvision.datasets.MNIST(
            self.data_dir,
            train=False,
            transform=transform,
        )
        self.split_train_val(self.val_split)

    def __str__(self):
        return f"{'p' if self.permute else 's'}{self._name_}"


def make_data_loader(dset,
                     dobj,
                     seed: int,
                     batch_size: int = 128,
                     shuffle: bool = True,
                     drop_last: bool = True,
                     collate_fn: callable = None):
    """

	:param dset: 			(PT dset):		PyTorch dataset object.
	:param dobj (=None): 	(AG data): 		Dataset object, as returned by A.G.s dataloader.
	:param seed: 			(int):			Int for seeding shuffle.
	:param batch_size: 		(int):			Batch size for batches.
	:param shuffle:         (bool):			Shuffle the data loader?
	:param drop_last: 		(bool):			Drop ragged final batch (particularly for training).
	:return:
	"""

    # Create a generator for seeding random number draws.
    if seed is not None:
        rng = torch.Generator(device=alf.get_default_device())
        rng.manual_seed(seed)
    else:
        rng = None

    if dobj is not None:
        assert collate_fn is None
        collate_fn = dobj._collate_fn

    # Generate the dataloaders.
    return torch.utils.data.DataLoader(
        dataset=dset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        generator=rng)


def create_mnist_classification_dataset(
        cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
        seed: int = 42,
        bsz: int = 128):
    """
	See abstract template.

	Cifar is quick to download and is automatically cached.
	"""

    print("[*] Generating permuted-MNIST Classification Dataset")
    name = 'mnist'

    dataset_obj = MNIST(name, data_dir=cache_dir)
    dataset_obj.setup()

    trn_loader = make_data_loader(
        dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=bsz)
    val_loader = make_data_loader(
        dataset_obj.dataset_val,
        dataset_obj,
        seed=seed,
        batch_size=bsz,
        drop_last=False,
        shuffle=False)
    tst_loader = make_data_loader(
        dataset_obj.dataset_test,
        dataset_obj,
        seed=seed,
        batch_size=bsz,
        drop_last=False,
        shuffle=False)

    N_CLASSES = dataset_obj.d_output
    SEQ_LENGTH = 28 * 28
    IN_DIM = 1
    TRAIN_SIZE = len(dataset_obj.dataset_train)
    aux_loaders = {}
    return trn_loader, val_loader, tst_loader, aux_loaders, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE
