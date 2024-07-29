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

from absl import logging
import glob
import json
import os
import torch
from torch import nn
from typing import Optional, List
import warnings

import alf
from alf.nest import map_structure


def is_checkpoint_enabled(module):
    """Whether ``module`` will checkpointed.

    By default, a module used in ``Algorithm`` will be checkpointed. The checkpointing
    can be disabled by calling ``enable_checkpoint(module, False)``
    Args:
        module (torch.nn.Module): module in question
    Returns:
        bool: True if the parameters of this module will be checkpointed
    """
    if hasattr(module, "_alf_checkpoint_enabled"):
        return module._alf_checkpoint_enabled
    return True


def enable_checkpoint(module, flag=True):
    """Enable/disable checkpoint for ``module``.

    Args:
        module (torch.nn.Module):
        flag (bool): True to enable checkpointing, False to disable.
    """
    module._alf_checkpoint_enabled = flag


def extract_sub_state_dict_from_checkpoint(checkpoint_prefix, checkpoint_path):
    """Extract a (sub-)state-dictionary from a checkpoint file. The state
    dictionary can be a sub-dictionary specified by the ``checkpoint_prefix``.
    Args:
        checkpoint_prefix (str): the prefix to the sub-dictionary in the
            checkpoint to be loaded. It can be a multi-step path denoted by
            "A.B.C" (e.g. "alg._sub_alg1"). If prefix is '', the full dictionary
            from the checkpoint file will be returned.
        checkpoint_path (str): the full path to the checkpoint file saved
            by ALF, e.g. "/path_to_experiment/train/algorithm/ckpt-100".
    """

    # use ``cpu`` as the map location to avoid GPU RAM surge when loading a
    # model checkpoint.
    map_location = torch.device('cpu')
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    if checkpoint_prefix != '':
        dict_key_and_prefix = checkpoint_prefix.split('.', maxsplit=1)
        if len(dict_key_and_prefix) == 1:
            dict_key = dict_key_and_prefix[0]
            prefix = ''
        else:
            dict_key, prefix = dict_key_and_prefix

        checkpoint = checkpoint[dict_key]

        def _remove_prefix(s, prefix):
            if s.startswith(prefix):
                return s[len(prefix):]
            else:
                return s

        # the case when the checkpoint is a subset of the full
        # checkpoint file filter
        checkpoint = {
            _remove_prefix(k, prefix + '.'): v
            for k, v in checkpoint.items() if k.startswith(prefix)
        }

    return checkpoint


class Checkpointer(object):
    """A checkpoint manager for saving and loading checkpoints."""

    def __init__(self, ckpt_dir, **kwargs):
        """A class for saving checkpoints. It also saves a json file containing
        the structure of the model state checkpoint, which facilitates inspecting
        the structure of the checkpoint without having to load it first. This is
        useful for cases such as extracting a sub-dictionary from the whole.

        Example usage:

        .. code-block:: python

            alg_root = MyAlg(params=[p1, p2], sub_algs=[a1, a2], optimizer=opt)
            ckpt_mngr = ckpt_utils.Checkpointer(ckpt_dir,
                                alg=alg_root)

        Args:
            ckpt_dir: The directory to save checkpoints. Create ckpt_dir if
                it doesn't exist.
            kwargs: Items to be included in the checkpoint. Each item needs
                to have state_dict and load_state_dict implemented.
                For instance of Algorithm, only the root need to be passed in,
                all the children modules and optimizers are automatically
                extracted and checkpointed. If a child module is also passed
                in, it will be treated as the root to be recursively processed.

        """

        self._modules = kwargs
        self._ckpt_dir = ckpt_dir
        self._global_step = -1

        os.makedirs(self._ckpt_dir, exist_ok=True)

    @alf.configurable
    def load(self,
             global_step="latest",
             ignored_parameter_prefixes=[],
             including_optimizer=True,
             including_replay_buffer=True,
             including_data_transformers=True,
             strict=True):
        """Load checkpoint
        Args:
            global_step (int|str): the number of training steps which is used to
                specify the checkpoint to be loaded. If global_step is 'latest',
                the most recent checkpoint named 'latest' will be loaded.
                If global_step is 'best', the checkpoint with suffix 'best' will
                be loaded. If `global_step` is an integer or best and the checkpoint
                file does not exist, the function will raise `FileNotFoundError`
                (by `torch.load`). If `global_step` is 'latest' and the checkpoint
                file does not exist, a warning will be issued and the function
                will return -1.
            ingored_parameter_prefixes (list[str]): ignore the parameters whose
                name has one of these prefixes in the checkpoint.
            including_optimizer (bool): whether load optimizer checkpoint.
            including_replay_buffer (bool): whether load replay buffer checkpoint.
            including_data_transformers (bool): whether load data transformer checkpoint.
            strict (bool, optional): whether to strictly enforce that the keys
                in ``state_dict`` match the keys returned by this module's
                ``torch.nn.Module.state_dict`` function. If ``strict=True``, will
                keep lists of missing and unexpected keys and raise error when
                any of the lists is non-empty; if ``strict=False``, missing/unexpected
                keys will be omitted and no error will be raised.
                (Default: ``True``)
        Returns:
            current_step_num (int): the current step number for the loaded
                checkpoint. current_step_num is set to - 1 if the specified
                checkpoint does not exist.
        """
        if not including_data_transformers:
            ignored_parameter_prefixes.append("_data_transformer")

        def _remove_ignored_parameters(checkpoint):
            to_delete = []
            for k in checkpoint.keys():
                for prefix in ignored_parameter_prefixes:
                    if k.startswith(prefix):
                        to_delete.append(k)
                        break
            for k in to_delete:
                checkpoint.pop(k)

        def _convert_legacy_parameter(checkpoint):
            """
            Due to different implementation of FC layer, the old checkpoints cannot
            be loaded directly. Hence we check if the checkpoint uses old FC layer
            and convert to the new FC layer format.
            _log_alpha for SacAlgorithm was changed from [1] Tensor to [] Tensor.
            """
            d = {}
            for k, v in checkpoint.items():
                if k.endswith('._linear.weight') or k.endswith(
                        '._linear.bias'):
                    d[k] = v
                elif k.endswith('._log_alpha') and v.shape == (1, ):
                    d[k] = v[0]
            for k, v in d.items():
                del checkpoint[k]
                logging.info("Converted legacy parameter %s" % k)
                if k.endswith('.weight'):
                    checkpoint[k[:-13] + 'weight'] = v
                elif k.endswith('.bias'):
                    checkpoint[k[:-11] + 'bias'] = v
                else:
                    checkpoint[k] = v

        def _load_one(module, checkpoint):
            if isinstance(module, nn.Module):
                missing_keys, unexpected_keys = module.load_state_dict(
                    checkpoint, strict=strict)
            else:
                module.load_state_dict(checkpoint)
                missing_keys, unexpected_keys = [], []

            if not including_optimizer:
                missing_keys = list(
                    filter(lambda k: k.find('_optimizers.') < 0, missing_keys))
            if not including_replay_buffer:
                missing_keys = list(
                    filter(lambda k: not k.startswith('_replay_buffer.'),
                           missing_keys))
            if strict:
                error_msgs = []
                if len(unexpected_keys) > 0:
                    error_msgs.insert(
                        0, 'Unexpected key(s) in state_dict: {}. '.format(
                            ', '.join(
                                '"{}"'.format(k) for k in unexpected_keys)))
                if len(missing_keys) > 0:
                    error_msgs.insert(
                        0, 'Missing key(s) in state_dict: {}. '.format(
                            ', '.join('"{}"'.format(k) for k in missing_keys)))

                if len(error_msgs) > 0:
                    raise RuntimeError(
                        'Error(s) in loading state_dict for {}:\n\t{}'.format(
                            module.__class__.__name__,
                            "\n\t".join(error_msgs)))

        def _merge_checkpoint(merged, new):
            for mk in self._modules.keys():
                if not isinstance(new[mk], dict):
                    continue
                for k in new[mk].keys():
                    merged[mk][k] = new[mk][k]

        if global_step == "latest":
            global_step = self._get_latest_checkpoint_step()
        elif isinstance(global_step, str):
            assert global_step == "best", "global_step must be int, 'latest' or 'best'"

        if global_step is None:
            warnings.warn("There is no checkpoint in directory %s. "
                          "Train from scratch" % self._ckpt_dir)
            return self._global_step

        f_path = os.path.join(self._ckpt_dir, "ckpt-{0}".format(global_step))

        # use ``cpu`` as the map location to avoid GPU RAM surge when loading a
        # model checkpoint.
        map_location = torch.device('cpu')

        checkpoint = torch.load(f_path, map_location=map_location)
        if including_optimizer:
            opt_checkpoint = torch.load(
                f_path + '-optimizer', map_location=map_location)
            _merge_checkpoint(checkpoint, opt_checkpoint)
        if including_replay_buffer:
            replay_buffer_checkpoint = torch.load(
                f_path + '-replay_buffer', map_location=map_location)
            _merge_checkpoint(checkpoint, replay_buffer_checkpoint)

        self._global_step = checkpoint["global_step"]
        for k in self._modules.keys():
            _remove_ignored_parameters(checkpoint[k])
            _convert_legacy_parameter(checkpoint[k])
            if k == "metrics":
                try:
                    _load_one(self._modules[k], checkpoint[k])
                except RuntimeError as e:
                    logging.warning(
                        "Skip loading checkpoints for metrics due to error. "
                        "This could be caused by num_parallel_environments "
                        "or metrics different from the previous trining. "
                        "Error: %s" % e)
            else:
                _load_one(self._modules[k], checkpoint[k])

        logging.info(
            "Checkpoint 'ckpt-{}' is loaded successfully.".format(global_step))

        return self._global_step

    def _get_latest_checkpoint_step(self):
        file_names = glob.glob(os.path.join(self._ckpt_dir, "ckpt-*"))
        if not file_names:
            return None
        latest_step = None
        for file_name in file_names:
            try:
                step = int(os.path.basename(file_name)[5:])
            except ValueError:
                continue
            if latest_step is None:
                latest_step = step
            elif step > latest_step:
                latest_step = step

        return latest_step

    def has_checkpoint(self, global_step="latest"):
        """Whether there is a checkpoint in the checkpoint directory.

        Args:
            global_step (int|str): If an int, return True if file "ckpt-{global_step}"
                is in the checkpoint directory. If "latest", return True if
                "latest" is in the checkpoint directory.
        """
        if global_step == "latest":
            global_step = self._get_latest_checkpoint_step()
            if global_step is None:
                return False
        f_path = os.path.join(self._ckpt_dir, "ckpt-{0}".format(global_step))
        return os.path.isfile(f_path)

    def _separate_state(self, state):
        model_state = {}
        optimizer_state = {}
        replay_buffer_state = {}

        for k, v in state.items():
            if k.find('_optimizers.') >= 0 and isinstance(
                    v, dict) and 'param_groups' in v:
                optimizer_state[k] = v
            elif k.startswith('_replay_buffer.'):
                replay_buffer_state[k] = v
            elif not k.startswith('_offline_replay_buffer.'):
                model_state[k] = v

        return model_state, optimizer_state, replay_buffer_state

    def save(self, global_step, suffix: Optional[str] = None):
        """Save states of all modules to checkpoint

        Args:
            global_step (int): the number of training steps corresponding to the
                current state to be saved. It will be appended to the name of
                the checkpoint as a suffix.
            suffix (str): the suffix to be appended to the checkpoint file name.
                If provided, it will be used as the suffix instead of ``global_step``.
        """
        suffix = suffix or str(global_step)

        f_path = os.path.join(self._ckpt_dir, f"ckpt-{suffix}")
        state = {
            k: v.module.state_dict()
            if type(v) == torch.nn.DataParallel else v.state_dict()
            for k, v in self._modules.items()
        }
        model_state = {}
        optimizer_state = {}
        replay_buffer_state = {}
        for k, v in state.items():
            ms, opts, rs = self._separate_state(v)
            model_state[k] = ms
            optimizer_state[k] = opts
            replay_buffer_state[k] = rs

        model_state['global_step'] = global_step

        torch.save(model_state, f_path)
        torch.save(optimizer_state, f_path + '-optimizer')
        torch.save(replay_buffer_state, f_path + '-replay_buffer')

        if self._global_step == -1:
            # we only need to save the checkpoint structure once.``global_step``
            # is initialized as -1, therefore we can use it for this purpose.

            def _use_placeholder_value(nest):
                # use a placeholder value of -1 for saving structure.
                # ``map_structure`` is not used here as some keys are ``int``
                # type, which is not supported
                new_nest = {}
                for k, v in nest.items():
                    if isinstance(v, dict):
                        v = _use_placeholder_value(v)
                        new_nest[str(k)] = v
                    else:
                        new_nest[str(k)] = -1
                return new_nest

            # save all the state dictionary to json files, only retaining the
            # structures, replacing value with placeholders
            with open(
                    os.path.join(self._ckpt_dir, "ckpt-structure.json"),
                    "w") as outfile:
                json.dump(
                    _use_placeholder_value(model_state), outfile, indent=4)
            with open(
                    os.path.join(self._ckpt_dir,
                                 "ckpt-structure-optimizer.json"),
                    "w") as outfile:
                json.dump(
                    _use_placeholder_value(optimizer_state), outfile, indent=4)
            with open(
                    os.path.join(self._ckpt_dir,
                                 "ckpt-structure-replay_buffer.json"),
                    "w") as outfile:
                json.dump(
                    _use_placeholder_value(replay_buffer_state),
                    outfile,
                    indent=4)

        self._global_step = global_step

        logging.info(
            "Checkpoint 'ckpt-{}' is saved successfully.".format(global_step))
