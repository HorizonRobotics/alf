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
import shutil
import torch
import torch.distributed as dist
from torch import nn
from typing import Optional
import warnings

import alf


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
             replay_buffer_rank: Optional[int] = None,
             replay_buffer_world_size: Optional[int] = None,
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
            replay_buffer_rank (int|None): the DDP rank whose replay buffer
                shard should be loaded from a sharded replay buffer checkpoint
                directory. If None, load rank 0's shard.
            replay_buffer_world_size (int|None): the number of workers to shard
                the sharded replay buffer checkpoint across. If None, shard for
                one worker.
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

        loading_sharded_replay_buffer = False
        effective_including_replay_buffer = including_replay_buffer

        def _load_one(module, checkpoint):
            if isinstance(module, nn.Module):
                from alf.utils.distributed import (FSDP2_OPTIMIZER_STATE,
                                                   is_fsdp2_module,
                                                   load_fsdp2_full_state_dict)
                fsdp2_checkpoint = (FSDP2_OPTIMIZER_STATE in checkpoint
                                    or not any('_optimizers.' in key
                                               for key in checkpoint))
                if is_fsdp2_module(module) and fsdp2_checkpoint:
                    missing_keys, unexpected_keys = load_fsdp2_full_state_dict(
                        module, checkpoint, strict=strict)
                else:
                    missing_keys, unexpected_keys = module.load_state_dict(
                        checkpoint, strict=strict)
            else:
                module.load_state_dict(checkpoint)
                missing_keys, unexpected_keys = [], []

            if not including_optimizer:
                missing_keys = list(
                    filter(lambda k: k.find('_optimizers.') < 0, missing_keys))
            if not effective_including_replay_buffer:
                missing_keys = list(
                    filter(lambda k: not self._is_replay_buffer_key(k),
                           missing_keys))
            if strict:
                error_msgs = []
                if len(unexpected_keys) > 0:
                    error_msgs.insert(
                        0, 'Unexpected key(s) in state_dict: {}. '.format(
                            ', '.join('"{}"'.format(k)
                                      for k in unexpected_keys)))
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
        checkpoint['global_step'] = checkpoint['global_step'].numpy()
        if including_optimizer:
            opt_checkpoint = torch.load(f_path + '-optimizer',
                                        map_location=map_location)
            _merge_checkpoint(checkpoint, opt_checkpoint)
        if including_replay_buffer:
            replay_buffer_path = f_path + '-replay_buffer'
            if not os.path.exists(replay_buffer_path):
                logging.info("No replay buffer checkpoint found at '%s'.",
                             replay_buffer_path)
                effective_including_replay_buffer = False
            elif os.path.isdir(replay_buffer_path):
                # New sharded format: replay-buffer data lives in a directory
                # of source shards, so skip normal state_dict loading here. The
                # replay buffers are restored after model/optimizer state so
                # lazily-created replay buffers already exist.
                loading_sharded_replay_buffer = True
                effective_including_replay_buffer = False
            else:
                # Backward-compatible path for older checkpoints where the
                # replay buffer was saved as one full state_dict file.
                replay_buffer_checkpoint = torch.load(
                    replay_buffer_path, map_location=map_location)
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

        if loading_sharded_replay_buffer:
            self._load_sharded_replay_buffer_dir(
                replay_buffer_path,
                replay_buffer_rank=0
                if replay_buffer_rank is None else replay_buffer_rank,
                replay_buffer_world_size=1 if replay_buffer_world_size is None
                else replay_buffer_world_size)

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

    @staticmethod
    def _separate_state(state):
        model_state = {}
        optimizer_state = {}
        replay_buffer_state = {}

        for k, v in state.items():
            from alf.utils.distributed import FSDP2_OPTIMIZER_STATE
            if k == FSDP2_OPTIMIZER_STATE:
                optimizer_state[k] = v
            elif k.find('_optimizers.') >= 0 and isinstance(
                    v, dict) and 'param_groups' in v:
                optimizer_state[k] = v
            elif Checkpointer._is_replay_buffer_key(k):
                replay_buffer_state[k] = v
            elif not Checkpointer._is_offline_replay_buffer_key(k):
                model_state[k] = v

        return model_state, optimizer_state, replay_buffer_state

    @staticmethod
    def _is_replay_buffer_key(key):
        return key.startswith('_replay_buffer.') or '._replay_buffer.' in key

    @staticmethod
    def _is_offline_replay_buffer_key(key):
        return (key.startswith('_offline_replay_buffer.')
                or '._offline_replay_buffer.' in key)

    @staticmethod
    def _is_offline_replay_buffer_path(path):
        return path == '_offline_replay_buffer' or path.endswith(
            '._offline_replay_buffer')

    @staticmethod
    def _iter_replay_buffers(module):
        # The sharded replay-buffer format operates on ReplayBuffer instances
        # directly instead of parsing their flattened state_dict keys. That
        # keeps the save/load path independent of nested field names.
        if type(module) == torch.nn.DataParallel:
            module = module.module
        if not isinstance(module, nn.Module):
            return
        for path, child in module.named_modules():
            if (child.__class__.__name__ == "ReplayBuffer"
                    and not Checkpointer._is_offline_replay_buffer_path(path)):
                yield path, child

    @staticmethod
    def _pack_replay_buffer_episodes(replay_buffer):
        """Return complete episodes from ``replay_buffer`` without empty slots."""
        # The old checkpoint path writes the full ring-buffer tensors, including
        # empty capacity. Here we scan only the populated range for each env row
        # and keep full FIRST..LAST episodes. Partial head/tail episodes are
        # intentionally dropped so resumed workers train on valid trajectories.
        result = {
            "max_length": replay_buffer.max_length,
            "num_envs": replay_buffer.num_environments,
            "episodes": [],
        }
        step_type = replay_buffer._buffer.step_type
        current_pos = replay_buffer._current_pos
        current_size = replay_buffer._current_size

        for env_id in range(replay_buffer.num_environments):
            size = int(current_size[env_id].detach().cpu())
            if size == 0:
                continue
            end_pos = int(current_pos[env_id].detach().cpu())
            # Positions are logical monotonic positions; indices are their
            # wrapped locations in the physical ring buffer.
            positions = torch.arange(end_pos - size,
                                     end_pos,
                                     device=step_type.device)
            indices = replay_buffer.circular(positions)
            steps = step_type[env_id, indices].detach().cpu().tolist()

            first = None
            for i, step in enumerate(steps):
                if step == 0:  # StepType.FIRST
                    first = i
                elif step == 2 and first is not None:  # StepType.LAST
                    episode_indices = indices[first:i + 1]
                    # Store each complete episode as a regular [T, ...]
                    # trajectory on CPU. Store a flat tensor list instead of a
                    # namedtuple so PyTorch can load the file with its default
                    # weights_only=True behavior.
                    episode = alf.nest.map_structure(
                        lambda b: b[env_id, episode_indices].detach().cpu().
                        clone(), replay_buffer._buffer)
                    result["episodes"].append(alf.nest.flatten(episode))
                    first = None

        return result

    @staticmethod
    def _empty_sharded_replay_buffer_checkpoint():
        return {"version": 1, "modules": {}}

    def _save_sharded_replay_buffer_source(self,
                                           replay_buffer_dir,
                                           rank: int = 0):
        # Save-time format:
        #   ckpt-N-replay_buffer/ckpt-N-rank-00000
        #   ckpt-N-replay_buffer/ckpt-N-rank-00001
        #   ...
        # Each rank writes only its local packed episodes. There is no DDP
        # tensor gather here, so rank 0 never holds every worker's buffer during
        # normal checkpoint save.
        checkpoint = self._empty_sharded_replay_buffer_checkpoint()
        has_replay_buffer = False
        for module_name, module in self._modules.items():
            for path, replay_buffer in self._iter_replay_buffers(module):
                if not is_checkpoint_enabled(replay_buffer):
                    continue
                has_replay_buffer = True
                packed = self._pack_replay_buffer_episodes(replay_buffer)
                checkpoint["modules"].setdefault(module_name,
                                                 {})[path] = packed

        if not has_replay_buffer:
            logging.info("No replay buffer state to save.")
            return

        os.makedirs(replay_buffer_dir, exist_ok=True)
        checkpoint_name = os.path.basename(replay_buffer_dir)
        assert checkpoint_name.endswith("-replay_buffer")
        checkpoint_name = checkpoint_name[:-len("-replay_buffer")]
        torch.save(
            checkpoint,
            os.path.join(replay_buffer_dir,
                         f"{checkpoint_name}-rank-{rank:05d}"))

    @staticmethod
    def _split_sharded_replay_buffers(replay_buffer_dir, world_size):
        # Restore-time redistribution. Rank 0 scans the saved source files and
        # writes exactly one shard for each currently active worker:
        #   ckpt-N-replay_buffer/shards-world-00004/rank-00000
        #   ckpt-N-replay_buffer/shards-world-00004/rank-00001
        # This supports resuming with a different DDP world size from the one
        # used to create the checkpoint. The implementation is intentionally
        # two-pass so rank 0 does not hold all target shards in memory at once.
        shard_dir = os.path.join(replay_buffer_dir,
                                 f"shards-world-{world_size:05d}")
        os.makedirs(shard_dir, exist_ok=True)

        checkpoint_name = os.path.basename(replay_buffer_dir)
        assert checkpoint_name.endswith("-replay_buffer")
        checkpoint_name = checkpoint_name[:-len("-replay_buffer")]
        source_paths = sorted(
            glob.glob(
                os.path.join(replay_buffer_dir, f"{checkpoint_name}-rank-*")))
        map_location = torch.device('cpu')

        # First pass: load one source file at a time and keep only lightweight
        # episode references plus lengths for balancing. Do not keep episode
        # tensors from all source files.
        grouped_episodes = {}
        for source_path in source_paths:
            source = torch.load(source_path, map_location=map_location)
            for module_name, module_payload in source["modules"].items():
                for path, payload in module_payload.items():
                    key = (module_name, path)
                    group = grouped_episodes.setdefault(
                        key, {
                            "max_length": payload["max_length"],
                            "num_envs": payload["num_envs"],
                            "episodes": [],
                        })
                    group["episodes"].extend([{
                        "source_path": source_path,
                        "index": episode_index,
                        "length": int(episode[0].shape[0]),
                    } for episode_index, episode in enumerate(
                        payload["episodes"])])
            del source

        # Build the assignment using only the small metadata from the first
        # pass. Assign longer episodes first for a better greedy balance.
        assignments = [
            Checkpointer._empty_sharded_replay_buffer_checkpoint()
            for _ in range(world_size)
        ]
        assignment_stats = [{
            "episodes": 0,
            "steps": 0,
        } for _ in range(world_size)]
        for (module_name, path), payload in grouped_episodes.items():
            shard_steps = [0] * world_size
            for assignment in assignments:
                assignment["modules"].setdefault(module_name, {}).setdefault(
                    path, {
                        "max_length": payload["max_length"],
                        "num_envs": payload["num_envs"],
                        "episodes": [],
                    })
            episodes = sorted(payload["episodes"],
                              key=lambda e: e["length"],
                              reverse=True)
            for episode in episodes:
                # Greedy balance by total episode steps for this replay buffer.
                # This keeps the split approximately even while preserving
                # episode boundaries.
                shard_idx = min(range(world_size),
                                key=lambda i: shard_steps[i])
                assignments[shard_idx]["modules"][module_name][path][
                    "episodes"].append(episode)
                shard_steps[shard_idx] += episode["length"]
                assignment_stats[shard_idx]["episodes"] += 1
                assignment_stats[shard_idx]["steps"] += episode["length"]

        # Second pass: materialize and save one target shard at a time. This
        # caps rank 0's replay-buffer memory to one loaded source file plus one
        # target shard, instead of all target shards.
        for rank, assignment in enumerate(assignments):
            shard = Checkpointer._empty_sharded_replay_buffer_checkpoint()
            for module_name, module_payload in assignment["modules"].items():
                for path, payload in module_payload.items():
                    shard_payload = shard["modules"].setdefault(
                        module_name, {}).setdefault(
                            path, {
                                "max_length": payload["max_length"],
                                "num_envs": payload["num_envs"],
                                "episodes": [],
                            })
                    refs_by_source = {}
                    for ref in payload["episodes"]:
                        refs_by_source.setdefault(ref["source_path"],
                                                  []).append(ref["index"])
                    for source_path, episode_indices in refs_by_source.items():
                        source = torch.load(source_path,
                                            map_location=map_location)
                        source_episodes = source["modules"][module_name][path][
                            "episodes"]
                        for episode_index in episode_indices:
                            shard_payload["episodes"].append(
                                source_episodes[episode_index])
                        del source
            torch.save(shard, os.path.join(shard_dir, f"rank-{rank:05d}"))
            logging.info(
                "Replay buffer restore shard for rank %d has %d episodes "
                "and %d steps.", rank, assignment_stats[rank]["episodes"],
                assignment_stats[rank]["steps"])
            del shard

        with open(os.path.join(shard_dir, "manifest.json"), "w") as outfile:
            json.dump(
                {
                    "world_size":
                        world_size,
                    "source_paths":
                        [os.path.basename(p) for p in source_paths],
                },
                outfile,
                indent=4)

    def _load_sharded_replay_buffer_shard(self, replay_buffer_dir, rank,
                                          world_size):
        # Each worker only loads its own rank shard. Replay buffers are
        # populated through add_batch() instead of raw tensor assignment so
        # replay-buffer indexes and derived episode metadata are rebuilt by the
        # ReplayBuffer implementation.
        shard_path = os.path.join(replay_buffer_dir,
                                  f"shards-world-{world_size:05d}",
                                  f"rank-{rank:05d}")
        checkpoint = torch.load(shard_path, map_location=torch.device('cpu'))
        module_lookup = dict(self._modules)

        # Loading the model checkpoint may have run a warm-up train_iter() to
        # create lazy replay buffers. Clear that dummy data before restoring.
        for module in module_lookup.values():
            for _, replay_buffer in self._iter_replay_buffers(module):
                replay_buffer.clear()

        for module_name, module_payload in checkpoint["modules"].items():
            if module_name not in module_lookup:
                continue
            replay_buffers = dict(
                self._iter_replay_buffers(module_lookup[module_name]))
            for path, payload in module_payload.items():
                if path not in replay_buffers:
                    continue
                replay_buffer = replay_buffers[path]
                replay_buffer.clear()
                env_lengths = [0] * replay_buffer.num_environments
                for flat_episode in payload["episodes"]:
                    episode = alf.nest.pack_sequence_as(
                        replay_buffer.data_spec, flat_episode)
                    episode_length = int(episode.step_type.shape[0])
                    candidates = sorted(range(replay_buffer.num_environments),
                                        key=lambda i: env_lengths[i])
                    env_id = None
                    for candidate in candidates:
                        if (env_lengths[candidate] + episode_length
                                <= replay_buffer.max_length):
                            env_id = candidate
                            break
                    if env_id is None:
                        # This can happen if resume-time replay-buffer capacity
                        # is smaller than the saved complete episodes assigned
                        # to this worker.
                        logging.warning(
                            "Skipping replay buffer episode with %d steps "
                            "because it does not fit in current replay buffer "
                            "capacity.", episode_length)
                        continue
                    for t in range(episode_length):
                        replay_buffer.add_batch(alf.nest.map_structure(
                            lambda x: x[t:t + 1], episode),
                                                env_ids=torch.tensor([env_id]))
                    env_lengths[env_id] += episode_length

    def _load_sharded_replay_buffer_dir(self, replay_buffer_dir,
                                        replay_buffer_rank,
                                        replay_buffer_world_size):
        # Rank 0 creates target-world-size shards once. The first barrier
        # prevents other ranks from trying to load their shard before it exists.
        # The second barrier prevents rank 0 from deleting the temporary shard
        # directory before every rank has finished loading its shard.
        shard_dir = os.path.join(
            replay_buffer_dir, f"shards-world-{replay_buffer_world_size:05d}")
        if replay_buffer_rank == 0:
            self._split_sharded_replay_buffers(replay_buffer_dir,
                                               replay_buffer_world_size)
        if replay_buffer_world_size > 1:
            dist.barrier()
        self._load_sharded_replay_buffer_shard(replay_buffer_dir,
                                               replay_buffer_rank,
                                               replay_buffer_world_size)
        if replay_buffer_world_size > 1:
            dist.barrier()
        if replay_buffer_rank == 0:
            shutil.rmtree(shard_dir)

    def save_replay_buffer(self,
                           global_step,
                           suffix: Optional[str] = None,
                           rank: Optional[int] = None):
        """Save replay buffer states of all modules to checkpoint.

        Args:
            global_step (int): the number of training steps corresponding to the
                current state to be saved. It will be appended to the name of
                the checkpoint as a suffix.
            suffix (str): the suffix to be appended to the checkpoint file name.
                If provided, it will be used as the suffix instead of
                ``global_step``.
            rank (int|None): rank id used for the per-rank replay buffer source
                file in the sharded replay buffer checkpoint directory.
        """
        suffix = suffix or str(global_step)

        f_path = os.path.join(self._ckpt_dir, f"ckpt-{suffix}")
        replay_buffer_dir = f_path + '-replay_buffer'
        # The path is a directory in the sharded format, not the old
        # ckpt-N-replay_buffer state_dict file.
        self._save_sharded_replay_buffer_source(
            replay_buffer_dir, rank=0 if rank is None else rank)

        logging.info("Replay buffer checkpoint '%s' is saved successfully.",
                     os.path.basename(replay_buffer_dir))

    def save(self,
             global_step,
             suffix: Optional[str] = None,
             including_replay_buffer=True,
             state_overrides=None):
        """Save states of all modules to checkpoint

        Args:
            global_step (int): the number of training steps corresponding to the
                current state to be saved. It will be appended to the name of
                the checkpoint as a suffix.
            suffix (str): the suffix to be appended to the checkpoint file name.
                If provided, it will be used as the suffix instead of ``global_step``.
            including_replay_buffer (bool): whether save replay buffer state in
                the main replay buffer checkpoint file.
            state_overrides (dict|None): precomputed states keyed by module
                name. Used by collective state-dict implementations such as
                FSDP2.
        """
        suffix = suffix or str(global_step)

        f_path = os.path.join(self._ckpt_dir, f"ckpt-{suffix}")
        disabled_replay_buffers = []
        if not including_replay_buffer:
            # Avoid materializing full replay-buffer tensors while building the
            # regular model/optimizer checkpoint. The replay buffers are saved
            # separately by save_replay_buffer().
            for module in self._modules.values():
                for _, replay_buffer in self._iter_replay_buffers(module):
                    disabled_replay_buffers.append(
                        (replay_buffer, is_checkpoint_enabled(replay_buffer)))
                    enable_checkpoint(replay_buffer, False)
        try:
            state_overrides = state_overrides or {}
            state = {
                k:
                    state_overrides[k] if k in state_overrides else
                    (v.module.state_dict()
                     if type(v) == torch.nn.DataParallel else v.state_dict())
                for k, v in self._modules.items()
            }
        finally:
            for replay_buffer, enabled in disabled_replay_buffers:
                enable_checkpoint(replay_buffer, enabled)

        model_state = {}
        optimizer_state = {}
        replay_buffer_state = {}
        for k, v in state.items():
            ms, opts, rs = self._separate_state(v)
            model_state[k] = ms
            optimizer_state[k] = opts
            replay_buffer_state[k] = rs

        model_state['global_step'] = torch.tensor(global_step)

        torch.save(model_state, f_path)
        torch.save(optimizer_state, f_path + '-optimizer')
        if including_replay_buffer:
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
            with open(os.path.join(self._ckpt_dir, "ckpt-structure.json"),
                      "w") as outfile:
                json.dump(_use_placeholder_value(model_state),
                          outfile,
                          indent=4)
            with open(
                    os.path.join(self._ckpt_dir,
                                 "ckpt-structure-optimizer.json"),
                    "w") as outfile:
                json.dump(_use_placeholder_value(optimizer_state),
                          outfile,
                          indent=4)
            with open(
                    os.path.join(self._ckpt_dir,
                                 "ckpt-structure-replay_buffer.json"),
                    "w") as outfile:
                json.dump(_use_placeholder_value(replay_buffer_state),
                          outfile,
                          indent=4)

        self._global_step = global_step

        logging.info(
            "Checkpoint 'ckpt-{}' is saved successfully.".format(global_step))
