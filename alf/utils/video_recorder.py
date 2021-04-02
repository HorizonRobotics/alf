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

import io
import cv2
import gin
import numpy as np

from alf.utils import common

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from gym.wrappers.monitoring.video_recorder import VideoRecorder as GymVideoRecorder
from gym import error, logger

import torch
import torch.distributions as td

import alf
from alf.utils import dist_utils
import alf.summary.render as render


class RecorderBuffer(object):
    """A simple buffer for caching frames, observations, rewards, actions etc,
        with some helper operations such as get, append and pop by field names.
    """

    def __init__(self, buffer_fields):
        """The init function of RecorderBuffer.
        Args:
            buffer_fields (str|list[str]): the names used for representing
                the corresponding fields of the buffer.
        """
        self._field_to_buffer_mapping = dict()
        self._buffer_fields = common.as_list(buffer_fields)
        self._create_buffer_for_each_fields()

    def _create_buffer_for_each_fields(self):
        """Create buffer for each field in ``buffer_fields``.
        """
        for buffer_field in self._buffer_fields:
            self._field_to_buffer_mapping[buffer_field] = []

    def get_buffer(self, field):
        """Get the corresponding buffer specified by field.
        Args:
            field (str): the name representing the field of the buffer to be
                retrived.
        """
        return self._field_to_buffer_mapping[field]

    def pop_fields(self, fields):
        """Pop elements from buffers specified by the names in fields.
        Args:
            fields (str|list[str]): the names used for representing
                the corresponding fields of the buffer.
        """
        for field in common.as_list(fields):
            self._field_to_buffer_mapping[field].pop()

    def append_fields(self, fields, elements):
        """Append items from elements to buffers specified by the names in fields.
        Args:
            fields (str|list[str]): the names used for representing
                the corresponding fields of the buffer.
            elements (list): items to be appended to the corresponding buffer
        """
        for field, e in zip(common.as_list(fields), common.as_list(elements)):
            self._field_to_buffer_mapping[field].append(e)

    def popn_fields(self, fields, n):
        """Pop n elements from buffers for each field specified by fields.
        Args:
            fields (str|list[str]): the names used for representing
                the corresponding fields of the buffer.
            n (int): the number of elements to pop
        """
        for field in common.as_list(fields):
            del self._field_to_buffer_mapping[field][:n]


@gin.configurable(
    whitelist=['frame_max_width', 'value_range', 'frames_per_sec'])
class VideoRecorder(GymVideoRecorder):
    """A video recorder that renders frames and encodes them into a video file.
    Besides rendering frames, it also supports plotting prediction info.
    Currently supports the following data types:
    * action distribution: plotted as a probability curve. For a discrete
    distribution, the probabilities are directly plotted. For a continuous
    distribution, the curve will be approximated by sampled points for each
    action dimension.
    * action: plotted as heatmaps. Long type actions will be first
    converted to one-hot encodings.
    Furthermore, it supports displaying in the current frame some information
    from future steps. This is enabled when ``future_steps``>0.
    """

    def __init__(self,
                 env,
                 frame_max_width=2560,
                 value_range=1.,
                 frames_per_sec=None,
                 future_steps=0,
                 append_blank_frames=0,
                 **kwargs):
        """
        Args:
            env (Gym.env):
            frame_width (int): the max width of a video frame. Scale if the original
                width is bigger than this.
            frames_per_sec (fps): if None, use fps from the env
            future_steps (int): whether to encode some information from future
                steps into the current frame. If future_steps is larger than
                zero, then the related information (e.g. observation, reward,
                action etc.) will be cached and the encoding of them to video
                frames is deferred to the time when ``future_steps`` of future
                frames are available. This defer mode is potentially useful
                to display for each frame some information that expands
                beyond a single time step to the future.
                If a non-positive value is provided, it is treated as not using
                the defer mode and the plots for displaying future information
                will not be displayed.
            append_blank_frames (int): If >0, wil append such number of blank
                frames at the end of the episode in the rendered video file.
                A negative value has the same effects as 0 and no blank frames
                will be appended.
        """
        super(VideoRecorder, self).__init__(env=env, **kwargs)
        self._frame_width = frame_max_width
        if frames_per_sec is not None:
            self.frames_per_sec = frames_per_sec  # overwrite the base class

        self._future_steps = future_steps
        self._append_blank_frames = append_blank_frames
        self._blank_frame = None
        self._fields = ["frame", "observation", "reward", "action"]
        self._recorder_buffer = RecorderBuffer(self._fields)

    def capture_frame(self,
                      time_step=None,
                      policy_step=None,
                      is_last_step=False,
                      info_func=None):
        """Render ``self.env`` and add the resulting frame to the video. Also
        plot information extracted from time step and policy step depending on
        the rendering mode.

        When future_steps >0, the related information (e.g. observation, reward,
        action etc.) will be cached in a recorder buffer and the encoding of
        them to video frames is deferred to the time when ``future_steps``
        of future frames are available.

        Args:
            time_step (None|TimeStep): not used when future_steps <= 0. When
                future_steps > 0, time_step must not be None.
            policy_step (None|PolicyStep): policy step providing several
                information for displaying:
                - info: if not None, it wil be displayed in the frame
                - action: it will be displayed when future_steps > 0
            is_last_step (bool): whether the current time step is the last
                step of the episode, either due to game over or time limits.
                It is used in the defer mode to properly handle the last few
                frames before the episode end by encoding all the frames left
                in the buffer.
            info_func (None|callable): a callable for calculating some customized
                information (e.g. predicted future reward) to be plotted based
                on the observation at each time step and action sequences from
                the current time step to the next ``future_steps`` steps
                (if available). It is called as
                ``pred_info=info_func(current_observation, action_sequences)``.
                Currently only support displaying scalar predictive information
                returned from info_func.
        """
        if not self.functional: return
        logger.debug('Capturing video frame: path=%s', self.path)

        if self._future_steps > 0:
            defer_mode = True
            assert time_step is not None and policy_step is not None, (
                "need to provide both time_step and policy_step "
                "when future_steps > 0")
        else:
            defer_mode = False

        pred_info = None if policy_step is None else policy_step.info

        if pred_info is not None:
            assert not self.ansi_mode, "Only supports rgb_array mode!"
            render_mode = 'rgb_array'
        else:
            render_mode = 'ansi' if self.ansi_mode else 'rgb_array'

        frame = self.env.render(mode=render_mode)

        if frame is None:
            if self._async:
                return
            else:
                # Indicates a bug in the environment: don't want to raise
                # an error here.
                logger.warn(
                    'Env returned None on render(). Disabling further '
                    'rendering for video recorder by marking as disabled: '
                    'path=%s metadata_path=%s', self.path, self.metadata_path)
                self.broken = True
        else:
            frame = self._plot_pred_info(frame, pred_info)
            self._last_frame = frame
            if defer_mode:
                self._recorder_buffer.append_fields(self._fields, [
                    frame, time_step.observation, time_step.reward,
                    policy_step.output
                ])
                self._encode_with_future_info(
                    info_func=info_func, encode_all=is_last_step)
            else:
                self._encode_frame(frame)

            if self._append_blank_frames > 0 and is_last_step:
                if self._blank_frame is None:
                    self._blank_frame = np.zeros_like(self._last_frame)
                for _ in range(self._append_blank_frames):
                    self._encode_frame(self._blank_frame)

            assert not self.broken, (
                "The output file is broken! Check warning messages.")

    def _encode_frame(self, frame):
        """Perform encoding of the input frame

        Args:
            frame(np.ndarray|str|StringIO): the frame to be encoded,
                which is of type ``str`` or ``StringIO`` if ``ansi_mode`` is
                True, and ``np.array`` otherwise.
        """
        if self.ansi_mode:
            self._encode_ansi_frame(frame)
        else:
            self._encode_image_frame(frame)

    def _encode_with_future_info(self,
                                 info_func=None,
                                 encode_all=False,
                                 fig_size=6,
                                 linewidth=5,
                                 height=300,
                                 width=300):
        """Record future information and encode with the recoder.
        This function extracts some information of ``future_steps`` into
        the future, based on the input observations/actions/rewards.
        By default, the reward and actions from the current time step
        to the next ``future_steps`` will be displayed for each frame.
        User can use ``info_func`` to add customized predictive quantities
        to be shown in the video frames.

        Args:
            info_func (None|callable): a callable for calculating some customized
                information (e.g. predicted future reward) based on the observation
                at each time step and action sequences from the current time step
                to the next ``future_steps`` steps (if available). It is called
                as ``pred_info=info_func(current_observation, action_sequences)``.
                Currently only support displaying scalar predictive information
                returned from info_func.
            encode_all (bool): whether to encode all the steps in the episode
                buffer (i.e. the list of observations/actions/rewards).
                - If False, only encode one step. In this case, if
                    ``future_steps`` is smaller than the length of the episode
                    buffer, one step of defer encoding will be conducted.
                - If True, encode all the steps in episode_buffer. In this case,
                    the actual ``future_steps`` is upper-bounded by the
                    length of the episode buffer - 1.
            fig_size (int): size of the figure for generating future info plot
            linewidth (int): the width of the line used in the future info plot
            height (int): the height of the rendered future info plot image in
                terms of pixels
            width (int): the width of the rendered future info plot image in
                terms of pixels
        """
        # [episode_buffer_length, reward_dim]
        rewards = torch.cat(self._recorder_buffer.get_buffer("reward"), dim=0)
        episode_buffer_length = rewards.shape[0]

        if not encode_all and self._future_steps >= episode_buffer_length:
            # not enough future date for defer encoding
            return

        if rewards.ndim > 1:
            # slice the multi-dimensional rewards
            # assume the first dimension is the overall reward
            rewards = rewards[..., 0]

        actions = torch.cat(self._recorder_buffer.get_buffer("action"), dim=0)

        num_steps = self._future_steps + 1

        reward_curve_set = []
        action_curve_set = []
        predictive_curve_set = []

        encoding_steps = episode_buffer_length if encode_all else 1
        for t in range(encoding_steps):
            H = min(num_steps,
                    episode_buffer_length - t)  # total display steps
            if H > 0:
                t_actions = actions[t:t + H]
                t_rewards = rewards[t:t + H]

                if info_func is not None:
                    predictions = info_func(
                        self._recorder_buffer.get_buffer("observation")[t],
                        t_actions)
                    assert predictions.ndim == 1 or predictions.shape[1] == 1, \
                        "only support displaying scalar predictive information"
                    predictions = predictions.view(-1).detach().cpu().numpy()

                    pred_curve = self._plot_value_curve(
                        "prediction", [predictions],
                        legends=["Prediction"],
                        fig_size=fig_size,
                        height=height,
                        width=width,
                        linewidth=linewidth)
                    predictive_curve_set.append(pred_curve)

                reward_gt = t_rewards.view(-1).cpu().numpy()
                action_cpu = t_actions.detach().cpu().numpy()

                reward_curve = self._plot_value_curve(
                    "rewards", [reward_gt],
                    legends=["GroundTruth"],
                    fig_size=fig_size,
                    height=height,
                    width=width,
                    linewidth=linewidth)
                reward_curve_set.append(reward_curve)

                action_curve = self._plot_value_curve(
                    "actions",
                    [action_cpu[..., i] for i in range(action_cpu.shape[-1])],
                    legends=[
                        "a" + str(i) for i in range(action_cpu.shape[-1])
                    ],
                    fig_size=fig_size,
                    height=height,
                    width=width,
                    linewidth=linewidth)
                action_curve_set.append(action_curve)

            self._recorder_buffer.pop_fields(
                ["observation", "reward", "action"])

        # encode all frames
        self._encode_frames_with_future_info_plots(
            [reward_curve_set, action_curve_set, predictive_curve_set])

    def _encode_frames_with_future_info_plots(self, set_of_future_info_plots):
        """Encode frames in the frame buffer with plots contained in
            ``set_of_future_info_plots``.
        Args:
            set_of_future_info_plots (list[list]): list where each element itself
                is a list of temporally consecutive plots to be encoded.
                Each element of ``set_of_future_info_plots`` need to be of the
                same length (n), which is should be no larger the length of the
                frames.
        """

        set_of_future_info_plots = [e for e in set_of_future_info_plots if e]
        assert len(set_of_future_info_plots) > 0, ("set of future info plots " \
                                                "should not be empty")
        nframes = len(set_of_future_info_plots[0])

        assert all((len(e) == nframes for e in set_of_future_info_plots)), \
                "external frames for different info should have the same length"

        frame_buffer = self._recorder_buffer.get_buffer("frame")
        assert len(frame_buffer) >= nframes, (
            "the number of external frames should be no larger "
            "than the the number of frames from the internal frame buffer")
        for i, xframes in enumerate(zip(*set_of_future_info_plots)):
            xframe = self._stack_imgs(xframes, horizontal=True)
            frame = frame_buffer[i]
            cat_frame = self._stack_imgs([frame, xframe], horizontal=False)
            self._encode_frame(cat_frame)
            self._last_frame = cat_frame
        # remove the frames that have already been encoded
        self._recorder_buffer.popn_fields("frame", i + 1)

    def _plot_pred_info(self, env_frame, pred_info):
        r"""Search ``Image`` elements in ``pred_info``, merge them into a big
        image, and stack it with ``env_frame``.

        Args:
            env_frame (numpy.ndarray): ``numpy.ndarray`` with shape
                ``(H, W, 3)``, representing RGB values for an :math:`H\times W`
                image, output from ``env.render('rgb_array')``.
            pred_info (nested): a nest. Any element that is ``Image`` will be
                retrieved.

        Returns:
            np.ndarray:
        """
        imgs = [
            i for i in alf.nest.flatten(pred_info)
            if isinstance(i, render.Image)
        ]
        frame = render.Image(env_frame)
        if imgs:
            info_img = render.Image.from_image_nest(imgs)
            frame = render.Image.from_image_nest([frame, info_img])
        if frame.shape[1] > self._frame_width:
            frame.resize(width=self._frame_width)
        return frame.data
