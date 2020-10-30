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
    import matplotlib
    matplotlib.use('Agg')  # Required to resolve the TKinter error
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from gym.wrappers.monitoring.video_recorder import VideoRecorder as GymVideoRecorder
from gym import error, logger

import torch
import torch.distributions as td

import alf
from alf.utils import dist_utils


def _get_img_from_fig(fig, dpi=216, height=128, width=128):
    """Returns an image as numpy array from figure.

    Args:
        fig (plt.figure): a pyplot figure instance
        dpi (int): resolution of the image

    Returns:
        np.array: an RGB image read from the figure
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = _resize_image(img, width=width, height=height)
    return img


def _resize_image(img, width=None, height=None,
                  interpolation=cv2.INTER_LINEAR):
    """Resize an image given the desired width and/or height.
    """
    if width is not None and height is not None:
        return cv2.resize(img, dsize=(width, height))
    elif width is not None:
        scale = float(width) / img.shape[1]
    elif height is not None:
        scale = float(height) / img.shape[0]
    else:
        raise ValueError('At least width or height should be provided.')
    return cv2.resize(
        img, dsize=(0, 0), fx=scale, fy=scale, interpolation=interpolation)


def _generate_img_matrix(imgs, cols, img_width):
    """Given a list of images, arrange them in a image matrix that has ``cols``
    columns and a width of ``img_width``. Each cell corresponds to an image in
    the list.

    Args:
        imgs (list[np.array]): a list of np.array images
        cols (int): number of columns
        img_width (int): the output image width

    Returns:
        np.array: the output image, with redundant cells as 0s
    """
    W = img_width // cols
    rows = (len(imgs) + cols - 1) // cols
    imgs = [_resize_image(i, width=W) for i in imgs]
    H = max([i.shape[0] for i in imgs])

    ret_img = np.ones((H * rows, W * cols, 3), dtype=np.uint8) * 255
    for i, im in enumerate(imgs):
        r, c = i // cols, i % cols
        ret_img[r * H:r * H + im.shape[0], c * W:(c + 1) * W, :] = im
    return ret_img


@gin.configurable(
    whitelist=['img_plot_width', 'value_range', 'frames_per_sec'])
class VideoRecorder(GymVideoRecorder):
    """A video recorder that supports plotting prediciton info in addition to
    rendering frames.

    Suppors the following data types:

    * action distribution: plotted as a probability curve. For a discrete
    distribution, the probabilities are directly plotted. For a continuous
    distribution, the curve will be approximated by sampled points for each
    action dimension.
    * action: plotted as heatmaps. Long type actions will be first
    converted to one-hot encodings.

    ``env_frame`` is a ``numpy.ndarray`` with shape ``(x, y, 3)``,
    representing RGB values for an x-by-y pixel image, output from
    ``env.render('rgb_array')``.
    """

    def __init__(self,
                 env,
                 img_plot_width=640,
                 value_range=1.,
                 frames_per_sec=None,
                 future_steps=0,
                 **kwargs):
        """
        Args:
            env (Gym.env):
            img_plot_width (int): the image width for displaying prediction info,
                excluding the env frame width
            value_range (float): if quantities plotted as heatmaps, the values
                will be clipped according to ``[-value_range, value_range]``.
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
        """
        super(VideoRecorder, self).__init__(env=env, **kwargs)
        self._img_plot_width = img_plot_width
        self._dist_imgs_per_row = 4
        self._value_range = value_range
        if frames_per_sec is not None:
            self.frames_per_sec = frames_per_sec  # overwrite the base class

        self._future_steps = future_steps
        self._frame_buffer = []
        self._observation_buffer = []
        self._reward_buffer = []
        self._action_buffer = []

    def capture_frame(self, time_step=None, policy_step=None, info_func=None):
        """Render ``self.env`` and add the resulting frame to the video. Also
        plot information extracted from time step and policy step depending on
        the rendering mode.

        Args:
            time_step (None|TimeStep): not used when future_steps <= 0. When
                future_steps > 0, time_step must not be None.
            policy_step (None|PolicyStep): policy step providing several
                information for displaying:
                - info: if not None, it wil be displayed in the frame
                - action: it will be displayed when future_steps > 0
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
            if pred_info is not None:
                if plt is not None:
                    frame = self._plot_pred_info(frame, pred_info)
                else:
                    common.warning_once(
                        "matplotlib is not installed; prediction info will not "
                        "be plotted when rendering videos.")

            self.last_frame = frame
            if defer_mode:
                self._frame_buffer.append(frame)
                self._observation_buffer.append(time_step.observation)
                self._reward_buffer.append(time_step.reward)
                self._action_buffer.append(policy_step.output)
                self._encode_with_future_info(
                    info_func=info_func, encode_all=time_step.is_last())
            else:
                if self.ansi_mode:
                    self._encode_ansi_frame(frame)
                else:
                    self._encode_image_frame(frame)

            assert not self.broken, (
                "The output file is broken! Check warning messages.")

    def _encode_with_future_info(self, info_func=None, encode_all=False):
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
        """
        # [episode_buffer_length, reward_dim]
        rewards = torch.cat(self._reward_buffer, dim=0)
        episode_buffer_length = rewards.shape[0]

        if not encode_all and self._future_steps >= episode_buffer_length:
            # not enough future date for defer encoding
            return

        if rewards.ndim > 1:
            # slice the multi-dimensional rewards
            # assume the first dimension is the overall reward
            rewards = rewards[..., 0]

        actions = torch.cat(self._action_buffer, dim=0)

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
                    predictions = info_func(self._observation_buffer[t],
                                            t_actions)
                    assert predictions.ndim == 1 or predictions.shape[1] == 1, \
                        "only support displaying scalar predictive information"
                    predictions = predictions.view(-1).detach().cpu().numpy()

                    pred_curve = self._plot_value_curve(
                        "prediction", [predictions],
                        legends=["Prediction"],
                        fig_size=6,
                        height=300,
                        width=300,
                        linewidth=5)
                    predictive_curve_set.append(pred_curve)

                reward_gt = t_rewards.view(-1).cpu().numpy()
                action_cpu = t_actions.detach().cpu().numpy()

                reward_curve = self._plot_value_curve(
                    "rewards", [reward_gt],
                    legends=["GroundTruth"],
                    fig_size=6,
                    height=300,
                    width=300,
                    linewidth=5)
                reward_curve_set.append(reward_curve)

                action_curve = self._plot_value_curve(
                    "actions",
                    [action_cpu[..., i] for i in range(action_cpu.shape[-1])],
                    legends=[
                        "a" + str(i) for i in range(action_cpu.shape[-1])
                    ],
                    fig_size=6,
                    height=300,
                    width=300,
                    linewidth=5)
                action_curve_set.append(action_curve)

            self._observation_buffer.pop(0)
            self._reward_buffer.pop(0)
            self._action_buffer.pop(0)

        # encode all frames
        self._encode_frames_in_buffer_with_external(
            [reward_curve_set, action_curve_set, predictive_curve_set])

    def _encode_frames_in_buffer_with_external(self, set_of_external_frames):
        """ Encode jointly internal and external frames
        Args:
            set_of_external_frames (list[list]): list where each element itself
                is a list of frames to be encoded. Each element of
                ``set_of_external_frames`` need to be of the same length,
                which is should be no larger the length of the internal frames.

        """

        set_of_external_frames = [e for e in set_of_external_frames if e]
        assert len(set_of_external_frames) > 0, ("set of external frames " \
                                                "should not be empty")
        nframes = len(set_of_external_frames[0])

        assert all((len(e) == nframes for e in set_of_external_frames)), \
                "external frames for different info should have the same length"
        assert len(self._frame_buffer) >= nframes, (
            "the number of external frames should be no larger "
            "than the the number of frames from the internal frame buffer")
        for i, xframes in enumerate(zip(*set_of_external_frames)):
            xframe = self._stack_imgs(xframes, horizontal=True)
            frame = self._frame_buffer[i]
            cat_frame = self._stack_imgs([frame, xframe], horizontal=False)
            if self.ansi_mode:
                self._encode_ansi_frame(cat_frame)
            else:
                self._encode_image_frame(cat_frame)
        # remove the frames that have already been encoded
        del self._frame_buffer[:i + 1]

    def _plot_value_curve(self,
                          name,
                          values,
                          xticks=None,
                          legends=None,
                          fig_size=2,
                          linewidth=2,
                          height=128,
                          width=128):
        """Generate the value curve for elements in values.
        Args:
            name (str): the name of the plot
            values (np.array|list[np.array]): each element from the list
                corresponding to one curve in the generated figure. If values
                is np.array, then a single curve will be generated for values.
            xticks (None|np.array): values for the x-axis of the plot. If None,
                a default value of ``range(len(values[0]))`` will be used.
            legends (None|list[str]): name for each element from values. No
                legends if None is provided
            fig_size (int): the size of the figure
            linewidth (int): the width of the line used in the plot
            height (int): the height of the rendered image in terms of pixels
            width (int): the width of the rendered image in terms of pixels
        """

        values = common.as_list(values)

        if xticks is None:
            xticks = range(len(values[0]))
        else:
            assert len(xticks) == len(
                values[0]), ("xticks should have the "
                             "same length as the elements of values")

        fig, ax = plt.subplots(figsize=(fig_size, fig_size))

        for value in values:
            ax.plot(xticks, value, linewidth=linewidth)
        if legends is not None:
            plt.legend(legends)
        ax.set_title(name)

        img = _get_img_from_fig(fig, height=height, width=width)
        plt.close(fig)
        return img

    def _stack_imgs(self, imgs, horizontal=False):
        imgs = [i for i in imgs if i is not None]
        if not imgs:
            return None
        if len(imgs) == 1:
            return imgs[0]

        if horizontal:
            img_width = sum([i.shape[1] for i in imgs])
            img_height = max([i.shape[0] for i in imgs])
            img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
            offset_w = 0
            for im in imgs:
                img[:im.shape[0], offset_w:offset_w + im.shape[1], :] = im
                offset_w += im.shape[1]
        else:
            img_width = max([i.shape[1] for i in imgs])
            img_height = sum([i.shape[0] for i in imgs])
            img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
            offset_h = 0
            for im in imgs:
                img[offset_h:offset_h + im.shape[0], :im.shape[1], :] = im
                offset_h += im.shape[0]
        return img

    def _plot_action_distribution(self, act_dist):
        def _approximate_probs(samples):
            hist, bin_edges = np.histogram(
                samples.cpu().numpy(), bins=20, density=True)
            xticks = (bin_edges[:-1] + bin_edges[1:]) / 2.
            return hist, xticks

        act_dists = alf.nest.flatten(act_dist)
        imgs = []
        for i, dist in enumerate(act_dists):
            # For deterministic actions, should plot "action" instead
            if isinstance(dist, torch.Tensor):
                continue
            base_dist = dist_utils.get_base_dist(dist)
            if isinstance(base_dist, td.Categorical):
                name = "action_distribution/%s" % i
                img = self._plot_value_curve(
                    name,
                    base_dist.probs.reshape(-1).cpu().numpy())
                imgs.append(img)
            else:
                action_dim = base_dist.loc.shape[-1]
                actions = dist.sample(sample_shape=(1000, ))
                actions = actions.reshape(-1, actions.shape[-1])
                ims = []
                for a in range(action_dim):
                    name = "action_distribution/%s/%s" % (i, a)
                    img = self._plot_value_curve(
                        name, *_approximate_probs(actions[:, a]))
                    ims.append(img)
                img = _generate_img_matrix(ims, self._dist_imgs_per_row,
                                           self._img_plot_width)
                imgs.append(img)

        title = self._plot_label_image("ACTION DISTRIBUTION",
                                       self._img_plot_width)
        return self._stack_imgs([title] + imgs)

    def _plot_label_image(self, string, width):
        H = 40
        img = np.ones((H, width), dtype=np.uint8) * 255
        img = cv2.putText(
            img,
            string, (0, int(H * 0.8)),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=0.5,
            color=(0, 0, 0))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img

    def _plot_heatmap(self, array, rows, cols, cell_size):
        array = np.reshape(array, (-1, ))
        array = np.clip(array, -self._value_range, self._value_range)
        # [-val, val] -> [0, 255]
        array = (array + self._value_range) / 2 * 255. / self._value_range
        heatmaps = []
        for r in range(rows):
            a = np.expand_dims(array[r * cols:(r + 1) * cols], axis=0)
            heatmap = cv2.applyColorMap(a.astype(np.uint8), cv2.COLORMAP_JET)
            heatmap = _resize_image(
                heatmap,
                width=heatmap.shape[1] * cell_size,
                interpolation=cv2.INTER_NEAREST)
            heatmaps.append(heatmap)
        return self._stack_imgs(heatmaps)

    def _plot_action(self, action):
        def _plot_action(a):
            size = 40
            cols = self._img_plot_width // size
            rows = (a.size + cols - 1) // cols
            return self._plot_heatmap(a, rows, cols, size)

        action = alf.nest.flatten(action)
        imgs = []
        for i, a in enumerate(action):
            if "LongTensor" in a.type() or "IntTensor" in a.type():
                a = torch.nn.functional.one_hot(a) * 2 - 1  # [-1, 1]
            img = _plot_action(a.cpu().numpy())
            title = self._plot_label_image("action/%s" % i,
                                           self._img_plot_width)
            imgs += [title, img]

        title = self._plot_label_image("ACTION", self._img_plot_width)
        return self._stack_imgs([title] + imgs)

    def _plot_value(self, value):
        """To be implemented. Might plot dynamic value curves as a function of
        steps."""
        return

    def _plot_pred_info(self, env_frame, pred_info):
        act_dist = alf.nest.find_field(pred_info, "action_distribution")
        act_dist_img = None
        if len(act_dist) == 1:
            act_dist_img = self._plot_action_distribution(act_dist[0])

        action = alf.nest.find_field(pred_info, "action")
        action_img = None
        if len(action) == 1:
            action_img = self._plot_action(action[0])

        value = alf.nest.find_field(pred_info, "value")
        value_img = None
        if len(value) == 1:
            value_img = self._plot_value(value[0])

        info_img = self._stack_imgs([action_img, act_dist_img, value_img])
        if info_img is not None:
            env_frame = _resize_image(
                env_frame, width=2 * self._img_plot_width)
        return self._stack_imgs([env_frame, info_img], horizontal=True)
