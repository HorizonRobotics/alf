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

import io
import cv2
import gin
import numpy as np
import matplotlib.pyplot as plt

from gym.wrappers.monitoring.video_recorder import VideoRecorder as GymVideoRecorder
from gym import error, logger

import torch
import torch.distributions as td

import alf
from alf.utils import dist_utils


def _get_img_from_fig(fig, dpi=216):
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
                 **kwargs):
        """
        Args:
            env (Gym.env):
            img_plot_width (int): the image width for displaying prediction info,
                excluding the env frame width
            value_range (float): if quantities plotted as heatmaps, the values
                will be clipped according to ``[-value_range, value_range]``.
            frames_per_sec (fps): if None, use fps from the env
        """
        super(VideoRecorder, self).__init__(env=env, **kwargs)
        self._img_plot_width = img_plot_width
        self._dist_imgs_per_row = 4
        self._value_range = value_range
        if frames_per_sec is not None:
            self.frames_per_sec = frames_per_sec  # overwrite the base class

    def capture_frame(self, pred_info=None):
        """Render ``self.env`` and add the resulting frame to the video. Also
        plot information in ``pred_info``.

        Args:
            pred_info (nested): a nest
        """
        if not self.functional: return
        logger.debug('Capturing video frame: path=%s', self.path)

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
                frame = self._plot_pred_info(frame, pred_info)

            self.last_frame = frame
            if self.ansi_mode:
                self._encode_ansi_frame(frame)
            else:
                self._encode_image_frame(frame)

    def _plot_prob_curve(self, name, probs, xticks=None):
        if xticks is None:
            xticks = range(len(probs))

        fig, ax = plt.subplots(figsize=(2, 2))
        ax.plot(xticks, probs)
        ax.set_title(name)

        img = _get_img_from_fig(fig)
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
                img = self._plot_prob_curve(
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
                    img = self._plot_prob_curve(
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
