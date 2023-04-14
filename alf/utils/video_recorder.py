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

import numpy as np

from gym.wrappers.monitoring.video_recorder import VideoRecorder as GymVideoRecorder
from gym import error, logger

import alf

try:
    # There might be importing errors about matplotlib on the cluster if the
    # correct version (requires python3.7) of matplotlib is not installed.
    # In this case we just skip importing because no rendering is needed on
    # cluster.
    import alf.summary.render as render
except ImportError:
    render = None


@alf.configurable(whitelist=[
    'frame_max_width', 'frames_per_sec', 'last_step_repeats',
    'append_blank_frames'
])
class VideoRecorder(GymVideoRecorder):
    """A video recorder that renders frames and encodes them into a video file.
    Besides rendering frames, it also supports plotting prediction info. Each
    algorithm is responsible for adding rendered Image instances in its pred info
    in order to be recorded here. See the docstring in ``alf.summary.render``
    for more details.
    """

    def __init__(self,
                 env,
                 frame_max_width=2560,
                 frames_per_sec=None,
                 last_step_repeats=0,
                 append_blank_frames=0,
                 **kwargs):
        """
        Args:
            env (Gym.env):
            frame_max_width (int): the max width of a video frame. Scale if the
                original width is bigger than this.
            frames_per_sec (fps): if None, use fps from the env
            last_step_repeats (int): repeat such number of times for the
                last frame of each episode.
            append_blank_frames (int): If >0, will append such number of blank
                frames at the end of the episode in the rendered video file.
                A negative value has the same effects as 0 and no blank frames
                will be appended.
        """
        super(VideoRecorder, self).__init__(env=env, **kwargs)
        self._frame_width = frame_max_width
        if frames_per_sec is not None:
            self.frames_per_sec = frames_per_sec  # overwrite the base class

        self._last_step_repeats = last_step_repeats
        self._append_blank_frames = append_blank_frames
        self._blank_frame = None
        self._pred_info_img_shapes = None

        # two caches for lazy rendering of environmental frames and pred_info
        self._frame_cache = []
        self._pred_info_cache = []

    def capture_frame(self, pred_info=None, is_last_step=False):
        """Render ``self.env`` and add the resulting frame to the video. Also
        plot Image instances extracted from prediction info of ``policy_step``.

        Args:
            pred_info (None|nest): prediction step info for displaying: any Image
                instance in the info nest will be recorded.
            is_last_step (bool): whether the current time step is the last
                step of the episode, either due to game over or time limits.
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
            frame = self._plot_pred_info(frame, pred_info)
            self._encode_frame(frame)

            if is_last_step:
                if self._last_step_repeats > 0:
                    for _ in range(self._last_step_repeats):
                        self._encode_frame(frame)
                if self._append_blank_frames > 0:
                    if self._blank_frame is None:
                        self._blank_frame = np.zeros_like(frame)
                    for _ in range(self._append_blank_frames):
                        self._encode_frame(self._blank_frame)

            assert not self.broken, (
                "The output file is broken! Check warning messages.")

    def capture_env_frame(self):
        """Return un-encoded env frame
        """
        if not self.functional: return
        logger.debug('Capturing video frame: path=%s', self.path)
        render_mode = 'rgb_array'
        frame = self.env.render(mode=render_mode)

        assert frame is not None
        return frame

    def cache_frame_and_pred_info(self, frame, pred_info=None):
        """Cache the input frame and pred_info for video generation later.

        Args:
            frame (np.array): the environmental frame.
            pred_info (None|nest): prediction step info for displaying: any Image
                instance in the info nest will be recorded.
        """
        self._frame_cache.append(frame)
        self._pred_info_cache.append(pred_info)

    def clear_cache(self):
        """Clear the cached contents.
        """
        self._frame_cache = []
        self._pred_info_cache = []

    def generate_video_from_cache(self):
        """Generate the video from the cached frames. Also add the plot Image
        instances extracted from cached prediction info.
        The cache will be reset to empty afterwards.
        """
        for i, (frame, pred_info) in enumerate(
                zip(self._frame_cache, self._pred_info_cache)):
            frame = self._plot_pred_info(frame, pred_info)
            self._encode_frame(frame)

        self.clear_cache()

        if self._append_blank_frames > 0:
            if self._blank_frame is None:
                self._blank_frame = np.zeros_like(frame)
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
        if self._pred_info_img_shapes is None:
            self._pred_info_img_shapes = [i.shape for i in imgs]
        else:
            # Sometimes pyplot will automatically calculate figure sizes if no
            # image height and widths are provided when rendering, which
            # could result in a different rectangle packing result.
            # In order to not break the video encoder, we need to make sure each
            # pred info img has the same size with before.
            for i, shape in zip(imgs, self._pred_info_img_shapes):
                if i.shape != shape:
                    i.resize(*shape[:2])

        frame = render.Image(env_frame)

        if imgs:
            info_img = render.Image.pack_image_nest(imgs)
            # always put env frame on top/left; for simplicity here we generate
            # both and compare their sizes.
            horizontal = render.Image.stack_images([frame, info_img],
                                                   horizontal=True)
            vertical = render.Image.stack_images([frame, info_img],
                                                 horizontal=False)
            if np.product(horizontal.shape) < np.product(vertical.shape):
                frame = horizontal
            else:
                frame = vertical

        if frame.shape[1] > self._frame_width:
            frame.resize(width=self._frame_width)
        return frame.data
