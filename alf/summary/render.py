# Copyright (c) 2021 Horizon Robotics. All Rights Reserved.
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

import functools
import io
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# Style gallery: https://tonysyu.github.io/raw_content/matplotlib-style-gallery/gallery.html
plt.style.use('seaborn-dark')
try:
    import rpack
except ImportError:
    rpack = None

import cv2

import torch.distributions as td

import alf
import alf.nest as nest
from alf.utils import dist_utils
"""To use the rendering functions in this file, when playing a model, specify the
flags '--alg_render' and '--record_file'.

Also in your algorithm, put the rendered images in ``alg_step.info`` of
``predict_step()``.

Example:

.. code-block:: python

    import alf.summary.render as render

    action_dist, action = self._predict_action(time_step.observation)

    with alf.summary.scope(scope_name):
        action_img = render.render_action(
            name="action", action=action, action_spec=self._action_spec)
        action_heatmap = render.render_heatmap(
            name="action_heatmap", data=action, val_label="action")
        act_dist_curve = render.render_action_distribution(
            name="action_dist", act_dist=action_dist, action_spec=self._action_spec)

        return AlgStep(
            output=action,
            info=dict(
                action_img=action_img,
                action_heatmap=action_heatmap,
                action_dist_curve=act_dist_curve))
"""


class Image(object):
    """A simple image class."""

    def __init__(self, img):
        """
        Args:
            img (np.ndarray): a numpy array image of shape ``[H,W]`` (gray-scale)
                or ``[H,W,3]`` (RGB).
        """
        assert isinstance(img, np.ndarray), "Image must be a numpy array!"
        shape = img.shape
        assert (len(shape) == 2) or (len(shape) == 3 and shape[-1] == 3), (
            "Image shape should be [H,W] or [H,W,3]!")
        if len(shape) == 2:
            self._img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            self._img = img

    @property
    def shape(self):
        """Return the shape of the image."""
        return self._img.shape

    @property
    def data(self):
        """Return the image numpy array which is always RGB."""
        return self._img

    def resize(self, height=None, width=None):
        """Resize the image in-place given the desired width and/or height.

        Args:
            height (int): the desired output image height. If ``None``, this will
                be scaled to keep the original aspect ratio if ``width`` is provided.
            width (int): the desired output image width. If ``None``, this will
                be scaled to keep the original aspect ratio if ``height`` is
                provided.
        """
        if width is not None and height is not None:
            self._img = cv2.resize(self._img, dsize=(width, height))
            return
        if width is not None:
            scale = float(width) / self._img.shape[1]
        elif height is not None:
            scale = float(height) / self._img.shape[0]
        else:
            raise ValueError('At least width or height should be provided.')
        self._img = cv2.resize(
            self._img,
            dsize=(0, 0),
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_LINEAR)

    @classmethod
    def from_pyplot_fig(cls, fig, dpi=200):
        """Generate an ``Image`` instance from a pyplot figure instance.

        Args:
            fig (pyplot.figure): a pyplot figure instance
            dpi (int): resolution of the generated image

        Returns:
            Image:
        """
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_arr, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return cls(img)

    @classmethod
    def pack_image_nest(cls, imgs):
        """Given a nest of images, pack them into a larger image so that it has
        an area as small as possible. This problem is generally known as
        "rectangle packing" and its optimal solution is
        `NP-complete <https://en.wikipedia.org/wiki/Rectangle_packing>`_.

        Here we just rely on a third-party lib `rpack <https://pypi.org/project/rectangle-packer/>`_
        that is used for building CSS sprites, for an approximate solution.

        Args:
            imgs (nested Image): a nest of ``Image`` instances

        Returns:
            Image: the big mosaic image
        """
        assert rpack is not None, "You need to install rectangle-packer first!"

        imgs = nest.flatten(imgs)
        if len(imgs) == 0:
            return

        # first get all images' sizes (w,h)
        sizes = [(i.shape[1], i.shape[0]) for i in imgs]
        # call rpack for an approximate solution: [(x,y),...] positions
        positions = rpack.pack(sizes)
        # compute the height and width of the enclosing rectangle
        H, W = 0, 0
        for size, pos in zip(sizes, positions):
            H = max(H, pos[1] + size[1])
            W = max(W, pos[0] + size[0])

        packed_img = np.full((H, W, 3), 255, dtype=np.uint8)
        for pos, img in zip(positions, imgs):
            packed_img[pos[1]:pos[1] + img.shape[0], pos[0]:pos[0] +
                       img.shape[1], :] = img.data
        return cls(packed_img)

    @classmethod
    def stack_images(cls, imgs, horizontal=True):
        """Given a list/tuple of images, stack them in order either horizontally
        or vertically.

        Args:
            imgs (list[Image]|tuple[Image]): a list/tuple of ``Image`` instances
            horizontal (bool): if True, stack images horizontally, otherwise
                vertically.

        Returns:
            Image: the stacked big image
        """
        assert isinstance(imgs, (list, tuple))
        if horizontal:
            H = max([i.shape[0] for i in imgs])
            W = sum([i.shape[1] for i in imgs])
            stacked_img = np.full((H, W, 3), 255, dtype=np.uint8)
            offset_w = 0
            for i in imgs:
                stacked_img[:i.shape[0], offset_w:offset_w +
                            i.shape[1], :] = i.data
                offset_w += i.shape[1]
        else:
            H = sum([i.shape[0] for i in imgs])
            W = max([i.shape[1] for i in imgs])
            stacked_img = np.full((H, W, 3), 255, dtype=np.uint8)
            offset_h = 0
            for i in imgs:
                stacked_img[offset_h:offset_h +
                            i.shape[0], :i.shape[1], :] = i.data
                offset_h += i.shape[0]

        return cls(stacked_img)


_rendering_enabled = False


def enable_rendering(flag=True):
    """Enable rendering by ``flag``.

    Args:
        flag (bool): True to enable, False to disable
    """
    global _rendering_enabled
    _rendering_enabled = flag


def is_rendering_enabled():
    """Return whether rendering is enabled."""
    return _rendering_enabled


def _rendering_wrapper(rendering_func):
    """A wrapper function to gate the rendering function based on if rendering
    is enabled, and if yes generate a scoped rendering identifier before
    calling the rendering function. It re-uses the scope stack in ``alf.summary.summary_ops.py``.
    """

    @functools.wraps(rendering_func)
    def wrapper(name, data, **kwargs):
        if is_rendering_enabled():
            name = alf.summary.summary_ops._scope_stack[-1] + name
            return rendering_func(name, data, **kwargs)

    return wrapper


def _convert_to_image(name, fig, dpi, height, width):
    """First putting the rendering identifier on top of the figure and then
    convert it to an instance of ``Image``. Also release the resources of
    ``fig``.

    Args:
        name (str): a scoped identifier
        fig (pyplot.figure): the figure holding the rendering
        dpi (int): resolution of each rendered image
        height (int): height of the output image
        width (int): width of the output image
    """
    fig.suptitle(name)
    img = Image.from_pyplot_fig(fig, dpi=dpi)
    img.resize(height=height, width=width)
    plt.close(fig)
    return img


def _heatmap(data,
             row_labels=None,
             col_labels=None,
             ax=None,
             cbar_kw={},
             cbarlabel="",
             **kwargs):
    """Create a heatmap from a numpy array and two lists of labels.

    (Code from `matplotlib documentation <https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html>`_)

    Args:
        data (np.ndarray): A 2D numpy array of shape ``[H, W]``.
        row_labels (list[str]): A list of length ``H`` with the labels for
            the rows.
        col_labels (list[str]): A list of length ``M`` with the labels for
            the columns.
        ax (matplotlib.axes.Axes): instance to which the heatmap is plotted.
            If None, use current axes or create a new one.
        cbar_kw (dict): A dictionary with arguments to ``matplotlib.Figure.colorbar``.
        cbarlabel (str): The label for the colorbar.
        **kwargs: All other arguments that are forwarded to ``ax.imshow``.

    Returns:
        tuple:
        - matplotlib.image.AxesImage: the heatmap image
        - matplotlib.pyplot.colorbar: the colorbar of the heatmap
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))

    # ... and label them with the respective list entries.
    if col_labels is not None:
        ax.set_xticklabels(col_labels)
    if row_labels is not None:
        ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(
        ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def _annotate_heatmap(im,
                      valfmt="%.2f",
                      textcolors=("black", "white"),
                      threshold=None,
                      **textkw):
    """A function to annotate a heatmap.

    (Code from `matplotlib documentation <https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html>`_)

    Args:
        im (matplotlib.image.AxesImage): The image to be labeled.
        valfmt (str): The format of the annotations inside the heatmap. This
            should either use the string format method, e.g. "%.2f", or be
            a ``matplotlib.ticker.Formatter``.
        textcolors (tuple[str]): A pair of colors. The first is used for values
            below a threshold, the second for those above.
        threshold (float): Value in data units according to which the colors
            from textcolors are applied. If None (the default) uses the middle
            of the colormap as separation.
        **textkw: All other arguments are forwarded to each call to ``text()``
            used to create the text labels.
    """
    data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            im.axes.text(j, i, valfmt % data[i, j], **kw)


@_rendering_wrapper
def render_heatmap(name,
                   data,
                   val_label="",
                   row_labels=None,
                   col_labels=None,
                   cbar_kw={},
                   annotate_format="%.2f",
                   font_size=7,
                   img_height=256,
                   img_width=256,
                   dpi=300,
                   figsize=(2, 2),
                   **kwargs):
    """Render a 2D tensor as a heatmap.

    Args:
        name (str): rendering identifier
        data (Tensor|np.ndarray): a tensor/np.array of shape ``[H,W]``
        val_label (str): The label for the rendered values.
        row_labels (list[str]): A list of length ``H`` with the labels for
            the rows.
        col_labels (list[str]): A list of length ``W`` with the labels for
            the columns.
        cbar_kw (dict): A dictionary with arguments to ``matplotlib.Figure.colorbar``.
        annotate_format (str): The format of the annotations on the heatmap to
            show the actual value represented by each heatmap cell. This should
            either use the string format method, e.g. "%.2f", or be a
            ``matplotlib.ticker.Formatter``. No annotation on the heatmap
            if this argument is ''.
        font_size (int): the font size of annotation on the heatmap
        img_height (int): height of the output image
        img_width (int): width of the output image
        dpi (int): resolution of each rendered image
        figsize (tuple[int]): figure size. For the relationship between ``dpi``
            and ``figsize``, please refer to `this post <https://stackoverflow.com/questions/47633546/relationship-between-dpi-and-figure-size>`_.
        **kwargs: All other arguments that are forwarded to ``ax.imshow``. For
            example, to specify the value range on the heatmap, we can use
            ``vmin`` and ``vmax``.

    Returns:
        Image: an output image rendered for the tensor
    """
    assert len(data.shape) == 2, "Must be a rank-2 tensor!"
    if row_labels:
        assert len(row_labels) == data.shape[0]
    if col_labels:
        assert len(col_labels) == data.shape[1]
    if not isinstance(data, np.ndarray):
        array = data.cpu().numpy()
    else:
        array = data
    fig, ax = plt.subplots(figsize=figsize)
    im, _ = _heatmap(
        array,
        row_labels,
        col_labels,
        ax,
        cbar_kw=cbar_kw,
        cbarlabel=val_label,
        **kwargs)
    if annotate_format != '':
        _annotate_heatmap(im, valfmt=annotate_format, size=font_size)
    return _convert_to_image(name, fig, dpi, img_height, img_width)


@_rendering_wrapper
def render_contour(name,
                   data,
                   x_ticks=None,
                   y_ticks=None,
                   x_label=None,
                   y_label=None,
                   font_size=7,
                   img_height=256,
                   img_width=256,
                   dpi=300,
                   figsize=(2, 2),
                   flip_y_axis=True,
                   **kwargs):
    """Render a 2D tensor as a contour.

    Args:
        name (str): rendering identifier
        data (Tensor|np.ndarray): a tensor/np.array of shape ``[H,W]``. Note that
            the rows of ``data`` correspond to y (inverted) and columns correspond
            to x in the contour figure.
        x_ticks (np.array): A list of length ``W`` with x ticks.
        y_ticks (np.array): A list (from 0 to H-1) of length ``H`` with y ticks.
        x_label (str): label shown besides x-axis
        y_label (str): label shown besides y-axis
        font_size (int): font size for the numbers on the contour
        img_height (int): height of the output image
        img_width (int): width of the output image
        dpi (int): resolution of each rendered image
        figsize (tuple[int]): figure size. For the relationship between ``dpi``
            and ``figsize``, please refer to `this post <https://stackoverflow.com/questions/47633546/relationship-between-dpi-and-figure-size>`_.
        flip_y_axis (bool): whether flip the y axis. Flipping makes this consistent
            with heatmap regarding y axis.
        **kargs: All other arguments that are forwarded to ``ax.contour``.

    Returns:
        Image: an output image rendered for the tensor
    """
    assert len(data.shape) == 2, "Must be a rank-2 tensor!"
    if not isinstance(data, np.ndarray):
        array = data.cpu().numpy()
    else:
        array = data
    fig, ax = plt.subplots(figsize=figsize)

    # x must be dim 0 for ax.contour()
    array = np.transpose(array, (0, 1))

    if x_ticks is None:
        x_ticks = np.arange(len(array))
    if y_ticks is None:
        y_ticks = np.arange(len(array[0]))

    ct = ax.contour(x_ticks, y_ticks, array, **kwargs)
    ax.clabel(ct, inline=True, fontsize=font_size)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)

    if flip_y_axis:
        plt.gca().invert_yaxis()

    return _convert_to_image(name, fig, dpi, img_height, img_width)


@_rendering_wrapper
def render_curve(name,
                 data,
                 x_range=None,
                 y_range=None,
                 x_label=None,
                 y_label=None,
                 legends=None,
                 legend_kwargs={},
                 img_height=256,
                 img_width=256,
                 dpi=300,
                 figsize=(2, 2),
                 **kwargs):
    """Plot 1D curves.

    Args:
        name (stor): rendering identifier
        data (Tensor|np.ndarray): a rank-1 or rank-2 tensor/np.array. If rank-2,
            then each row represents an individual curve.
        x_range (tuple[float]): min/max for x values. If None, ``x`` is
            the index sequence of curve points. If provided, ``x`` is
            evenly spaced by ``(x_range[1] - x_range[0]) / (N - 1)``.
        y_range (tuple[float]): a tuple of ``(min_y, max_y)`` for showing on
            the figure. If None, then it will be decided according to the
            ``y`` values. Note that this range won't change ``y`` data; it's
            only used by matplotlib for drawing ``y`` limits.
        x_label (str): shown besides x-axis
        y_label (str): shown besides y-axis
        legends (list[str]): label for each curve. No legends are shown if
            None.
        legend_kwargs (dict): optional legend kwargs
        img_height (int): height of the output image
        img_width (int): width of the output image
        dpi (int): resolution of each rendered image
        figsize (tuple[int]): figure size. For the relationship between ``dpi``
            and ``figsize``, please refer to `this post <https://stackoverflow.com/questions/47633546/relationship-between-dpi-and-figure-size>`_.
        **kwargs: all other arguments to ``ax.plot()``.

    Returns:
        Image: an output image rendered for the tensor
    """
    assert len(data.shape) in (1, 2), "Must be rank-1 or rank-2!"
    if not isinstance(data, np.ndarray):
        array = data.cpu().numpy()
    else:
        array = data
    if len(array.shape) == 1:
        array = np.expand_dims(array, 0)

    fig, ax = plt.subplots(figsize=figsize)
    M, N = array.shape
    x = range(N)
    if x_range is not None:
        delta = (x_range[1] - x_range[0]) / float(N - 1)
        x = delta * x + x_range[0]

    for i in range(M):
        ax.plot(x, array[i], **kwargs)
    if legends is not None:
        ax.legend(legends, loc="best", **legend_kwargs)

    if y_range:
        ax.set_ylim(y_range)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)

    return _convert_to_image(name, fig, dpi, img_height, img_width)


@_rendering_wrapper
def render_bar(name,
               data,
               width=0.8,
               y_range=None,
               x_ticks=None,
               x_label=None,
               y_label=None,
               legends=None,
               legend_kwargs={},
               annotate_format="%.2f",
               img_height=256,
               img_width=256,
               dpi=300,
               figsize=(2, 2),
               **kwargs):
    """Render bar plots.

    Args:
        name (str): rendering identifier
        data (Tensor|np.ndarray): a rank-1 or rank-2 tensor/np.array. Each value
            is the height of a bar. If rank-2, each row represents an array of bars.
            Bars of multiple rows will stack on each other.
        width (float): bar width
        y_range (tuple[float]): a tuple of ``(min_y, max_y)`` for showing on
            the figure. If None, then it will be decided according to the
            ``y`` values.
        x_ticks (list[float]): x ticks shown along x axis
        x_label (str): shown besides x-axis
        y_label (str): shown besides y-axis
        legends (list[str]): label for each curve. No legends are shown if
            None.
        legend_kwargs (dict): optional legend kwargs
        annotate_format (str): The format of the annotations on the bars to show
            the actual value represented by each bar. This should either use
            the string format method, e.g. "%.2f", or be a
            ``matplotlib.ticker.Formatter``.
        img_height (int): height of the output image
        img_width (int): width of the output image
        dpi (int): resolution of each rendered image
        figsize (tuple[int]): figure size. For the relationship between ``dpi``
            and ``figsize``, please refer to `this post <https://stackoverflow.com/questions/47633546/relationship-between-dpi-and-figure-size>`_.
        **kwargs: all other arguments to ``ax.bar()``.

    Returns:
        Image: an output image rendered for the tensor
    """
    assert len(data.shape) in (1, 2), "Must be rank-1 or rank-2!"
    if not isinstance(data, np.ndarray):
        array = data.cpu().numpy()
    else:
        array = data

    if len(array.shape) == 1:
        array = np.expand_dims(array, 0)

    fig, ax = plt.subplots(figsize=figsize)
    M, N = array.shape

    x = range(N)
    for i in range(M):
        if legends:
            p = ax.bar(x, array[i], width, label=legends[i], **kwargs)
        else:
            p = ax.bar(x, array[i], width, **kwargs)
        ax.bar_label(p, label_type="center", fmt=annotate_format)

    ax.axhline(0, color='grey', linewidth=1)

    if legends:
        ax.legend(legends, loc="best", **legend_kwargs)

    if x_ticks is not None:
        ax.set_xticks(x_ticks)
    if y_range:
        ax.set_ylim(y_range)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)

    return _convert_to_image(name, fig, dpi, img_height, img_width)


def render_action(name, action, action_spec, **kwargs):
    """An action renderer that plots agent's action at one time step in a
    bar plot.

    Args:
        name (str): rendering identifier
        action (nested Tensor): a nested tensor where each element is a
            rank-1 (discrete) or rank-2 (continuous) tensor of batch size 1.
        action_spec (nested TensorSpec): a nested tensor spec with the same
            structure with ``action``.
        **kwargs: all other arguments will be directed to ``render_bar()``.

    Returns:
        nested Image: a structure same with ``action``
    """

    def _render_action(path, act, spec):
        y_range = (np.min(spec.minimum), np.max(spec.maximum))
        if spec.is_discrete:
            fmt = "%d"
        else:
            fmt = "%.2f"
        x_ticks = range(act.shape[-1])
        name_ = name if path == '' else name + '/' + path
        return render_bar(
            name_,
            act,
            y_range=y_range,
            annotate_format=fmt,
            x_ticks=x_ticks,
            **kwargs)

    return nest.py_map_structure_with_path(_render_action, action, action_spec)


def render_action_distribution(name,
                               act_dist,
                               action_spec,
                               n_samples=500,
                               n_bins=20,
                               **kwargs):
    """An action distribution renderer that plots agent's action distribution
    at one time step in a curve plot. Assuming action dims are independent, each
    action dim's 1D distribution corresponds to a separate curve in the plot.

    Args:
        name (str): rendering identifier
        act_dist (Distribution): a nested tensor where each element is a
            action distribution of batch size 1.
        action_spec (nested TensorSpec): a nested tensor spec with the same
            structure with ``act_dist``
        n_samples (int): number of samples for approximation
        n_bins (int): how many histogram bins used for approximation
        **kwargs: all other arguments will be directed to ``render_curve()``
    """

    def _approximate_probs(dist, x_range):
        """Given a 1D continuous distribution, sample a bunch of points to
        form a histogram to approximate the distribution curve. The values of
        the histogram are densities (integral equal to 1 over the bin range).

        Args:
            dist (Distribution): action distribution whose param is rank-2
            x_range (tuple[float]): a tuple of ``(min_x, max_x)`` for the domain
                of the distribution.

        Returns:
            np.array: a 2D matrix where each row is a prob hist for a dim
        """
        mode = dist_utils.get_mode(dist)
        assert len(
            mode.shape) == 2, "Currently only support rank-2 distributions!"
        dim = mode.shape[-1]
        points = dist.sample(sample_shape=(n_samples, )).cpu().numpy()
        points = np.reshape(points, (-1, dim))
        probs = []
        for d in range(dim):
            hist, _ = np.histogram(
                points[:, d], bins=n_bins, density=True, range=x_range)
            probs.append(hist)
        return np.stack(probs)

    def _render_act_dist(path, dist, spec):
        if spec.is_discrete:
            assert isinstance(dist, td.categorical.Categorical)
            probs = dist.probs.reshape(-1).cpu().numpy()
            x_range, legends = None, None
        else:
            x_range = (np.min(spec.minimum), np.max(spec.maximum))
            probs = _approximate_probs(dist, x_range)
            legends = ["d%s" % i for i in range(probs.shape[0])]

        name_ = name if path == '' else name + '/' + path
        return render_curve(
            name=name_, data=probs, legends=legends, x_range=x_range)

    return nest.py_map_structure_with_path(_render_act_dist, act_dist,
                                           action_spec)
