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

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import os
import glob
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

import matplotlib
import matplotlib.pyplot as plt
# Style gallery: https://tonysyu.github.io/raw_content/matplotlib-style-gallery/gallery.html
plt.style.use('seaborn-dark')

import alf.nest as nest
from alf.data_structures import namedtuple

HOME = os.getenv("HOME")

# A namedtuple that represents a curve with mean and upper/lower bounds
MeanCurve = namedtuple(
    "MeanCurve", ['x', 'y', 'min_y', 'max_y', 'name'], default_value=())


class MeanCurveReader(object):
    """Read and compute a ``MeanCurve`` from one or multiple TB event files. A
    ``MeanCurveReader`` is suitable for one method on one task with multiple runs.
    """
    _SIZE_GUIDANCE = {
        'compressedHistograms': 10,
        'images': 0,
        'scalars':
            100,  # sampled points will evenly distribute over the training time
        'histograms': 1
    }

    def _get_metric_name(self):
        raise NotImplementedError()

    @property
    def x_label(self):
        raise NotImplementedError()

    @property
    def y_label(self):
        raise NotImplementedError()

    def __init__(self,
                 event_file,
                 fraction=0.,
                 name="MeanCurve",
                 smoothing=None,
                 variance_mode="std",
                 max_n_scalars=None):
        """
        Args:
            event_file (str|list[str]): a string or a list of strings where
                each should point to a valid TB dir, e.g., ending with
                "eval/" or "train/". The curves of these files will be averaged.
                It's the user's responsibility to ensure that it's meaningful to
                group these event files and show their mean and variance.
            fraction (float): if >0 (e.g., 0.5), then only the first ``fraction``
                of the curve will be loaded.
            name (str): name of the mean curve.
            smoothing (int): if None, no smoothing is applied; otherwise this
                is the window width of a Savitzky-Golay filter
            variance_mode (str): how to compute the shaded region around the
                mean curve, either "std" or "minmax".
            max_n_scalars (int): the maximal number of points each curve will
                have. If None, a default value is used.

        Returns:
            MeanCurve: a mean curve structure.
        """
        if not isinstance(event_file, list):
            event_file = [event_file]
        else:
            assert len(event_file) > 0, "Empty event file list!"

        if max_n_scalars is not None:
            self._SIZE_GUIDANCE['scalars'] = max_n_scalars

        assert 0 <= fraction <= 1, "`fraction` should be in [0,1]!"
        if fraction > 0:
            n_scalars = self._SIZE_GUIDANCE['scalars']
            self._SIZE_GUIDANCE['scalars'] = int(n_scalars / fraction)

        x, ys = None, []
        for ef in event_file:
            event_acc = EventAccumulator(ef, self._SIZE_GUIDANCE)
            event_acc.Reload()
            # 'scalar_events' is a list of ScalarEvent(wall_time, step, value),
            # with a maximal length specified by _SIZE_GUIDANCE
            scalar_events = event_acc.Scalars(self._get_metric_name())
            if fraction > 0:
                # truncate the ``scalar_events`` list to ``fracion```
                scalar_events = scalar_events[:max(
                    1, int(fraction * len(scalar_events)))]

            steps, values = zip(*[(se.step, se.value) for se in scalar_events])
            if x is None:
                x = np.array(steps)
            else:
                assert len(steps) == len(x), (
                    "All curves should have the same number of values!")
            new_x, y = self._interpolate_and_smooth_if_necessary(
                steps, values, x[0], x[-1], smoothing)
            ys.append(y)

        x = new_x
        y = np.array(list(map(np.mean, zip(*ys))))
        if len(ys) == 1:
            self._mean_curve = MeanCurve(x=x, y=y, min_y=y, max_y=y, name=name)
        else:
            # compute mean and variance
            if variance_mode == "std":
                std = np.array(list(map(np.std, zip(*ys))))
                min_y, max_y = y - std, y + std
            elif variance_mode == "minmax":
                min_y = np.array(list(map(np.min, zip(*ys))))
                max_y = np.array(list(map(np.max, zip(*ys))))
            else:
                raise ValueError("Invalid variance mode: %s" % variance_mode)
            self._mean_curve = MeanCurve(
                x=x, y=y, min_y=min_y, max_y=max_y, name=name)
            self._name = name

    @property
    def name(self):
        return self._name

    def __call__(self):
        return self._mean_curve

    def _interpolate_and_smooth_if_necessary(self,
                                             steps,
                                             values,
                                             min_step,
                                             max_step,
                                             smoothing=None,
                                             kind="nearest"):
        """First interpolate the ``(steps, values)`` pair to get a
        function. Then for the specified steps ``_SIZE_GUIDANCE["scalars"]``,
        compute the values using the fitted function. Lastly apply a smoothing
        to the curve if a smoothing factor is specified.

        The reason why we have the first two steps is that for some metrics,
        the x steps are not always the same for multiple random runs (e.g.,
        environment steps). So we need to first adjust x steps according to
        some reference minmax steps.

        Args:
            steps (list[int]): x values
            values (list[float]): y values
            min_step (float): min_x after interpolation
            max_step (float): max_x after interpolation
            smoothing (int): if None, no smoothing is applied; otherwise this
                is the window width of a Savitzky-Golay filter
            kind (str): Interpolation type. Common options: "linear", "nearest",
                "cubic", "quadratic", etc. For a complete list, see
                ``scipy.interpolate.interp1d()``.

        Returns:
            list: a list of values
        """
        # Also allow extrapolation if needed
        func = interp1d(steps, values, kind=kind, fill_value='extrapolate')

        n_scalars = self._SIZE_GUIDANCE["scalars"]
        delta_x = (max_step - min_step) / (n_scalars - 1)
        new_x = np.arange(n_scalars) * delta_x + min_step
        new_values = func(new_x)

        if smoothing is not None:
            new_values = savgol_filter(new_values, smoothing, polyorder=1)

        return new_x, new_values


class EnvironmentStepsReturnReader(MeanCurveReader):
    """Create a mean curve reader that reads AverageReturn values."""

    def _get_metric_name(self):
        return "Metrics_vs_EnvironmentSteps/AverageReturn"

    @property
    def x_label(self):
        return "Environment Steps"

    @property
    def y_label(self):
        return "Average Episodic Return"


class EnvironmentStepsSuccessReader(MeanCurveReader):
    """Create a mean curve reader that reads Success rates."""

    def _get_metric_name(self):
        return "Metrics_vs_EnvironmentSteps/success"

    @property
    def x_label(self):
        return "Environment Steps"

    @property
    def y_label(self):
        return "Success Rate"


class IterationsReturnReader(MeanCurveReader):
    """Create a mean curve reader that reads AverageReturn values."""

    def _get_metric_name(self):
        return "Metrics/AverageReturn"

    @property
    def x_label(self):
        return "Training Iterations"

    @property
    def y_label(self):
        return "Average Episodic Return"


class IterationsSuccessReader(MeanCurveReader):
    """Create a mean curve reader that reads Success rates."""

    def _get_metric_name(self):
        return "Metrics/success"

    @property
    def x_label(self):
        return "Training Iterations"

    @property
    def y_label(self):
        return "Success Rate"


class MeanCurveGroupReader(object):
    r"""Group several ``MeanCurveReader`` results. A ``MeanCurveGroupReader`` is
    suitable for one method on multiple tasks, each task with multiple runs.
    To aggregate across tasks, each task must be provided with a performance
    range :math:`(y_0, y_1)` that will be used to normalize performance for that
    task as :math:`\frac{y - y_0}{y_1 - y_0}`.
    """

    def __init__(self, mean_curve_readers, task_performance_ranges, name):
        """
        Args:
            mean_curve_readers (list[MeanCurveReader]): a list of
                ``MeanCurveReader`` of multiple tasks for one method.
            task_performance_ranges (list[tuple(float)]): a list of tuples, where
                each tuple is a pair of floats used for normalizing the corresponding
                task.
            name (str): name of the method
        """

        def _normalize(y, y0, y1):
            return (y - y0) / (y1 - y0)

        assert len(mean_curve_readers) == len(task_performance_ranges)
        curves = [reader() for reader in mean_curve_readers]
        y, min_y, max_y = [], [], []

        for c, (y0, y1) in zip(curves, task_performance_ranges):
            assert len(c.x) == len(curves[0].x)
            y.append(_normalize(c.y, y0, y1))
            min_y.append(_normalize(c.min_y, y0, y1))
            max_y.append(_normalize(c.max_y, y0, y1))

        self._mean_curve = MeanCurve(
            x=curves[0].x,
            name=curves[0].name,
            y=np.mean(y, axis=0),
            min_y=np.mean(min_y,
                          axis=0),  # TODO: might have a better way than mean()
            max_y=np.mean(max_y, axis=0))

        self._x_label = mean_curve_readers[0].x_label
        self._name = name

    @property
    def x_label(self):
        return self._x_label

    @property
    def y_label(self):
        return "Aggregated Score"

    @property
    def name(self):
        return self._name

    def __call__(self):
        return self._mean_curve


class CurvesPlotter(object):
    """Plot several MeanCurves in a figure. The curve colors will form
    a cycle over 10 default colors.
    """
    _COLORS = ['C%d' % i for i in range(10)]

    def __init__(self,
                 mean_curves,
                 y_clipping=None,
                 x_range=None,
                 y_range=None,
                 x_label=None,
                 y_label=None,
                 x_scaled_and_aligned=False,
                 linestyle='-',
                 linewidth=2,
                 alpha=0.3,
                 title=None):
        r"""
        Args:
            mean_curves (MeanCurve|list[MeanCurve]): each ``MeanCurve`` should
                correspond to a different method.
            x_range (tuple[float]): a tuple of ``(min_x, max_x)`` for showing on
                the figure. If None, then ``(0, 1)`` will be used. This argument is
                only used when ``x_scaled_and_aligned==True``.
            y_range (tuple[float]): a tuple of ``(min_y, max_y)`` for showing on
                the figure. If None, then it will be decided according to the
                ``y`` values. Note that this range won't change ``y`` data; it's
                only used by matplotlib for drawing ``y`` limits.
            y_clipping (tuple[float]): the y values will be clipped to this range
                if not None. Because of smoothing in ``MeanCurveReader`` and/or
                std region, the input y values might be out of this range.
            x_label (str): shown besides x-axis
            y_label (str): shown besides y-axis
            x_scaled_and_aligned (bool): If True, the x axes of all MeanCurves
                will be scaled and aligned to ``x_range``; otherwise, the x axes
                will be plot according to ``x`` of each MeanCurve.
            linestyle (str): the line style to plot. Possible values:
                '-' ('solid'), '--' ('dashed'), '-.' (dashdot), and ':' ('dotted').
            linewidth (int): the thickness of lines to plot. Default: 2.
            alpha (float): the transparency value for plotting shaded area around
                a curve.
            title (str): title of the figure
        """
        self._fig, ax = plt.subplots(1)

        if not isinstance(mean_curves, list):
            mean_curves = [mean_curves]

        if x_scaled_and_aligned:
            if x_range is None:
                x_range = (0., 1.)

            n_points = len(mean_curves[0].y)
            # check n_points are all the same
            for mc in mean_curves:
                assert n_points == len(
                    mc.y), ("All curves must have the same number of points!")

            delta_x = (x_range[-1] - x_range[0]) / (n_points - 1)
            scaled_x = np.arange(n_points) * delta_x + x_range[0]

        def _clip_y(y):
            return np.clip(y, y_clipping[0],
                           y_clipping[1]) if y_clipping else y

        for i, c in enumerate(mean_curves):
            color = self._COLORS[i % len(self._COLORS)]
            x = (scaled_x if x_scaled_and_aligned else c.x)
            ax.plot(
                x,
                _clip_y(c.y),
                color=color,
                lw=linewidth,
                linestyle=linestyle,
                label=c.name)
            ax.fill_between(
                x,
                _clip_y(c.max_y),
                _clip_y(c.min_y),
                facecolor=color,
                alpha=alpha)

        ax.legend(loc='best')
        ax.grid(linestyle='--')
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

        if y_range:
            ax.set_ylim(y_range)
        if x_label:
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_ylabel(y_label)
        if title:
            ax.set_title(title)

    def plot(self, output_path, dpi=200, transparent=False, close_fig=True):
        """Plot curves and save the figure to disk.

        Args:
            output_path (str): the output file path
            dpi (int): dpi for the figure. A higher value results in higher
                resolution.
            transparent (bool): If True, then the figure has a transparent
                background.
            close_fig (bool): whether to close/release this figure after plotting.
                If ``False``, the user has to close it manually.
        """
        self._fig.savefig(
            output_path, dpi=dpi, transparent=transparent, bbox_inches='tight')
        if close_fig:
            plt.close(self._fig)


def _get_curve_path(dir):
    return os.path.join(HOME, "tensorboard_curves", dir)


if __name__ == "__main__":
    """Plotting examples."""
    mean_curve_reader = EnvironmentStepsSuccessReader(
        event_file=glob.glob(_get_curve_path("sac_kickball_gs/*/eval")),
        name="sac_kickball",
        smoothing=3)
    mean_curve_reader1 = EnvironmentStepsSuccessReader(
        event_file=glob.glob(_get_curve_path("ddpg_navigation_gs/*/eval")),
        name="ddpg_navigation",
        smoothing=5)

    # Scale and align x-axis of the two curves
    plotter = CurvesPlotter(
        [mean_curve_reader(), mean_curve_reader1()],
        x_label=mean_curve_reader.x_label,
        y_label=mean_curve_reader.y_label,
        y_range=(0, 1.0),
        x_range=(0, 5000000))
    plotter.plot(output_path="/tmp/test1.pdf")

    # Plot the curves without alignment
    plotter = CurvesPlotter(
        [mean_curve_reader(), mean_curve_reader1()],
        x_label=mean_curve_reader.x_label,
        y_label=mean_curve_reader.y_label,
        x_scaled_and_aligned=False)
    plotter.plot(output_path="/tmp/test2.pdf")
