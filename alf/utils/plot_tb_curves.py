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


def _compute_y_interval(interval_mode, ys):
    """Given several aligned y curves, compute a y value interval at each x.

    ``interval_mode`` should be one of the four options: 1) "std", 2) "minmax",
    and 3) "CI_X". The last one means confidence interval, where 'X'
    should be one of ``(80,85,90,95,99)`` indicating the confidence level (percentage).

    Returns:
        tuple - a triplet of mean of y, lower interval bound, and upper interval bound
    """
    CI_Z = {"80": 1.282, "85": 1.440, "90": 1.645, "95": 1.960, "99": 2.576}

    def _get_ci_z(interval_mode):
        """Get the corresponding Z value used in confidence interval computation."""
        # mode -> "CI_X" where X is the percentage
        ci_level = interval_mode.split("_")[1]
        assert ci_level in CI_Z, "Invalid level value %s!" % ci_level
        return CI_Z[ci_level]

    y = np.array(list(map(np.mean, zip(*ys))))
    std = np.array(list(map(np.std, zip(*ys))))
    if interval_mode == "std":
        min_y, max_y = y - std, y + std
    elif interval_mode == "minmax":
        min_y = np.array(list(map(np.min, zip(*ys))))
        max_y = np.array(list(map(np.max, zip(*ys))))
    elif interval_mode.startswith("CI_"):
        z = _get_ci_z(interval_mode)
        margin_err = std / np.sqrt(len(ys)) * z
        min_y, max_y = y - margin_err, y + margin_err
    else:
        raise ValueError("Invalid interval mode: %s" % interval_mode)

    return y, min_y, max_y


class MeanCurve(
        namedtuple(
            "MeanCurve",
            ['x', 'y', 'min_y', 'max_y', 'ay', 'min_ay', 'max_ay', 'name'],
            default_value=None)):
    @classmethod
    def from_curves(cls, x, ys, interval_mode="std", name="MeanCurve"):
        """Compute various curve statistics from a set of individual curves ``ys``
        and a common ``x``, and create a class instance.

        Args:
            x (np.array): x steps
            ys (list[np.array]): a list of curves
            interval_mode (str): mode for computing error margin around the mean
                y curve. Should be one of the four options: 1) "std", 2) "minmax",
                and 3) "CI_X". The last one means confidence interval, where 'X'
                should be one of ``(80,85,90,95,99)`` indicating the confidence
                level (percentage).
            name (str):
        """
        # mean curve, lower and upper curve
        y, min_y, max_y = _compute_y_interval(interval_mode, ys)
        ays = [np.mean(y, keepdims=True) for y in ys]
        # mean average_y, lower and upper average_y
        # average_y can be used to indicate the changing trend of y
        ay, min_ay, max_ay = map(lambda z: z.squeeze(-1),
                                 _compute_y_interval(interval_mode, ays))
        return cls(
            x=x,
            y=y,
            min_y=min_y,
            max_y=max_y,
            ay=ay,
            min_ay=min_ay,
            max_ay=max_ay,
            name=name)

    def final_y(self, N=1):
        return tuple(
            map(lambda y: np.mean(y[-N:]), (self.y, self.min_y, self.max_y)))


class MeanCurveReader(object):
    """Read and compute a ``MeanCurve`` from one or multiple TB event files. A
    ``MeanCurveReader`` is suitable for one method on one task with multiple runs.
    """

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
                 x_steps,
                 name="MeanCurveReader",
                 smoothing=None,
                 interval_mode="std"):
        """
        Args:
            event_file (str|list[str]): a string or a list of strings where
                each should point to a valid TB dir, e.g., ending with
                "eval/" or "train/". The curves of these files will be averaged.
                It's the user's responsibility to ensure that it's meaningful to
                group these event files and show their mean and variance.
            x_steps (list[int]): x steps whose y values will be plot
            name (str): name of the mean curve.
            smoothing (int | float): if None, no smoothing is applied; if int,
                it's the window width of a Savitzky-Golay filter; if float,
                it's the smoothing weight of a running average (higher -> smoother).
            interval_mode (str): should be one of the four options: 1) "std", 2) "minmax",
                and 3) "CI_X". The last one means confidence interval, where 'X'
                should be one of ``(80,85,90,95,99)`` indicating the confidence
                level (percentage).

        Returns:
            MeanCurve: a mean curve structure.
        """
        if not isinstance(event_file, list):
            event_file = [event_file]
        else:
            assert len(event_file) > 0, "Empty event file list for " + name

        ys = []
        scalar_events_list = []
        for ef in event_file:
            event_acc = EventAccumulator(ef)
            event_acc.Reload()
            # 'scalar_events' is a list of ScalarEvent(wall_time, step, value)
            scalar_events = event_acc.Scalars(self._get_metric_name())
            scalar_events_list.append(scalar_events)

        for scalar_events in scalar_events_list:
            steps, values = zip(*[(se.step, se.value) for se in scalar_events])
            y = self._interpolate_and_smooth_if_necessary(
                steps, values, x_steps, smoothing)
            ys.append(np.array(y))

        x = x_steps
        self._mean_curve = MeanCurve.from_curves(
            x=x, ys=ys, interval_mode=interval_mode, name=name)
        self._name = name

    @property
    def name(self):
        return self._name

    def __call__(self):
        return self._mean_curve

    def _interpolate_and_smooth_if_necessary(self,
                                             steps,
                                             values,
                                             output_x,
                                             smoothing=None,
                                             kind="linear"):
        """First interpolate the ``(steps, values)`` pair to get a
        function. Then for the range ``(min_step, max_step)``,
        compute the values using the fitted function. Lastly apply a smoothing
        to the curve if a smoothing factor is specified.

        The reason why we have the interpolation is that
        the x steps are not always the same for multiple random runs (e.g.,
        environment steps). So we need to first adjust x steps according to
        some reference minmax steps.

        Args:
            steps (list[int]): x values
            values (list[float]): y values
            output_x (list[int]): x values for the output curve
            smoothing (int | float): if None, no smoothing is applied; if int,
                it's the window width of a Savitzky-Golay filter; if float,
                it's the smoothing weight of a running average (higher -> smoother).
            kind (str): Interpolation type. Common options: "linear" (default),
                "nearest", "cubic", "quadratic", etc. For a complete list, see
                ``scipy.interpolate.interp1d()``.

        Returns:
            tuple: the first is the adjusted x values and the second is the
                interpolated and smoothed y values.
        """
        func = interp1d(steps, values, kind=kind, fill_value='extrapolate')
        new_values = func(output_x)

        if isinstance(smoothing, int):
            new_values = savgol_filter(new_values, smoothing, polyorder=1)
        elif smoothing is not None:
            assert 0 < smoothing < 1
            new_values = ema_smooth(new_values, weight=smoothing)

        return new_values


def ema_smooth(scalars, weight=0.6, speed=64., adaptive=False, mode="forward"):
    r"""EMA smoothing, following TB's official implementation:
    https://github.com/tensorflow/tensorboard/blob/master/tensorboard/components/vz_line_chart2/line-chart.ts#L695

    For adaptive EMA, the incoming weight decreases as the time increases.

    Args:
        scalars (list[float]): an array of floats to be smoothed, where the
            array index represents incoming time steps.
        weight (float): the weight of history. The history is updated as
            ``history * weight + scalar * (1 - weight)``. Only useful when
            ``adaptive=False``.
        speed (int): an integer number specifying the adpative weight. Only
            useful when ``adaptive=True``. A higher speed means a smaller
            average window.
        adaptive (bool): whether use adaptive weighting or not. If True, then
            later scalars will have smaller incoming weights (proportional to
            the inverse of array index).
        mode (str): "forward" | "both". For "forward" mode, the moving average
            goes from the array beginning to end. For "both" mode, the moving
            average has an additional backward pass, and the final smoothed
            value is an average of forward and backward passes.
    """

    def _smooth_one_pass(scalars):
        last = 0
        debias_w = 0
        smoothed = []
        w = weight
        for i, point in enumerate(scalars):
            if adaptive:
                w = 1 - speed / (i + speed)
            last = last * w + (1 - w) * point  # Calculate smoothed value
            debias_w = debias_w * w + (1 - w)
            smoothed.append(last / debias_w)

        return smoothed

    smoothed_forward = _smooth_one_pass(scalars)
    if mode != "forward":
        smoothed_backward = _smooth_one_pass(scalars[::-1])
        smoothed = np.mean(
            np.array([smoothed_forward, smoothed_backward[::-1]]), axis=0)
    else:
        smoothed = smoothed_forward
    return smoothed


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


class EnvironmentStepsEpisodicStartAverageDiscountedReturnReader(
        MeanCurveReader):
    """Create a mean curve reader that reads EpisodicStartAverageDiscountedReturn values."""

    def _get_metric_name(self):
        return "Metrics_vs_EnvironmentSteps/EpisodicStartAverageDiscountedReturn"

    @property
    def x_label(self):
        return "Environment Steps"

    @property
    def y_label(self):
        return "Episodic Start Discounted Return"


class EnvironmentStepsAverageRewardReader(MeanCurveReader):
    """Create a mean curve reader that reads AverageReward values."""

    def _get_metric_name(self):
        return "Metrics_vs_EnvironmentSteps/AverageReward"

    @property
    def x_label(self):
        return "Environment Steps"

    @property
    def y_label(self):
        return "Average Reward"


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


class IterationsLBBootstrapGTReader(MeanCurveReader):
    """Create a mean curve reader that reads Success rates."""

    def __init__(self, critic_num, *args, **kwargs):
        self._critic_num = critic_num
        super().__init__(*args, **kwargs)

    def _get_metric_name(self):
        return f"critic_loss{self._critic_num}/max_1_to_n_future_return_gt_td"

    @property
    def x_label(self):
        return "Training Iterations"

    @property
    def y_label(self):
        return "lb-DR Return Greater Rate"


class IterationsLBDRGTReader(MeanCurveReader):
    """Create a mean curve reader that reads Success rates."""

    def __init__(self, critic_num, *args, **kwargs):
        self._critic_num = critic_num
        super().__init__(*args, **kwargs)

    def _get_metric_name(self):
        return f"critic_loss{self._critic_num}/episodic_return_gt_td"

    @property
    def x_label(self):
        return "Training Iterations"

    @property
    def y_label(self):
        return "lb-DR Return Greater Rate"


class IterationsLBGDGTReader(MeanCurveReader):
    """Create a mean curve reader that reads Success rates."""

    def __init__(self, critic_num, *args, **kwargs):
        self._critic_num = critic_num
        super().__init__(*args, **kwargs)

    def _get_metric_name(self):
        return f"critic_loss{self._critic_num}/goal_return_gt_td"

    @property
    def x_label(self):
        return "Training Iterations"

    @property
    def y_label(self):
        return "lb-GD Return Greater Rate"


class IterationsValuesReader(MeanCurveReader):
    """Create a mean curve reader that reads Success rates."""

    def __init__(self, critic_num, *args, **kwargs):
        self._critic_num = critic_num
        super().__init__(*args, **kwargs)

    def _get_metric_name(self):
        return f"critic_loss{self._critic_num}/values/mean"

    @property
    def x_label(self):
        return "Training Iterations"

    @property
    def y_label(self):
        return "Mean Training Value"


class MeanCurveGroupReader(object):
    r"""Group several ``MeanCurveReader`` results. A ``MeanCurveGroupReader`` is
    suitable for one method on multiple tasks, each task with multiple runs.
    To aggregate across tasks, each task must be provided with a performance
    range :math:`(y_0, y_1)` that will be used to normalize performance for that
    task as :math:`\frac{y - y_0}{y_1 - y_0}`. If the ranges are not provided,
    no normalization will be done.

    The aggregation is simply averaging the statistics of individual ``MeanCurve``.
    """

    def __init__(self,
                 mean_curve_readers,
                 task_performance_ranges=None,
                 name="MeanCurveGroupReader"):
        """
        Args:
            mean_curve_readers (list[MeanCurveReader]): a list of
                ``MeanCurveReader`` of multiple tasks for one method. It's the
                user's responsibility to ensure that it's meaningful to
                group these task event files and show their mean and variance.
            task_performance_ranges (list[tuple(float)]): a list of tuples, where
                each tuple is a pair of floats used for normalizing the corresponding
                task. If None, no normalization will be performed.
            name (str): name of the method
        """

        def _normalize(y, y0, y1):
            return (y - y0) / (y1 - y0)

        if task_performance_ranges is None:
            task_performance_ranges = [(0., 1.)] * len(mean_curve_readers)

        assert len(mean_curve_readers) == len(task_performance_ranges)
        curves = [reader() for reader in mean_curve_readers]

        agg_vals = dict(y=[], min_y=[], max_y=[], ay=[], min_ay=[], max_ay=[])

        for c, (y0, y1) in zip(curves, task_performance_ranges):
            assert len(c.x) == len(curves[0].x)
            for key in agg_vals.keys():
                agg_vals[key].append(_normalize(getattr(c, key), y0, y1))

        for key, val in agg_vals.items():
            agg_vals[key] = np.mean(val, axis=0)

        self._mean_curve = MeanCurve(
            x=curves[0].x, name=curves[0].name, **agg_vals)

        self._x_label = mean_curve_readers[0].x_label
        self._name = name

    @property
    def x_label(self):
        return self._x_label

    @property
    def y_label(self):
        return "Normalized Score"

    @property
    def name(self):
        return self._name

    def __call__(self):
        return self._mean_curve


class CurvesPlotter(object):
    """Plot several ``MeanCurve``s in a figure. The curve colors will form
    a cycle over 10 default colors. The user should make sure that the ``MeanCurve``s
    to plot are meaningful to be compared in one figure.

    For each ``MeanCurve``, its ``y`` field will be plotted as the mean, its
    ``min_y`` and ``max_y`` will be plotted by a shaded area around ``y``, and
    its ``x`` determines the x-axis range.
    """
    _COLORS = ['C%d' % i for i in range(10)]

    def __init__(self,
                 mean_curves,
                 y_clipping=None,
                 x_range=None,
                 y_range=None,
                 x_ticks=None,
                 x_label=None,
                 y_label=None,
                 x_scaled_and_aligned=False,
                 figsize=(4, 4),
                 dpi=100,
                 linestyle='-',
                 linewidth=2,
                 std_alpha=0.3,
                 bg_color=None,
                 grid_color=None,
                 plot_mean_only=False,
                 legend_kwargs=dict(loc="best"),
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
            x_ticks (list[float]): x ticks shown along x axis
            y_clipping (tuple[float]): the y values will be clipped to this range
                if not None. Because of smoothing in ``MeanCurveReader`` and/or
                std region, the input y values might be out of this range.
            x_label (str): shown besides x-axis
            y_label (str): shown besides y-axis
            x_scaled_and_aligned (bool): If True, the x axes of all MeanCurves
                will be scaled and aligned to ``x_range``; otherwise, the x axes
                will be plot according to ``x`` of each MeanCurve.
            figsize (tuple[int]): a tuple of ints determining the size of the
                figure in inches. A larger figure size will allow for longer texts,
                more axes or more ticklabels to be shown.
            dpi (int): Dots per inches. How many pixels each inch contains. A
                ``figsize`` of ``(w,h)`` consists of ``w*h*dpi**2`` pixels.
            linestyle (str|list[str]): the line style to plot. Possible values:
                '-' ('solid'), '--' ('dashed'), '-.' (dashdot), and ':' ('dotted').
                If a string, then all curves will have the same style; otherwise
                each option will apply to the corresponding curve.
            linewidth (int): the thickness of lines to plot. Default: 2.
            std_alpha (float): the transparency value for plotting shaded area around
                a curve.
            bg_color (str): the background color of the figure
            grid_color (str): color of the dashed grid lines
            plot_mean_only (bool): Whether only plot the mean curve without
                shaded regions.
            legend_kwargs (dict): kwargs for plotting the legend. If None, then
                no legend will be plotted.
            title (str): title of the figure
        """
        self._fig, ax = plt.subplots(1, figsize=figsize, dpi=dpi)

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

        if not isinstance(linestyle, list):
            linestyle = [linestyle] * len(mean_curves)
        elif len(linestyle) < len(mean_curves):
            linestyle += linestyle[-1:] * (len(mean_curves) - len(linestyle))

        for i, c in enumerate(mean_curves):
            if i < len(mean_curves) - 1:
                color = self._COLORS[i % len(self._COLORS)]

            else:  # assume the last method is best; "black" for highlighting
                color = "black"
            x = (scaled_x if x_scaled_and_aligned else c.x)
            ax.plot(
                x,
                _clip_y(c.y),
                color=color,
                lw=linewidth,
                linestyle=linestyle[i],
                label=c.name)
            if not plot_mean_only:
                ax.fill_between(
                    x,
                    _clip_y(c.max_y),
                    _clip_y(c.min_y),
                    facecolor=color,
                    alpha=std_alpha)

        if legend_kwargs is not None:
            ax.legend(**legend_kwargs)
        if bg_color is not None:
            ax.set_facecolor(bg_color)
        if grid_color is not None:
            ax.grid(linestyle='--', color=grid_color)
        else:
            ax.grid(linestyle='-')
        if x_ticks is not None:
            ax.set_xticks(x_ticks)
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


def plot(env, her, train, curves):
    """Plotting examples."""
    print(f"Plotting {train} {curves} curves for {env} tasks (her: {her})")

    FOUR_ATARI_GAMES = [
        "Frostbite",
        "Qbert",
        "Breakout",
        "Seaquest",  # use these four games for dqn, td-lambda and retrace
    ]
    SIX_ATARI_GAMES = FOUR_ATARI_GAMES + ["Atlantis", "SpaceInvaders"]
    ELEVEN_ATARI_GAMES = [
        "Enduro", "Riverraid", "Alien", "BankHeist", "Centipede",
        "FishingDerby", "IceHockey", "MsPacman", "RoadRunner", "TimePilot",
        "Zaxxon"
    ]

    SEVENTEEN_ATARI_GAMES = SIX_ATARI_GAMES + ELEVEN_ATARI_GAMES

    mstr_map = {
        "ac": "",
        "sac": "",  # baseline
        "dqn": "",  # baseline
        "ddpg": "",  # baseline
        "her": "",  # baseline
        "td3": "",  # baseline
        "tdl": "-loss_at_TDLoss-tdlambda_0.95-batlen_3",
        "retrace": "-loss_at_TDLoss-tdlambda_0.95-batlen_3-retrace",
        "opttight3": "-lbbts-btslambda_4-loss_at_TDLoss-tdlambda_1-batlen_3",
        "opttight4": "-lbbts-btslambda_4-loss_at_TDLoss-tdlambda_1-batlen_4",
        "lbtq_b3": "-lbbts-loss_at_TDLoss-tdlambda_1-batlen_3",
        "lbtq_b4": "-lbbts-loss_at_TDLoss-tdlambda_1-batlen_4",
        "lbtq_b8": "-lbbts-loss_at_TDLoss-tdlambda_1-batlen_8",
        "lbtq_b12": "-lbbts-loss_at_TDLoss-tdlambda_1-batlen_12",
        "lbtq_b8_o": "-lbbts-btsnonly-loss_at_TDLoss-tdlambda_1-batlen_8",
        "lbtq_b12_o": "-lbbts-btsnonly-loss_at_TDLoss-tdlambda_1-batlen_12",
        "gdist": "-gdist",
        "lbtq": "-lbtq",
        "lbtqgdist": "-lbtq-gdist"
    }
    method_tag = {
        "ac": "a2c",
        "ddpg": "ddpg",
        "td3": "td3",
        "sac": "sac",
        "dqn": "dqn",
        "her": "her",
        "tdl": "td-lambda",
        "retrace": "retrace",
        "opttight3": "opt-tighten-2step",
        "opttight4": "opt-tighten-3step",
        "lbtq_b3": "lb-b-2step (ours)",
        "lbtq_b4": "lb-b-3step (ours)",
        "lbtq_b8": "lb-b-7step (ours)",
        "lbtq_b12": "lb-b-11step (ours)",
        "lbtq_b8_o": "lb-b-7step-only (ours)",
        "lbtq_b12_o": "lb-b-11step-only (ours)",
        "lbtq": "lb-DR (ours)",
        "gdist": "lb-GD (ours)",
        "lbtqgdist": "lb-DR+GD (ours)"
    }
    task_map = {
        "PioneerPush": "pushst",
        "PioneerPushReach": "pushst-prch",
        "PioneerReach": "targetnavst"
    }

    if her:
        methods = ["lbtqgdist", "gdist", "her"]
    else:
        if env in ["atari", "atarirn"]:
            methods = ["lbtq", "sac"]
            # methods = ["lbtq", "tdl", "retrace", "sac"]
            # methods = ["lbtq", "dqn"]
        elif env == "atariac":
            methods = ["lbtq", "sac", "ac"]
        elif env == "atari_b":
            methods = ["lbtq_b4", "tdl", "retrace", "sac"]  # , "opttight4"
        elif env == "fetch":
            methods = ["lbtq", "ddpg"]  # , "tdl"
        elif env == "fetch_b":
            methods = ["lbtq_b3", "opttight3", "tdl", "ddpg"]
            # methods = ["lbtq_b3", "lbtq_b8", "lbtq_b12", "ddpg"]
            # methods = ["lbtq_b8", "lbtq_b12", "lbtq_b8_o", "lbtq_b12_o", "ddpg"]
        else:
            methods = ["lbtq", "td3"]

    if env == "fetch":
        critic_num = 0
        tasks = ["FetchPush", "FetchPickPlace", "FetchSlide"]
        if curves == "return":
            total_steps = 2000000
        else:
            total_steps = 1100
        cluster_str = "/tboardlog"
    elif env == "fetch_b":
        critic_num = 0
        tasks = ["FetchPush", "FetchPickPlace"]
        if curves == "return":
            total_steps = 5000000
        else:
            total_steps = 2680
    elif env == "pioneer":
        critic_num = 1
        tasks = ["PioneerPush", "PioneerPushReach"]
        total_steps = 0
        if curves == "return":
            task_total_steps = {
                "PioneerPush": 5000000,
                "PioneerPushReach": 14000000
            }
        else:
            task_total_steps = {"PioneerPush": 1800, "PioneerPushReach": 5000}
        cluster_str = ""
    elif env in ["atari", "atarirn"]:
        critic_num = 1
        tasks = SEVENTEEN_ATARI_GAMES
        if curves == "return":
            total_steps = 12000000
        else:
            total_steps = 50000
        cluster_str = "/tboardlog"
    elif env == "atariac":
        critic_num = 1
        tasks = ["Breakout"]
        total_steps = 12000000
        cluster_str = "/tboardlog"
    elif env == "atari_b":
        critic_num = 1
        tasks = FOUR_ATARI_GAMES
        if curves == "return":
            total_steps = 12000000
        else:
            total_steps = 50000
    else:
        assert False

    if curves == "return":
        plot_interval = 40 * 50
    else:
        if env in ["fetch", "fetch_b"]:
            plot_interval = 10
        elif env in ["atari", "atarirn", "atariac", "atari_b"]:
            plot_interval = 200
        elif env == "pioneer":
            plot_interval = 50

    print(
        f"total_steps: {total_steps or task_total_steps}, plot_interval: {plot_interval}"
    )

    def _run_name(m, t):
        mstr = mstr_map[m]
        bstr = "her" if her else "ddpg"
        if env == "fetch":
            rblen = ""
            if m in ["tdl", "retrace"]:
                rblen = "-rblen_10000"
            n = "%s%s%s-sparserwd-posrwd_0-succsince_0%s-sd_" % (
                bstr, t.lower(), mstr, rblen)
        elif env == "fetch_b":
            n = "ddpgfetch*/tboardlog/ddpg%s%s-sd_" % (t.lower(), mstr)
        elif env == "pioneer":
            n = "%s%s%s-curri_0-randrange_1-lr_0.001-crirep_2-herk_0.5-endsucc-sparserwd-posrwd_0-mdimr_0-hitpenal_0-randagentpp-it_5000" % (
                bstr, task_map[t], mstr)
        elif env == "atarirn":
            assert not her
            # rn = "-rnorm_at_AdaptiveNormalizer"
            rn = ""
            # lr = "-lr_0.001"
            lr = ""
            # if mstr == "-lbtq":
            #     upit = "-upit_8-batsz_250-coll_1e5"
            #     # upit = "-upit_4-batsz_500-coll_1e5"
            # else:
            # upit = "-upit_4-batsz_500-coll_1e5"
            upit = ""
            #    # upit = "-upit_4-batsz_500"
            # n = "sacbreakout%s%s-envn_%sNoFrameskip--v4%s%s-evit_1000-evepi_10-sd_?-" % (
            #     mstr, rn, t, lr, upit)
            n = "sacbreakout%s%s-envn_%sNoFrameskip--v4%s%s-sd_4" % (
                mstr, rn, t, lr, upit)

        elif env in ["atariac"]:  # "atari",
            assert not her
            if mstr == "-lbtq":
                if t in ["Breakout", "Seaquest", "SpaceInvaders"]:
                    mstr = "r-lbtq"
                upit = "-upit_8-batsz_250"
            else:
                upit = "-upit_4-batsz_500"
            n = "sacbreakout%s-envn_%sNoFrameskip--v4%s-evit_1000-evepi_100-sd_3" % (
                mstr, t, upit)
            if m in ["tdl", "retrace"]:
                assert t == "Breakout"
                n = "sacbreakout%s%s-evit_1000-evepi_100-sd_3" % (mstr, upit)
            if m == "ac":
                n = "acbreakout-envstps_12000000-evepi_100-sd_3"
        elif env in ["atari", "atari_b"]:
            assert not her
            if t in SIX_ATARI_GAMES:
                n = (
                    "*envn_%sNoFrameskip--v4-sd_3*/tboardlog/sacbreakout%s-envn_%sNoFrameskip--v4-sd_3*"
                    % (t, mstr, t))  # with sac baseline
                # n = ("*envn_%sNoFrameskip--v4-sd_3*/tboardlog/dqnbreakout%s-envn_%sNoFrameskip--v4-sd_3*" %
                #      (t, mstr, t))  # with dqn baseline
            else:
                n = "sacbreakout%s-envn_%sNoFrameskip--v4-sd_3" % (mstr, t)
        return n

    reader_cls = {
        "drgt":
            lambda *args, **kwargs: IterationsLBDRGTReader(
                critic_num, *args, **kwargs),
        "gdgt":
            lambda *args, **kwargs: IterationsLBGDGTReader(
                critic_num, *args, **kwargs),
        "bstgt":
            lambda *args, **kwargs: IterationsLBBootstrapGTReader(
                critic_num, *args, **kwargs),
        "value":
            lambda *args, **kwargs: IterationsValuesReader(
                critic_num, *args, **kwargs),
        "return":
            EnvironmentStepsReturnReader,
        "discreturn":
            EnvironmentStepsEpisodicStartAverageDiscountedReturnReader,
        "avgreward":
            EnvironmentStepsAverageRewardReader,
    }[curves]

    def curve_in_method(curves, m):
        return curves in ["value", "return", "discreturn", "avgreward"
                          ] or (curves, m) in [
                              ("drgt", "lbtq"),
                              ("gdgt", "gdist"),
                              ("gdgt", "lbtqgdist"),
                              ("bstgt", "opttight3"),
                              ("bstgt", "opttight4"),
                              ("bstgt", "lbtq_b3"),
                              ("bstgt", "lbtq_b4"),
                              ("bstgt", "lbtq_b8"),
                              ("bstgt", "lbtq_b12"),
                          ]

    def get_curve_path(dir="", m=None, t=None):
        _env = env
        if env == "atari_b":
            _env = "atari"
        p = os.path.join(os.getenv("HOME"), "tmp/iclr22/" + _env, dir)
        if dir == "":
            print(f"Writing to {p} for task {t}")
        else:
            print(f"Reading from {p} for task {t}, method {m}")
        return p

    curve_readers = [[
        reader_cls(
            glob.glob(
                get_curve_path(
                    "%s*%s/%s" % (_run_name(m, t), "" if
                                  (t in SIX_ATARI_GAMES or env == "fetch_b")
                                  else cluster_str, train),
                    t=t,
                    m=m)),
            np.arange(0, total_steps or task_total_steps[t], plot_interval),
            name="%s" % (method_tag[m]),
            smoothing=3) for t in tasks
    ] for m in methods if curve_in_method(curves, m)]

    print("Plotting", curves)
    for i, t in enumerate(tasks):
        _curves = [cr[i]() for cr in curve_readers]
        if not _curves:
            break
        min_y = min([min(c.min_y) for c in _curves])
        max_y = max([max(c.max_y) for c in _curves])
        y_margin = (max_y - min_y) * 0.05
        # Scale and align x-axis of methods on task1
        plotter = CurvesPlotter(
            _curves,
            x_label=curve_readers[0][0].x_label,
            y_label=curve_readers[0][0].y_label,
            y_range=(min_y - y_margin, max_y + y_margin),  # task_y_range[t],
            x_range=(0, total_steps or task_total_steps[t]))
        plotter.plot(
            output_path=os.path.join(
                get_curve_path(t=t), "vs".join(methods) + "-" + curves + "-" +
                t + "-" + train + ".pdf"))


if __name__ == "__main__":
    import sys
    if len(sys.argv) <= 1:
        env = "pioneer"  # one of "pioneer", "atari", "atariac", "atarirn", "atari_b", "fetch", "fetch_b"
    else:
        env = sys.argv[1]
    env_her = {
        "atari": [False, ],
        "atari_b": [False, ],
        "atarirn": [False, ],
        "atariac": [False, ],
        "fetch": [False, True],
        "fetch_b": [False],
        "pioneer": [True, ]
    }

    if env == "atariac":
        train_curves = [("eval", "return", False), ("eval", "discreturn",
                                                    False),
                        ("eval", "avgreward", False),
                        ("train", "discreturn", False),
                        ("train", "avgreward", False),
                        ("train", "return", False)]
    else:
        train_curves = [
            ("eval", "return", False),
            ("eval", "return", True),
            ("train", "value", False),
            ("train", "value", True),
            ("train", "drgt", False),
            ("train", "gdgt", True),
            ("train", "bstgt", False),
        ]
    for train, curves, her in train_curves:
        # train is "train" or "eval"
        # curves is one of "drgt", "gdgt", "return", "value"
        if her in env_her[env]:
            plot(env, her, train, curves)
