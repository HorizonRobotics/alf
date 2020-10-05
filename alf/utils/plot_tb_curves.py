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

import matplotlib
import matplotlib.pyplot as plt

import alf.nest as nest
from alf.data_structures import namedtuple

HOME = os.getenv("HOME")

MeanCurve = namedtuple(
    "MeanCurve", ['y', 'min_y', 'max_y', 'name'], default_value=())


class MeanCurveReader(object):
    """Read and compute a MeanCurve from one or multiple TB event files.
    """
    _SIZE_GUIDANCE = {
        'compressedHistograms': 10,
        'images': 0,
        'scalars':
            100,  # sampled points will even distribute over the training time
        'histograms': 1
    }

    def _get_metric_name(self):
        raise NotImplementedError()

    def __init__(self,
                 event_file,
                 name="MeanCurve",
                 smoothing=None,
                 variance_mode="std",
                 max_n_scalars=None):
        """
        Args:
            event_file (str|list[str]): a string or a list of strings where
                each should point to a valid TB dir, e.g., ending with
                "eval/" or "train/".
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

        if max_n_scalars is not None:
            self._SIZE_GUIDANCE['scalars'] = max_n_scalars

        curves = []
        for ef in event_file:
            event_acc = EventAccumulator(ef, self._SIZE_GUIDANCE)
            event_acc.Reload()
            # show all tags in the log file
            print("Event file {} has scalar tags: {}".format(
                ef,
                event_acc.Tags()['scalars']))

            # 'scalar_events' is a list of ScalarEvent(wall_time, step, value),
            # with a maximal length specified by _SIZE_GUIDANCE
            scalar_events = event_acc.Scalars(self._get_metric_name())
            values = [se.value for se in scalar_events]
            curve = self._interpolate_and_smooth_if_necessary(
                values, smoothing)
            curves.append(curve)

        y = np.array(list(map(np.mean, zip(*curves))))
        if len(curves) == 1:
            self._mean_curve = MeanCurve(y=y, min_y=y, max_y=y, name=name)
        else:
            # compute mean and variance
            if variance_mode == "std":
                std = np.array(list(map(np.std, zip(*curves))))
                min_y, max_y = y - std, y + std
            elif variance_mode == "minmax":
                min_y = np.array(list(map(np.min, zip(*curves))))
                max_y = np.array(list(map(np.max, zip(*curves))))
            else:
                raise ValueError("Invalid variance mode: %s" % variance_mode)
            self._mean_curve = MeanCurve(
                y=y, min_y=min_y, max_y=max_y, name=name)

    def __call__(self):
        return self._mean_curve

    def _interpolate_and_smooth_if_necessary(self, curve, smoothing=None):
        """First linearly interpolate the curve if the number of scalars is less
        than the number in _SIZE_GUIDANCE. Second apply a smoothing to the curve
        if a smoothing factor is specified.

        Args:
            curve (list): a scalar curve
            smoothing (int): if None, no smoothing is applied; otherwise this
                is the window width of a Savitzky-Golay filter

        Returns:
            list: a new curve
        """
        n_scalars = self._SIZE_GUIDANCE["scalars"]
        if len(curve) < n_scalars:
            from scipy.interpolate import interp1d
            x = np.arange(len(curve))
            func = interp1d(x, np.array(curve))
            delta_x = (x[-1] - x[0]) / (n_scalars - 1)
            new_x = np.arange(n_scalars) * delta_x + x[0]
            curve = func(new_x)

        if smoothing is not None:
            from scipy.signal import savgol_filter
            curve = savgol_filter(curve, smoothing, polyorder=1)

        return curve


class EnvironmentStepsReturnReader(MeanCurveReader):
    """Create a mean curve reader that reads AverageReturn values."""

    def _get_metric_name(self):
        return "Metrics_vs_EnvironmentSteps/AverageReturn"


class EnvironmentStepsSuccessReader(MeanCurveReader):
    """Create a mean curve reader that reads Success rates."""

    def _get_metric_name(self):
        return "Metrics_vs_EnvironmentSteps/success"


class CurvesPlotter(object):
    """Plot several MeanCurves in a figure. The curve colors will form
    a cycle over 10 default colors.
    """
    _COLORS = ['C%d' % i for i in range(10)]
    _LINE_WIDTH = 2

    def __init__(self,
                 mean_curves,
                 x_range=None,
                 x_label=None,
                 y_label=None,
                 title=None):
        """
        Args:
            mean_curves (MeanCurve|list[MeanCurve]):
            x_range (tuple[float]): a tuple of (min_x, max_x) for showing on
                the figure. If None, then (0, 1) will be used.
            x_label (str): shown besides x-axis
            y_label (str): shown besides y-axis
            title (str): title of the figure
        """
        self._fig, ax = plt.subplots(1)

        if not isinstance(mean_curves, list):
            mean_curves = [mean_curves]

        if x_range is None:
            x_range = (0., 1.)

        n_points = len(mean_curves[0].y)
        delta_x = (x_range[-1] - x_range[0]) / (n_points - 1)
        x = np.arange(n_points) * delta_x + x_range[0]

        for i, c in enumerate(mean_curves):
            color = self._COLORS[i % len(self._COLORS)]
            ax.plot(x, c.y, color=color, lw=self._LINE_WIDTH, label=c.name)
            ax.fill_between(x, c.max_y, c.min_y, facecolor=color, alpha=0.3)

        ax.legend(loc='upper left')
        ax.grid(linestyle='--')

        if x_label:
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_ylabel(y_label)
        if title:
            ax.set_title(title)

    def plot(self, output_path, dpi=200, transparent=False):
        """Plot curves and save the figure to disk.

        Args:
            output_path (str): the output file path
            dpi (int): dpi for the figure. A higher value results in higher
                resolution.
            transparent (bool): If True, then the figure has a transparent
                background.
        """
        self._fig.savefig(
            output_path, dpi=dpi, transparent=transparent, bbox_inches='tight')
        plt.close(self._fig)


def _get_curve_path(dir):
    return os.path.join(HOME, "tensorboard_curves", dir)


if __name__ == "__main__":
    """A plotting example."""
    event_files = glob.glob(_get_curve_path("sac_kickball_gs/*/eval"))
    mean_curve_reader = EnvironmentStepsReturnReader(
        event_file=event_files, name="sac_kickball", smoothing=3)

    mean_curve_reader1 = EnvironmentStepsReturnReader(
        event_file=[
            _get_curve_path("tasac_carla/train"),
            _get_curve_path("tasac_carla_1td/train")
        ],
        name="tasac_carla",
        smoothing=5)

    plotter = CurvesPlotter(
        [mean_curve_reader(), mean_curve_reader1()],
        x_label="Env frames",
        y_label="Average Return",
        x_range=(0, 5000000))
    plotter.plot(output_path="/tmp/test.pdf")
