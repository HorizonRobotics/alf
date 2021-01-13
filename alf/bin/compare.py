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
r"""Compare two algorithms on a set of fixed task initializations.

Run:
```bash
python3 -m alf.bin.compare \
  --root_dir1=~/tmp/ac_cart_pole \
  --root_dir2=~/tmp/ddpg_cart_pole \
  --alsologtostderr
```
Prefix with ```DISPLAY= vglrun -d :7 ``` if running remotely with virtual_gl.
The cleared DISPLAY env_var is so that gzclients are not created.
gzclients are not being torn down after play and can occupy too many xserver
connections.
Set the proper DISPLAY variable when recording video.
"""

from absl import app
from absl import flags
from absl import logging
from subprocess import Popen, TimeoutExpired
import collections
import heapq
import numpy as np
import os
import re
import time

flags.DEFINE_string(
    'root_dir', None,
    'Root directory for both algorithms, when gin1 and gin2 are set.')
flags.DEFINE_string('root_dir1', None, 'Root directory for algorithm one.')
flags.DEFINE_string('root_dir2', None, 'Root directory for algorithm two.')
flags.DEFINE_string('gin1', None, 'gin params for algorithm one.')
flags.DEFINE_string('gin2', None, 'gin params for algorithm two.')
flags.DEFINE_string('output_file', None, 'output html file.')
flags.DEFINE_integer('num_runs', 10, 'Compare on so many runs.')
flags.DEFINE_integer('start_from', 0, 'Start random seeds from here.')
flags.DEFINE_string(
    'common_gin', '', 'Common config for the two sides, '
    'e.g. "--gin_param=\'GoalTask.random_range=5\'"')
flags.DEFINE_bool('overwrite', False, 'Overwrite cached files.')

FLAGS = flags.FLAGS

AVG_R_METRIC = "AverageReturn"
AVG_R_DIFF = AVG_R_METRIC + "_diff"


def _return_diff(item):
    return abs(item[AVG_R_DIFF]) / (max(
        abs(float(item["alg1_" + AVG_R_METRIC])),
        abs(float(item["alg2_" + AVG_R_METRIC]))) + 1.e-5)


def _return_1_larger(item):
    return float(item["alg1_" + AVG_R_METRIC]) > float(
        item["alg2_" + AVG_R_METRIC])


def _file_exists(file):
    return (not FLAGS.overwrite and os.path.isfile(file)
            and os.stat(file).st_size > 100)


def _play_cmd(root_dir, seed, mp4_f, gin):
    cmd = ("python3 -m alf.bin.play "
           " --random_seed={seed} --num_episodes=1"
           " --verbosity=1 --root_dir={root_dir} {gin} --sleep_time_per_step=0"
           " --epsilon_greedy=0 --record_file={mp4} {g}").format(
               root_dir=root_dir,
               gin=gin,
               seed=seed,
               mp4=mp4_f,
               g=FLAGS.common_gin)
    cmds = cmd.split(" ")
    return cmd, cmds


def _get_metric(pattern, buffer, log_file=None):
    match = re.search(pattern, buffer)
    if log_file:
        assert match, "{} not found in {}, re-run?".format(pattern, log_file)
    if not match:
        return None
    return "{:.2f}".format(float(match.group(1)))


def _get_avg(data, metric, i):
    vs = [float(v["alg{}_{}".format(i + 1, metric)]) for v in data]
    return np.mean(vs)


def _create_html(data, all_data, metrics, abbr, root_dir1, root_dir2):
    """Creates the comparison in html content and return as string."""
    # Column ``AverageReturn_diff`` is after:
    #   one seed column, two sets of metrics, two log_file paths
    avgreturn_index = 2 * len(metrics) + 2 + 1
    seed_index = 0

    html = r"""
<html>
    <script src="https://code.jquery.com/jquery-3.5.1.min.js"
        integrity="sha256-9/aliU8dGd2tb6OSsuzixeV4y/faTqgFtohetphbbj0=" crossorigin="anonymous"></script>
    <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.10.21/css/jquery.dataTables.css">
    <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/1.10.21/js/jquery.dataTables.js"></script>
    <script>
        $(document).ready(function () {
            $('#table_id').DataTable({
                "autoWidth": false,
                "aLengthMenu": [ [25, 50, 100, 200, -1], [25, 50, 100, 200, "All"] ], "iDisplayLength": -1,
                "order": [[""" + '{}, "desc"], [{}, "asc"]]'.format(
        avgreturn_index, seed_index) + r"""
            });
        });
    </script>
    <body><h3>Compare difference between algorithms</h3>
    """

    # Summary:
    html += "\n<pre>"
    if FLAGS.common_gin:
        html += "common_args: {}\n".format(FLAGS.common_gin)
    html += "Alg1: {}\ngin1: {}\n".format(root_dir1, FLAGS.gin1)
    for m in metrics:
        html += "&nbsp;&nbsp;&nbsp;|{}: {:.2f}".format(
            m, _get_avg(all_data, m, 0))
    html += "\nAlg2: {}\ngin2: {}\n".format(root_dir2, FLAGS.gin2)
    for m in metrics:
        html += "&nbsp;&nbsp;&nbsp;|{}: {:.2f}".format(
            m, _get_avg(all_data, m, 1))
    html += "\nnum_items: {}, have data for: {}\n".format(
        FLAGS.num_runs, len(all_data))
    percentiles = [.05, .1, .2, .4, .8, .99]
    counts = [0] * len(percentiles)
    wins = [0] * len(percentiles)
    html += "proportion_diffs:\n"
    for item in data:
        diff = _return_diff(item)
        for i, p in enumerate(percentiles):
            if diff > p:
                counts[i] += 1
                if _return_1_larger(item):
                    wins[i] += 1
    for i, p in enumerate(percentiles):
        html += "diff > {:.2f}: {:.2f}, W: {:2d} vs L: {:2d}\n".format(
            p, counts[i] / len(all_data), wins[i], counts[i] - wins[i])

    # Table:
    html += """</pre><p>
    <table id="table_id" class="display" width="50%">
      <thead>
        <tr>"""
    if data:
        for k, _ in data[0].items():
            for i, metric in enumerate(metrics):
                if metric in k:
                    k = k.replace(metric, abbr[i])
            html += ("          <th style='word-break: break-word;'>{}</th>\n"
                     ).format(k)
    html += """
    </tr>
      </thead>
      <tbody>\n"""
    for item in data:
        html += "        <tr>\n"
        for k, v in item.items():
            if k in ["video1", "video2"] and v != "":
                v = """<video width="320" height="240" controls>
            <source src="{}" type="video/mp4">
            Your browser does not support the video tag.
          </video>""".format(v)
            html += "          <td>{}</td>\n".format(v)
        html += "        </tr>\n"
    html += """
      </tbody>
    </table>\n"""

    # Column header abbreviations:
    for i in range(len(metrics)):
        html += "<br>-- " + abbr[i] + ": " + metrics[i]
    html += r"""
  </body>
</html>"""
    return html


def _tokenize(s):
    if not s:
        return ""
    s = s.replace("--gin_param=", "")
    s = s.replace("/home/lezhao/tmp/", "")
    s = s.replace("goal_gen/SubgoalPlanningGoalGenerator.num_subgoals=",
                  "numsg_")
    s = s.replace("CEMOptimizer.iterations=", "cemiters_")
    s = s.replace("GoalTask.min_distance=", "mindist_")
    s = s.replace("GoalTask.random_range=", "randrange_")
    s = s.replace("'", "")
    s = s.replace('"', "")
    s = s.replace("=", "__")
    s = s.replace(" ", "-")
    s = s.replace("/", "_")
    return s


def _sort_options(s):
    if not s:
        return ""
    arr = s.split("-")
    arr.sort()
    return "-".join(arr)


def main(_):
    """main function."""
    # generate runs
    assert ((FLAGS.root_dir and FLAGS.gin1 != FLAGS.gin2)
            or (FLAGS.root_dir1 and FLAGS.root_dir2))
    root_dir1, root_dir2 = FLAGS.root_dir1, FLAGS.root_dir2
    if FLAGS.root_dir:
        root_dir1, root_dir2 = FLAGS.root_dir, FLAGS.root_dir
    dirs = [root_dir1, root_dir2]
    gins = [FLAGS.gin1, FLAGS.gin2]
    metrics = [AVG_R_METRIC, "AverageEpisodeLength"]
    abbr = ["R", "L"]
    gin_str = ""
    if FLAGS.common_gin:
        gin_str = _tokenize(FLAGS.common_gin)
        gin_str = "-" + gin_str

    data = []  # used for displaying diffs in final HTML
    all_data = []  # used for computing average stats
    for seed in range(FLAGS.num_runs):
        seed += FLAGS.start_from
        item = collections.OrderedDict()
        item["seed"] = seed
        vs = ["", ""]
        for i, root_dir in enumerate(dirs):
            _gin_str = gin_str
            if gins[i]:
                _gin_str += "-" + _tokenize(gins[i])
            _gin_str = _sort_options(_gin_str)
            mp4_f = root_dir + "/play-seed_{}{}.mp4".format(seed, _gin_str)
            log_file = root_dir + "/play-log-seed_{}{}.txt".format(
                seed, _gin_str)
            command, commands = _play_cmd(root_dir, seed, mp4_f, gins[i])
            command += " 2>> " + log_file
            if not _file_exists(mp4_f):
                f = open(log_file, 'w')
                assert f, "cannot write to " + log_file
                f.write(command + "\n")
                f.close()
                print(command)
                lf = open(log_file, 'a')
                p = Popen(commands, cwd=root_dir, stderr=lf)
                while True:
                    time.sleep(2)
                    lf.flush()
                    f = open(log_file, 'r')
                    assert f, log_file + " cannot be read."
                    lines = f.read().replace('\n', ' ')
                    f.close()
                    if _get_metric(r"\] " + AVG_R_METRIC + r": (\S+)", lines):
                        try:
                            p.wait(5)
                        except TimeoutExpired:
                            print("killing subprocess..")
                            p.kill()
                        break
                lf.close()
            vs[i] = mp4_f

            # extract values
            f = open(log_file, 'r')
            assert f, log_file + " cannot be read."
            lines = f.read().replace('\n', ' ')
            f.close()
            for metric in metrics:
                value = _get_metric(r"\] " + metric + r": (\S+)", lines,
                                    log_file)
                item["alg{}_{}".format(i + 1, metric)] = value
            item["logfile{}".format(
                i + 1)] = '<a href="file://{}">log_file</a>'.format(log_file)
        for metric in metrics:
            m1 = float(item["alg{}_{}".format(1, metric)])
            m2 = float(item["alg{}_{}".format(2, metric)])
            diff = m1 - m2
            item["{}_diff".format(metric)] = diff
        item["video1"] = vs[0]
        item["video2"] = vs[1]
        all_data.append(item)
        if _return_diff(item) > 0.05:  # >5% diff
            data.append(item)

    # Skipping this section as recording video affects the randomness in gazebo_base.
    # analyze results to record videos
    # abs_avg = [abs(v[AVG_R_DIFF]) for v in data]
    # idx = heapq.nlargest(FLAGS.num_videos, range(len(data)),
    #                      abs_avg.__getitem__)

    # create html
    output_file = FLAGS.output_file
    if not output_file:
        custom_gins = ""
        if FLAGS.gin1 or FLAGS.gin2:
            if root_dir1 == root_dir2:
                dir2_str = ""
            else:
                dir2_str = "-dir2_{}".format(_tokenize(root_dir2))
            custom_gins = "gin1_{}-gin2_{}{}".format(
                _tokenize(FLAGS.gin1), _tokenize(FLAGS.gin2), dir2_str)
        output_file = root_dir1 + "/compare{}-{}-startfrom_{}-numruns_{}.html".format(
            gin_str, custom_gins, FLAGS.start_from, FLAGS.num_runs)
    html = _create_html(data, all_data, metrics, abbr, root_dir1, root_dir2)
    f = open(output_file, 'w')
    assert f, "Cannot write to " + output_file
    f.write(html)
    f.close()
    print("Done: ", output_file)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(main)
