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
import collections
import heapq
import numpy as np
import os
import re

flags.DEFINE_string('root_dir1', None, 'Root directory for algorithm one.')
flags.DEFINE_string('root_dir2', None, 'Root directory for algorithm two.')
flags.DEFINE_string('output_file', None, 'output html file.')
flags.DEFINE_integer('num_runs', 10, 'Compare on so many runs.')
flags.DEFINE_integer('start_from', 0, 'Start random seeds from here.')
flags.DEFINE_integer('num_videos', 2, 'Record videos for so many top diffs.')
flags.DEFINE_string(
    'common_gin', '', 'Common config for the two sides, '
    'e.g. "--gin_param=\'GoalTask.random_range=5\'"')

FLAGS = flags.FLAGS

AVG_R_METRIC = "AverageReturn"
AVG_R_DIFF = AVG_R_METRIC + "_diff"


def return_diff(item):
    return abs(item[AVG_R_DIFF]) / (max(
        abs(float(item["alg1_" + AVG_R_METRIC])),
        abs(float(item["alg2_" + AVG_R_METRIC]))) + 1.e-5)


def file_exists(file):
    return os.path.isfile(file) and os.stat(file).st_size > 100


def play_cmd(root_dir, seed):
    return ("cd {root_dir} && "
            "python3 -m alf.bin.play --root_dir={root_dir}"
            " --random_seed={seed} --num_episodes=1"
            " --verbosity=1 --root_dir=`pwd` --sleep_time_per_step=0"
            " --epsilon_greedy=0 {g}").format(
                root_dir=root_dir, seed=seed, g=FLAGS.common_gin)


def get_metric(pattern, buffer, log_file):
    match = re.search(pattern, buffer)
    assert match, "{} not found in {}, remove and re-run?".format(
        pattern, log_file)
    return match.group(1)


def get_avg(data, metric, i):
    vs = [float(v["alg{}_{}".format(i + 1, metric)]) for v in data]
    return np.mean(vs)


def create_html(data, all_data, metrics):
    """Creates the comparison in html content and return as string."""
    # Column ``AverageReturn_diff`` is after:
    #   one seed column, two sets of metrics, two log_file paths
    avgreturn_index = 2 * len(metrics) + 2 + 1
    seed_index = 0

    html = r"""<html>
        <script src="https://code.jquery.com/jquery-3.5.1.min.js"
            integrity="sha256-9/aliU8dGd2tb6OSsuzixeV4y/faTqgFtohetphbbj0=" crossorigin="anonymous"></script>
        <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.10.21/css/jquery.dataTables.css">
        <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/1.10.21/js/jquery.dataTables.js"></script>
        <script>
            $(document).ready(function () {
            $('#table_id').DataTable({
                "aLengthMenu": [ [25, 50, 100, 200, -1], [25, 50, 100, 200, "All"] ], "iDisplayLength": -1,
                "order": [[""" + '{}, "desc"], [{}, "asc"]]'.format(
        avgreturn_index, seed_index) + r"""
            });
            });
        </script>
        <body>
        """

    # Summary:
    html += "Alg1: {}<br>".format(FLAGS.root_dir1)
    for m in metrics:
        html += "&nbsp;&nbsp;&nbsp;|{}: {}".format(m, get_avg(all_data, m, 0))
    html += "<br>Alg2: {}<br>".format(FLAGS.root_dir2)
    for m in metrics:
        html += "&nbsp;&nbsp;&nbsp;|{}: {}".format(m, get_avg(all_data, m, 1))
    html += "<br>num_items: {}, have data for: {}<br>".format(
        FLAGS.num_runs, len(all_data))
    percentiles = [.05, .1, .2, .4, .8, .99]
    counts = [0] * len(percentiles)
    html += "propotion_diffs:<br>\n"
    for item in data:
        diff = return_diff(item)
        for i, p in enumerate(percentiles):
            if diff > p:
                counts[i] += 1
    for i, p in enumerate(percentiles):
        html += "diff > {:.2f}: {}<br>\n".format(p, counts[i] / len(all_data))

    # Table:
    html += """<p>
    <table id="table_id" class="display">
      <thead>
        <tr>"""
    if data:
        for k, _ in data[0].items():
            html += "          <th>{}</th>\n".format(k)
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
    </table>
  </body>
</html>"""
    return html


def tokenize(s):
    s = s.replace("--gin_param=", "")
    s = s.replace("'", "")
    s = s.replace('"', "")
    s = s.replace("=", "__")
    s = s.replace(" ", "-")
    s = s.replace("/", "_")
    return s


def main(_):
    """main function"""
    # generate runs
    dirs = [FLAGS.root_dir1, FLAGS.root_dir2]
    metrics = [AVG_R_METRIC, "AverageEpisodeLength"]
    gin_str = ""
    if FLAGS.common_gin:
        gin_str = tokenize(FLAGS.common_gin)
        gin_str = "-" + gin_str

    data = []  # used for displaying diffs in final HTML
    all_data = []  # used for computing average stats
    for seed in range(FLAGS.num_runs):
        seed += FLAGS.start_from
        item = collections.OrderedDict()
        item["seed"] = seed
        for i in range(2):
            root_dir = dirs[i]
            log_file = root_dir + "/log-seed_{}{}.txt".format(seed, gin_str)
            command = play_cmd(root_dir, seed) + " 2>> {}".format(log_file)
            if not file_exists(log_file):
                f = open(log_file, 'w')
                assert f, "cannot write to " + log_file
                f.write(command + "\n")
                f.close()
                os.system(command)

            # extract values
            f = open(log_file, 'r')
            assert f, log_file + " cannot be read."
            lines = f.read().replace('\n', ' ')
            f.close()
            for metric in metrics:
                value = get_metric(r"\] " + metric + r": (\S+)", lines,
                                   log_file)
                item["alg{}_{}".format(i + 1, metric)] = value
            item["logfile{}".format(i)] = '<a href="file://{}">{}</a>'.format(
                log_file, log_file)
        for metric in metrics:
            m1 = float(item["alg{}_{}".format(1, metric)])
            m2 = float(item["alg{}_{}".format(2, metric)])
            diff = m1 - m2
            item["{}_diff".format(metric)] = diff
        all_data.append(item)
        if return_diff(item) > 0.05:  # >5% diff
            data.append(item)

    # analyze results to record videos
    abs_avg = [abs(v[AVG_R_DIFF]) for v in data]
    idx = heapq.nlargest(FLAGS.num_videos, range(len(data)),
                         abs_avg.__getitem__)
    for k, item in enumerate(data):
        vs = ["", ""]
        seed = item["seed"]
        if k in idx:
            for i, root_dir in enumerate(dirs):
                mp4_f = root_dir + "/play-seed_{}.mp4".format(seed)
                log_file = root_dir + "/play-log-seed_{}{}.txt".format(
                    seed, gin_str)
                command = play_cmd(root_dir,
                                   seed) + " --record_file={} 2>> {}".format(
                                       mp4_f, log_file)
                if not file_exists(mp4_f):
                    f = open(log_file, 'w')
                    assert f, "cannot write to " + log_file
                    f.write(command + "\n")
                    f.close()
                    os.system(command)
                vs[i] = mp4_f
        item["video1"] = vs[0]
        item["video2"] = vs[1]

    # create html
    output_file = FLAGS.output_file
    if not output_file:
        output_file = FLAGS.root_dir1 + "/compare{}-{}.html".format(
            gin_str, tokenize(FLAGS.root_dir2))
    html = create_html(data, all_data, metrics)
    f = open(output_file, 'w')
    assert f, "Cannot write to " + output_file
    f.write(html)
    f.close()


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    flags.mark_flag_as_required('root_dir1')
    flags.mark_flag_as_required('root_dir2')
    app.run(main)
