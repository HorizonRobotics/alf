# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
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
"""Utility functions for generate summary."""

from tensorboard.compat import tf2 as tf
from tensorboard.plugins.histogram import metadata

DEFAULT_BUCKET_COUNT = 30


def histogram_discrete(name,
                       data,
                       bucket_min,
                       bucket_max,
                       step=None,
                       description=None):
    """histogram for discrete data.

    Args:
        name (str): name for this summary
        data (Tensor): A `Tensor` integers of any shape.
        bucket_min (int): represent bucket min value
        bucket_max (int): represent bucket max value
            bucket count is calculate as `bucket_max - bucket_min + 1`
            and output will have this many buckets.
        step (None|tf.Variable):  step value for this summary. this defaults to
            `tf.summary.experimental.get_step()`
        description (str): Optional long-form description for this summary
    """
    summary_metadata = metadata.create_summary_metadata(
        display_name=None, description=description)
    summary_scope = (getattr(tf.summary.experimental, 'summary_scope', None)
                     or tf.summary.summary_scope)
    with summary_scope(
            name, 'histogram_summary',
            values=[data, bucket_min, bucket_max, step]) as (tag, _):
        with tf.name_scope('buckets'):
            bucket_count = bucket_max - bucket_min + 1
            data = data - bucket_min
            one_hots = tf.one_hot(
                tf.reshape(data, shape=[-1]), depth=bucket_count)
            bucket_counts = tf.cast(
                tf.reduce_sum(input_tensor=one_hots, axis=0), tf.float64)
            edge = tf.cast(tf.range(bucket_count), tf.float64)
            # histogram can not draw when left_edge == right_edge
            left_edge = edge - 1e-12
            right_edge = edge + 1e-12
            tensor = tf.transpose(
                a=tf.stack([left_edge, right_edge, bucket_counts]))

        return tf.summary.write(
            tag=tag, tensor=tensor, step=step, metadata=summary_metadata)


def histogram_continuous(name,
                         data,
                         bucket_min=None,
                         bucket_max=None,
                         bucket_count=DEFAULT_BUCKET_COUNT,
                         step=None,
                         description=None):
    """histogram for continuous data .

    Args:
        name (str): name for this summary
        data (Tensor): A `Tensor` of any shape.
        bucket_min (float|None): represent bucket min value,
            if None value of tf.reduce_min(data) will be used
        bucket_max (float|None): represent bucket max value,
            if None value tf.reduce_max(data) will be used
        bucket_count (int):  positive `int`. The output will have this many buckets.
        step (None|tf.Variable):  step value for this summary. this defaults to
            `tf.summary.experimental.get_step()`
        description (str): Optional long-form description for this summary
    """
    summary_metadata = metadata.create_summary_metadata(
        display_name=None, description=description)
    summary_scope = (getattr(tf.summary.experimental, 'summary_scope', None)
                     or tf.summary.summary_scope)
    with summary_scope(
            name,
            'histogram_summary',
            values=[data, bucket_min, bucket_max, bucket_count, step]) as (tag,
                                                                           _):
        with tf.name_scope('buckets'):
            data = tf.cast(tf.reshape(data, shape=[-1]), tf.float64)
            if bucket_min is None:
                bucket_min = tf.reduce_min(data)
            if bucket_max is None:
                bucket_max = tf.reduce_min(data)
            range_ = bucket_max - bucket_min
            bucket_width = range_ / tf.cast(bucket_count, tf.float64)
            offsets = data - bucket_min
            bucket_indices = tf.cast(
                tf.floor(offsets / bucket_width), dtype=tf.int32)
            clamped_indices = tf.clip_by_value(bucket_indices, 0,
                                               bucket_count - 1)
            one_hots = tf.one_hot(clamped_indices, depth=bucket_count)
            bucket_counts = tf.cast(
                tf.reduce_sum(input_tensor=one_hots, axis=0), dtype=tf.float64)
            edges = tf.linspace(bucket_min, bucket_max, bucket_count + 1)
            edges = tf.concat([edges[:-1], [bucket_max]], 0)
            edges = tf.cast(edges, tf.float64)
            left_edges = edges[:-1]
            right_edges = edges[1:]
            tensor = tf.transpose(
                a=tf.stack([left_edges, right_edges, bucket_counts]))
        return tf.summary.write(
            tag=tag, tensor=tensor, step=step, metadata=summary_metadata)
