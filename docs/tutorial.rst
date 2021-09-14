Tutorial
========

ALF is designed with **modularization** in mind. Unlike most RL libraries or frameworks
which implement different algorithms by repeating the almost entire RL pipeline in
separate source code files with little code reuse, ALF categorizes RL algorithms
and distills the common structure and logic within each category, so
that each algorithm only needs to implement or override its own exclusive logic.

Usually to create an ALF job, a user is expected to:

1. write a configuration file that configures both the environment (e.g., num
   of parallel envs, task name, etc) and the algorithm (e.g., training schedule,
   algorithm networks, etc);
2. or for customized algorithms, override some (hopefully a small number) functions
   of ALF provided algorithms.

After this, the ALF job can be executed to perform training, evaluation, and more.

RL training is a very complex process; some details are very tricky and error-prone,
and you certainly don't want to implement (or even touch!) the entire pipeline
each time. ALF strives to provide best RL practices. Thus ALF's modularization
provides at least two benefits:

1. *Fast prototyping without worrying about breaking things.*

   * If you just want to explore new models/networks of an existing
     algorithm on an existing task, no source code will be modified; only a python
     configuration file has to be set up.
   * If you want to either modify an existing environment or improve upon an
     algorithm, usually you only need to override some function to do so. This
     ensures the remaining part of the pipeline is unaffected.

2. *Reusing ALF's carefully designed training pipeline which contains a ton
   of critical details and tricks that help an algorithm's training.* For example,

   * Careful handling of environment step types and their discounts,
   * Temporally independent training of a rollout trajectory if no working memory
     is used,
   * A convenient working memory infrastructure for easily implementing
     models that are stateful,
   * Automatically applying various input data transformers during rollout
     and training,
   * Specifying different optimizers for different sub-algorithms,
   * Exploiting a variety of Tensorboard summary utils,
   * and many more...

Below are a series of examples for writing training files using ALF,
from simple to advanced usage. Each chapter is a detailed, step-by-step guide
walking through key ALF concepts. All the tutorial code files can
be found under ``<ALF_ROOT>/alf/examples/tutorial``.

.. note::

    This tutorial won't cover the technical details of different algorithms and
    models, as we assume the user learns them from other resources, e.g., the
    original papers. We only focus on how to use ALF as a tool to write them.

..
    The following chapter schedule might evolve as the tutorial proceeds

.. toctree::
    :maxdepth: 3

    tutorial/a_minimal_working_example
    tutorial/understanding_ALF_via_the_minimal_working_example
    tutorial/algorithm_interfaces
    tutorial/summary_metrics_and_tensorboard
    tutorial/environments_and_wrappers
    tutorial/alf_snapshot_and_advanced_play
    tutorial/customize_algorithms
    tutorial/customize_training_pipeline