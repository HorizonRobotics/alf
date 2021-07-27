Tutorial
========

ALF is designed with **modularization** in mind. Unlike most RL libraries or frameworks
which implement different algorithms by repeating the almost entire RL pipeline in
separate source code files with little code reuse, ALF categorizes RL algorithms
and distills the common structure and logic within each caterogy, so
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
   of crtical details and tricks that help an algorithm's training.* For example,

   * Careful handling of environment step types and their discounts,
   * Temporally independent training of a rollout trajectory if no episodic memory
     is used,
   * A convenient episodic memory infrastructure for easily implementing
     models that are stateful,
   * Automatically applying various input data transformers during rollout
     and training,
   * Specifying different optimizers for different sub-algorithms,
   * Exploiting a variety of tensorboard summary utils,
   * and many more...

Below are a series of examples for writing training files using ALF,
from simple to advanced usage. Each section is a detailed, step-by-step guide
walking through key ALF cencepts. All the tutorial code files can
be found under ``<ALF_ROOT>/alf/examples/tutorial``.

..
    The following section schedule might evolve as the tutorial proceeds

.. toctree::
    :maxdepth: 3

    tutorial/a_minimal_working_example
    tutorial/understanding_ALF_via_the_minimal_working_example
    tutorial/configuring_existing_algorithms
    tutorial/customize_environment_and_wrappers
    tutorial/customize_algorithms
    tutorial/advanced_play_and_alf_snapshot