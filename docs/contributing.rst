
Contribute to ALF
=================

Workflow
--------

1. Clone ALF

2. Install code style tools

.. code-block:: bash

    pip install pre-commit==1.17.0
    pip install cpplint==1.4.4
    pip install pydocstyle==4.0.0
    pip install pylint==2.3.1
    pip install yapf==0.28.0
    sudo apt install clang-format


3. At your local repo root, run

.. code-block:: bash

    pre-commit install

4. Make local changes

.. code-block:: bash

    git co -b PR_change_name origin/master


Make change to your code and test. You can run all the existing unittests
by the following command:

.. code-block:: bash

    python -m unittest discover -s alf -p "*_test.py" -v

Then commit your change to the local branch using :code:`git commit`.

5. Make pull request:

For Horizon team members or collaborators with the access, you can directly
push a branch to the ALF repo and then create a PR:

.. code-block:: bash

    git push origin PR_change_name

For the public without the access, you need to first fork the ALF repo, push
a local change to your forked repo, and then create a PR from there:

.. code-block:: bash

    git push <your_fork> PR_change_name

In either case, a PR can be created from the Github website.

6. Change your code based on review comments. The new change should be added
as NEW commit to your previous commits. Do not use :code:`--amend` option for the
commit, because then you will have to use :code:`-f` option to push your change to
github and review will be more difficult because the new change cannot
be separated from previous change. For the same reason, if you need to incorporate
the latest code from master, please avoid rebase. If your code is rebased, you won't
be able to push your PR without using `-f` option. So you should use :code:`git pull --rebase=false`
instead of :code:`git pull`. You can globally change the default behavior of :code:`git pull`
to not rebase by setting :code:`git config --global pull.rebase false` so that you can simply
use :code:`git pull`.

Coding standard
---------------

We follow `Google's coding style <http://google.github.io/styleguide/pyguide.html>`_.
Please comment all the public functions with the following style:

.. code-block:: python

    def func(a, b):
        """Short summary of the function

        Detailed explanation of the function. Including math equations and
        references. The explanation should be detail enough for the user to have a
        clear understanding of its function without reading its implementation.

        Args:
            a (type of a): purpose
            b (type of b): purpose
        Returns:
            return type:
            - return value1 (type 1): purpose
            - return value2 (type 2): purpose
        """

**NOTE** that in recent versions of Python, `type hints <https://docs.python.org/3/library/typing.html>`_ are supported to tag the types of the arguments to a functions and the type of the return value. When possible, it is **recommended** to add the type hints to the input arguments. When type hints are present, it is no longer required to have typing labels in the docstring. For example, the above code would become:

.. code-block:: python

    from typing import Tuple

    def func(a: <type of a>, b: <type of b>) -> Tuple[<type 1>, <type 2>]:
        """Short summary of the function

        Detailed explanation of the function. Including math equations and
        references. The explanation should be detail enough for the user to have a
        clear understanding of its function without reading its implementation.

        Args:
            a: purpose
            b: purpose
        Returns:
            return type:
            - return value1: purpose
            - return value2: purpose
        """

For a comprehensive guide on how to write docstrings for public functions, see
:doc:`notes/howto_docstring`.

