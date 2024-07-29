How to write docstrings (or RST files)
======================================

Basically our docstrings follow the `RST syntax <https://thomas-cokelaer.info/tutorials/sphinx/rest_syntax.html>`_
for hypermedia and the `Google style <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/#google-vs-numpy>`_
for documenting function arguments and returns. That means, you can write anything
that's supported by the RST syntax in a docstring, such as tables, figures,
math equations, hyperlinks, etc.

Note that indent and line break are very important when writing RST!

Raw docstring
-------------

Sometimes if there is backslash ``\`` in your docstring, chars will be escaped
when sphinx tries to interpret the string. So if you have latex commands or
backslashes, make sure to append a char 'r' to the front of the docstring. For
example:

.. code-block:: python

    r"""This is a raw docstring.

    The char 'r' should be appended to the front to mark it as raw. No escaping
    will be done.
    """

Math
----

You can use latex syntax to render math equations.

- Inline math uses::

    :math:`\log(3)`

- For a separate paragraph use math block::

    .. math::

        \begin{array}{lr}
            &3 + 5\\
            =&8\\
        \end{array}

  which will generate

  .. math::

    \begin{array}{lr}
        &3 + 5\\
        =&8\\
    \end{array}

Code segment
------------

Sometimes in a docstring we want to refer to the source code. It is best if code
can be visually different from plain text.

- Inline code::

    ``algorithm.train_step()`` or code:`algorithm.train_step()`

  which renders :code:`algorithm.train_step()`.

- Code block::

    .. code-block:: python

        def f(x):
            print(x)

    .. code-block:: bash

        cat file > /tmp/output

Class init docstring
--------------------

By default, sphinx will combine the docstring right below a class name with the
docstring of the ``__init__`` function. For example, the two docstrings below

.. code-block:: python

    class Algorithm(object):
        """An algorithm base class."""
        def __init__(self, observation_spec, action_spec):
            """The init function of Algorithm.
            """

will be combined into one description of the class constructor::

    An algorithm base class.
    The init function of Algorithm.

So try to avoid duplicate sentences at the two places.

Arguments
---------

The argument list should always starts with a string "Args:" with a line break,
after which the arguments will be listed and explained::

    Args:
        arg_name1 (arg_type1): description1
        arg_name2 (arg_type2): this is a long description that we have to change
            the line.
        ...
        arg_nameN (arg_typeN): descriptionN

Note that the indents are very important. Argument list should be one level lower
than "Args:". When changing a line for an argument's description, make sure to
indent the rest of the paragraph.

Returns
-------

For any python function, there is only one return. Even we have multiple results
output by a function, it's just a tuple consisting of multiple components. Thus
sphinx only supports rendering one return.

The return list should always starts with a string "Returns:" with a line break,
after which only **one** return type should be documented (no need to *name* the
return)::

    Returns:
        torch.Tensor: an output tensor

If the return is not a nest, then the description should just follow the return
type. If, however, we want to document a nest with its different fields, then one
way is to make a bullet list under the return type (notice the indent!)::

    Returns:
        AlgStep:
        - output (nested Tensor): policy action
        - state (nested Tensor): policy state

For an example, see the docstring and rendered result of ``RLAlgorithm.predict_step()``.

Raises
------

You can also document what exceptions a function will raise by "Raises:"::

    Raises:
        AttributeError: The ``Raises`` section is a list of all exceptions
            that are relevant to the interface.
        ValueError: If ``arg2`` is equal to ``arg1``.

Note that do not put quotes around error classes due to the bug fixed in a recent
Github `PR <https://github.com/sphinx-doc/sphinx/pull/6237>`_. Otherwise the doc
won't compile.

Notes and warnings
------------------

It's perfectly fine to add note and warning sections in a docstring. The rendered
text sections will be highlighted by colors.

::

    .. note::
        This is note text. Use a note for information you want the user to
        pay particular attention to.

        If note text runs over a line, make sure the lines wrap and are indented
        to the same level as the note tag. If formatting is incorrect, part of
        the note might not render in the HTML output.

        Notes can have more than one paragraph. Successive paragraphs must
        indent to the same level as the rest of the note.

.. note::
    This is note text. Use a note for information you want the user to
    pay particular attention to.

    If note text runs over a line, make sure the lines wrap and are indented
    to the same level as the note tag. If formatting is incorrect, part of
    the note might not render in the HTML output.

    Notes can have more than one paragraph. Successive paragraphs must
    indent to the same level as the rest of the note.

::

    .. warning::
        This is warning text. Use a warning for information the user must
        understand to avoid negative consequences.

        Warnings are formatted in the same way as notes. In the same way, lines
        must be broken and indented under the warning tag.

.. warning::
    This is warning text. Use a warning for information the user must
    understand to avoid negative consequences.

    Warnings are formatted in the same way as notes. In the same way, lines
    must be broken and indented under the warning tag.


More examples
-------------

For more examples, refer to
`<https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`_.

