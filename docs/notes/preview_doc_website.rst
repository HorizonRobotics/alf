How to preview this website locally
===================================


This website is automatically generated using `Sphinx
<https://pythonhosted.org/an_example_pypi_project/sphinx.html>`_ on
`<readthedocs.org>`_. It will get updated by RTD once a commit is pushed to ALF.
So normally an ALF user/developer doesn't have to worry about keeping this
website up-to-date every time some code change is made. For a complete setup
process of using RTD, please refer to this `PR
<https://github.com/HorizonRobotics/alf/pull/502>`_.

However, if you would like to preview the changes on this website locally, there
are generally two ways of doing this.

1. Non-VSCode users:

.. code-block:: bash

    cd $ALF_ROOT/docs
    sphinx-apidoc -f -o api ../alf `find .. -name '*_test.py'` ../alf/examples --templatedir _templates
    make html
    cd _build/html

Then open the file :code:`index.html` with a browser.

2. VSCode users:

Install the extension `reStructuredText` and open your vscode settings file
:code:`settings.json` and add the following configuration:

.. code-block:: json

    {
        "restructuredtext.builtDocumentationPath" : "${workspaceRoot}/docs/_build/html",
        "restructuredtext.confPath"               : "${workspaceFolder}/docs",
        "restructuredtext.updateOnTextChanged"    : "true",
        "restructuredtext.updateDelay"            : 1000
    }

Then open an :code:`.rst` file and use `shift+alt+r` (on Mac `cmd+shift+r`)
to open the preview. The extension will automatically generate the api files
and build html files.


Troubleshooting
---------------


1. If previewing fails with error `Could not import extension sphinxcontrib.napoleon`, try

.. code-block:: bash

    pip install sphinxcontrib-napoleon sphinx_rtd_theme

2. If preview pane is empty, wait for a while, generation can take time.

3. After file change, close preview tab and then generate preview again in
vscode to view the change.