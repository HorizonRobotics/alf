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

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../'))

# -- Project information -----------------------------------------------------

project = 'Agent Learning Framework (ALF)'
copyright = '2023, HorizonRobotics'
author = 'HorizonRobotics'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc', 'sphinx.ext.viewcode', 'sphinxcontrib.napoleon',
    'sphinx_autodoc_typehints'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'
html_theme_option = {'logo_only': True}
html_logo = "_static/logo.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Concat the doctstrings of Class and __init__
autoclass_content = 'both'

# If true, the current module name will be prepended to all description
# unit titles, e.g., alf.algorithms.algorithm.Algorithm
add_module_names = False

# use index.rst as the entry doc
master_doc = "index"

# API generation command:
# sphinx-apidoc -f -o api ../alf `find .. -name '*_test.py'` ../alf/examples --templatedir _templates

# -- Automatically generate API documentation --------------------------------


def run_apidoc(_):
    import glob
    from sphinx.ext import apidoc

    # ignore all files with "_test.py" suffix
    ignore_paths = glob.glob("../alf/**/*_test.py", recursive=True)
    # ignore files in the examples and bin directories
    ignore_paths.append("../alf/examples")

    argv = [
        "--force",  # Overwrite output files
        "--follow-links",  # Follow symbolic links
        #"--separate",  # Put each module file in its own page
        "--module-first",  # Put module documentation before submodule
        "--templatedir",
        "_templates",  # use our customized templates
        "-o",
        "api",  # Output path
        "../alf"  # include path
    ] + ignore_paths

    apidoc.main(argv)


def setup(app):
    app.connect('builder-inited', run_apidoc)


# HACK: build penv before building html docs
cur_dir = os.path.realpath(os.path.abspath('../'))
os.system(f"cd {cur_dir}/alf/environments; python3 make_penv.py")
