#!/usr/bin/env python
#
# xcdat documentation build configuration file, created by
# sphinx-quickstart on Fri Jun  9 13:47:02 2017.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another
# directory, add these directories to sys.path here. If the directory is
# relative to the documentation root, use os.path.abspath to make it
# absolute, like shown here.
#
import os
import sys
from typing import Dict

import sphinx_autosummary_accessors

sys.path.insert(0, os.path.abspath(".."))  # noqa: I001, I003
import xcdat  # noqa: I001, E402

# -- General configuration ---------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autosummary_accessors",
    "sphinx_copybutton",
    "nbsphinx",
]

# autosummary and autodoc configurations
autosummary_generate = True

# We need to mock "ESMF" imports because xesmf depends on ESMpy which depends on
# ESMF (has C modules). The Read The Docs build system does not have the
# dependencies for building these C modules which causes a Python ImportError
# with xesmf.
# Related issue: https://github.com/readthedocs/readthedocs.org/issues/5512
# Solution: https://docs.readthedocs.io/en/stable/faq.html#why-do-i-get-import-errors-from-libraries-depending-on-c-modules
autodoc_mock_imports = ["ESMF"]
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": True,
}
autodoc_typehints = "none"

# Napoleon configurations
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_preprocess_types = True

# sphinx-copybutton configurations
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates", sphinx_autosummary_accessors.templates_path]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "xCDAT"
copyright = "2022, xCDAT Developers"
author = "xCDAT Developers"

# The version info for the project you're documenting, acts as replacement
# for |version| and |release|, also used in various other places throughout
# the built documents.
#
# The short X.Y version.
version = xcdat.__version__
# The full version, including alpha/beta/rc tags.
release = xcdat.__version__

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "demos/1-25-23-cwss-seminar/xsearch-xcdat-example.ipynb",
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False


# -- Options for HTML output -------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_book_theme"

# Theme options are theme-specific and customize the look and feel of a
# theme further.  For a list of options available for each theme, see the
# documentation.
#
# sphinx_book_theme configurations
# https://sphinx-book-theme.readthedocs.io/en/latest/configure.html
html_logo = "_static/xcdat-logo.png"
html_title = "xCDAT Documentation"
html_theme_options = {
    "repository_url": "https://github.com/xCDAT/xcdat",
    "repository_branch": "master",
    "path_to_docs": "docs",
    "use_edit_page_button": True,
    "use_repository_button": True,
    "use_issues_button": True,
    "use_download_button": True,
    "use_fullscreen_button": True,
    "logo_only": True,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


# -- Options for HTMLHelp output ---------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "xcdatdoc"


# -- Options for LaTeX output ------------------------------------------

latex_elements: Dict[str, str] = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass
# [howto, manual, or own class]).
latex_documents = [
    (master_doc, "xcdat.tex", "xCDAT Documentation", "Tom Vo", "manual"),
]


# -- Options for manual page output ------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "xcdat", "xCDAT Documentation", [author], 1)]


# -- Options for Texinfo output ----------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "xcdat",
        "xCDAT Documentation",
        author,
        "xcdat",
        "One line description of project.",
        "Miscellaneous",
    ),
]


def html_page_context(app, pagename, templatename, context, doctree):
    # Disable edit button for docstring generated pages
    if "generated" in pagename:
        context["theme_use_edit_page_button"] = False


def setup(app):
    app.connect("html-page-context", html_page_context)
