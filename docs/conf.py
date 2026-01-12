"""Sphinx configuration for the Sentim project."""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure the project root and `src` directory are on the Python path for autodoc
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

project = "Sentim"
author = "Ilias Laoukili"
release = "0.1.0"
version = release
copyright = f"{datetime.now().year}, {author}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_attr_annotations = True

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "inherited-members": False,
}

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

language = "en"
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = "Sentim Documentation"

myst_enable_extensions = ["colon_fence"]
