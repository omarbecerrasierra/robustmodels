"""Sphinx configuration."""

from importlib.metadata import version as package_version

project = "robustmodels"
author = "omarbecerrasierra"
copyright = "2026, omarbecerrasierra"
release = package_version("robustmodels")

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
exclude_patterns = ["_build"]

autodoc_member_order = "bysource"
add_module_names = False
napoleon_google_docstring = False

html_theme = "furo"
html_title = f"robustmodels {release}"
