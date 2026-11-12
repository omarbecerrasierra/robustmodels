"""Sphinx configuration."""

from importlib.metadata import version as package_version

project = "robustmodels"
author = "Omar Becerra Sierra"
copyright = "2026, Omar Becerra Sierra"
release = package_version("robustmodels")

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_sitemap",
    "sphinxext.opengraph",
    "matplotlib.sphinxext.plot_directive",
]
plot_formats = [("png", 150), "svg"]
plot_html_show_source_link = False
exclude_patterns = ["_build"]
myst_heading_anchors = 3

autodoc_member_order = "bysource"
add_module_names = False
napoleon_google_docstring = False

html_theme = "furo"
html_title = "robustmodels: robust statistics for Python"

# Search engines: canonical URLs, a sitemap and page descriptions.
html_baseurl = "https://omarbecerrasierra.github.io/robustmodels/"
sitemap_url_scheme = "{link}"
sitemap_excludes = ["search.html", "genindex.html", "_modules/*"]
ogp_site_url = html_baseurl
ogp_social_cards = {"enable": False}
