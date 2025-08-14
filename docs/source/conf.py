# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'UH2SC'
copyright = '2025, Daniel L. Villa'
author = 'Daniel L. Villa'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration



extensions = [
   'sphinx.ext.autodoc',
]

templates_path = ['_templates']
exclude_patterns = []

html_theme_options = {
    "light_logo": "Thunderbird_Black.png",
    "dark_logo": "Thunderbird_White.png",
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
