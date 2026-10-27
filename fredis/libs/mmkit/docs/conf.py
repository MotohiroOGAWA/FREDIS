import os
import sys

import importlib
import inspect

sys.path.insert(0, os.path.abspath(".."))


def get_summary(module_name: str, obj_name: str, member_name: str) -> str:
    """
    Get the first summary line of a class member docstring.

    Parameters
    ----------
    module_name : str
        Module name such as ``"mmkit.Formula"``.
    obj_name : str
        Class name such as ``"Formula"``.
    member_name : str
        Member name such as ``"exact_mass"``.

    Returns
    -------
    str
        First non-empty line of the member docstring.
        Returns an empty string if unavailable.
    """
    candidates = [module_name]
    if "." in module_name:
        candidates.append(module_name.rsplit(".", 1)[0])

    for mod_name in candidates:
        try:
            module = importlib.import_module(mod_name)
            obj = getattr(module, obj_name)
            member = getattr(obj, member_name)

            if isinstance(member, property):
                doc = inspect.getdoc(member.fget)
            else:
                doc = inspect.getdoc(member)

            if not doc:
                continue

            for line in doc.splitlines():
                line = line.strip()
                if line:
                    return line
        except Exception:
            continue

    return ""

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'mmkit'
copyright = '2026, Motohiro Ogawa'
author = 'Motohiro Ogawa'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
autosummary_context = {
    "get_summary": get_summary,
}
autodoc_typehints = "description"

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
html_css_files = ['custom.css']
