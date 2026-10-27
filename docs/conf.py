"""Build the FREDIS user guides and references from the checked-out implementation."""

from importlib.metadata import version as distribution_version
import inspect
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent / "_ext"))

project = "FREDIS"
author = "Motohiro Ogawa"
copyright = "2026, Motohiro Ogawa"
release = distribution_version("fredis")
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.linkcode",
    "myst_nb",
    "sphinxcontrib.mermaid",
    "fredis_cli",
]
source_suffix = {".rst": "restructuredtext", ".md": "myst-nb"}
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "substitution"]
myst_substitutions = {"release": release}
myst_heading_anchors = 4
nb_execution_mode = "off"
exclude_patterns = [
    "_build",
    "**/.ipynb_checkpoints",
    "**/*.local.*",
    "examples/README.md",
]
autodoc_typehints = "none"
autodoc_member_order = "bysource"
autodoc_class_signature = "separated"

html_theme = "furo"
html_title = "FREDIS documentation"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "source_repository": "https://github.com/MotohiroOGAWA/FREDIS/",
    "source_branch": "main",
    "source_directory": "docs/",
    "navigation_with_keys": True,
}


def omit_generated_dataclass_docstring(app, what, name, obj, options, lines):
    """Keep autodoc's signature, rather than parsing dataclass-generated prose as RST."""
    if (
        what == "class"
        and hasattr(obj, "__dataclass_fields__")
        and lines
        and lines[0].startswith(obj.__name__ + "(")
    ):
        lines.clear()


def setup(app):
    app.connect("autodoc-process-docstring", omit_generated_dataclass_docstring)


def linkcode_resolve(domain, info):
    """Link each documented Python object to its implementation lines."""
    if domain != "py":
        return None
    try:
        obj = sys.modules[info["module"]]
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)
        obj = inspect.unwrap(obj)
        filename = Path(inspect.getsourcefile(obj)).resolve().relative_to(ROOT)
        lines, first = inspect.getsourcelines(obj)
    except (AttributeError, KeyError, TypeError, ValueError, OSError):
        return None
    last = first + len(lines) - 1
    return (
        "https://github.com/MotohiroOGAWA/FREDIS/blob/main/"
        f"{filename.as_posix()}#L{first}-L{last}"
    )
