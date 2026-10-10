"""FREDIS documentation; the product overview is shared with Workbench."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
project = 'FREDIS'
author = 'Motohiro Ogawa'
release = '0.1.0'
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx.ext.viewcode',
              'sphinx_autodoc_typehints', 'myst_nb', 'jupyter_sphinx']
html_theme = 'furo'
html_title = 'FREDIS documentation'
source_suffix = {'.rst': 'restructuredtext', '.md': 'myst-nb'}
myst_enable_extensions = ['colon_fence', 'deflist']
nb_execution_mode = 'off'
exclude_patterns = ['_build', '**/.ipynb_checkpoints']
html_static_path = []
