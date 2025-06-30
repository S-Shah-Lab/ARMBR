import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

project = 'ARMBR'
html_title = 'ARMBR Documentation'
html_theme = 'furo'  # or 'sphinx_book_theme'

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",   # This allows Google-style and NumPy-style docstrings
    "sphinx.ext.viewcode",   # Adds [source] links
    "sphinx.ext.intersphinx" # For linking to NumPy, SciPy, etc.
]

autoclass_content = "both"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "inherited-members": True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = True
