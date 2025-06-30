"""
ARMBR: Artifact-reference multivariate backward regression
===========================================================

ARMBR is a novel method for EEG blink artifact removal with minimal data requirements.

This package provides:
- A standalone `run_armbr()` function for applying the algorithm.
- A scikit-learn and MNE-compatible `ARMBR` class with `.fit()`, `.apply()`, `.plot()` methods.
"""

from .armbr import ARMBR, run_armbr, __version__