ARMBR: Artifact-Reference Multivariate Backward Regression
===========================================================

Welcome to the official documentation of the **ARMBR** algorithm.

ARMBR (Artifact-Reference Multivariate Backward Regression) is a robust EEG blink artifact removal technique that works effectively with minimal calibration data and as few as two frontal electrodes (e.g., FP1 and FP2). It is well-suited for pediatric and clinical EEG where long recordings or dense montages may not be available.

.. note::
   ARMBR is designed to be compatible with both **MNE-Python** and **scikit-learn**, and it can be used as a standalone function or as a class object with `.fit()`, `.apply()`, and `.plot()` methods.

**Main Features**
-----------------

- Works with raw EEG signals, epochs, or preprocessed data
- Requires only reference channels for eye blinks (e.g., FP1/FP2)
- Minimal calibration data required (as little as 5–10 seconds)
- Compatible with pipelines based on MNE or scikit-learn
- Supports masking and visualization
- Fast and interpretable, no ICA or source modeling needed

**Documentation Structure**
---------------------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   usage
   examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   modules
   armbr

.. toctree::
   :maxdepth: 1
   :caption: Project Info

   license
   changelog

----

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
