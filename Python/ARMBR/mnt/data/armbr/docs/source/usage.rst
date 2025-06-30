Usage Guide
===========

This section describes how to use ARMBR in different workflows.

Basic Example
-------------

.. code-block:: python

   from armbr import ARMBR
   import mne

   # Load your raw EEG data
   raw = mne.io.read_raw_fif('your_file.fif', preload=True)

   # Fit ARMBR
   armbr = ARMBR().fit(raw, blink_chs=['Fp1', 'Fp2'])

   # Apply blink removal
   raw_clean = armbr.apply(raw)

   # Plot component properties
   armbr.plot()

You can also use the functional version:

.. code-block:: python

   from armbr import run_armbr

   cleaned, *_ = run_armbr(raw.get_data(), blink_chs=[0, 1], sfreq=raw.info['sfreq'])
