Examples
========

1. Minimal Example with Raw EEG
-------------------------------

.. code-block:: python

   raw = mne.io.read_raw_fif("subject_raw.fif", preload=True)
   armbr = ARMBR().fit(raw, blink_chs=["Fp1", "Fp2"])
   raw_clean = armbr.apply(raw)
   armbr.plot()

2. Using ARMBR with Epochs
--------------------------

.. code-block:: python

   epochs = mne.Epochs(raw, events, event_id)
   armbr = ARMBR().fit(epochs, blink_chs=["Fp1", "Fp2"])
   epochs_clean = armbr.apply(epochs)

3. Functional API
-----------------

.. code-block:: python

   from armbr import run_armbr

   eeg = raw.get_data()
   sfreq = raw.info["sfreq"]
   blink_chs = [0, 1]

   cleaned, alpha, mask, comp, pattern, proj = run_armbr(eeg, blink_chs, sfreq)
