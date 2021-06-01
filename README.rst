Thermoelastic Stress Analysis (TSA)
---------------------------------------------

.. image:: 
   :target: 

Perform interactive frequency-based analysis for thermal acquisition and data visualization: digital lock-in and real-frequency reconstruction using FFT approach 

The reference signal is reconstruct digitaly, so need the frequency and the phase. The real frequency for the load can be different from the frequency set in the load/acquistion sistem. The digital signal may be out of phase with the infrared response and this will produce an apparent phase shift in the recorde thermoelastic signal.The "freq_detection" method try to reconstruction the reference signal from the thermal video, without the need of acquisiton.

Installing this package
-----------------------

Use `import` to use it by:

.. code-block:: console

   import sys
   sys.path.insert(0, '/path/to/application/app/folder')
   import pyTSA
    
Simple examples
---------------

Here is a simple example on how to use the code:

.. code-block:: python

   fs = 120 # Sampling rate [Hz]
   path = 'video.npy'
   analysis = pytsa.TSA(fa,path)
   fr = 5 # load frequency [Hz]
   analysis.lockin(fr)
   analysis.view_result()



Reference:

