Thermoelastic Stress Analysis (TSA)
---------------------------------------------

.. image:: 
   :target: 

Perform interactive frequency-based analysis for thermal acquisition and data visualization: digital lock-in and real-frequency reconstruction using FFT approach 

The reference signal is rebuilt digitaly, so need the frequency and phase. The real frequency for the load can be different from the frequency set in the load/acquistion sistem. The reference signal is may be out of phase with the infrared response and this will produce an apparent phase shift in the recorde thermoelastic signal.The "freq_detection" method try reconstruction of the reference signal from the thermal video, without the acquisiton of the signal.

Installing this package
-----------------------

Use `import` to use it by:

.. code-block:: console

    $ import pyTSA


Simple examples
---------------

Here is a simple example on how to use the code:

.. code-block:: python

   $ fa = 120 # [Hz]
   $ path = 'video.npy'
   $ analysis = pytsa.TSA(fa,path)
   $ fr = 5 # [Hz]
   $ analysis.lockin(fr)
   $ analysis.view_result()



Reference:

