Overview
========

This Python package provides utilities to work with National Instruments
DAQs on top of the PyDAQmx package. Utilities include scanning the system
for attached DAQs and the ability to start and control an analog input
acquisition on another process on the local or a remote machine and grab
the acquired data (this is useful when one wants to work in a 64 bit
environment because the NI drivers can only be interfaced from a 32 bit
environment, which in this package is isolated to another process).

The package's source code is found at
https://github.com/frejanordsiek/pynidaqutils

Installation
============

This package may not work on Python < 3.0.

This package requires the PyDAQmx package.

To install pynidaqutils, download the package and run the command::

    python3 setup.py install

