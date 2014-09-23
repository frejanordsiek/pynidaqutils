Overview
========

This Python package provides utilities to work with National Instruments
DAQs on top of the PyDAQmx package. Utilities include scanning the system
for attached DAQs and the ability to start and control an analog input
acquisition in another python process (server) on the local or a remote
machine and return the acquired data (this is useful when one wants to
work in a 64 bit environment because the NI drivers can only be
interfaced from a 32 bit environment, which in this package is isolated
to another process) to the client.

.. warning::
   
   When running the server and clients on different machines, the
   communication connection between the two over a socket is not
   encrypted nor is any sort of authentication used. Thus, this feature
   should only be used on a trusted private network.

The package's source code is found at
https://github.com/frejanordsiek/pynidaqutils


Installation
============

This package is supported for Python version 2.6 and later.

This package requires the numpy package for both the server and the
client, the PyDAQmx package on the server, and the paramiko package on
the client if the client and the server are on different machines
(remote acquisition).

.. warning::
   
   The server cannot operate on a machine that doesn't have niDAQmxBase
   installed and the ``lsdaq`` in the executable path. niDAQmx, the
   default on Windows, will not work since it does not have the
   ``lsdaq`` utility. Note that the PyDAQmx package must be one with
   niDAQmxBase support, so it must be a version greater than ``1.2.5.2``
   (highest version on PyPi as of 2014-09-23) or be one of the bleeding
   edge versions/forks on github such as the official one,
   `clade/PyDAQmx <https://github.com/clade/PyDAQmx>`_, or forks
   `MarcoForte/PyDAQmx <https://github.com/MarcoForte/PyDAQmx>`_ and
   `frejanordsiek/PyDAQmx:DAQmxBase_support <https://github.com/frejanordsiek/PyDAQmx/tree/DAQmxBase_support>`_.
   Note that out of the forks, the first one is the most up to date.

To install pynidaqutils, download the package and run the command::

    pythonX setup.py install

where X is the particular python you want to install it for.


Getting Started
===============

First, import the modules::

    >>> import pynidaqutils
    >>> import pynidaqutils.analog_input

The :py:mod:`pynidaqutils` module had hardware information for various
National Instrument DAQ's as well as a function that returns which DAQ's
are attached to the computer::

    >>> pynidaqutils.list_daqs()
    {'Dev1': {'hw': 'USB0::0x3923::0x7270::0186F411::RAW', 'type': 'NI USB-6211'}}

The easiest way to do analog input is to make a
:py:class:`pynidaqutils.analog_input.DaqInterface`, which can start a
server (:py:class:`pynidaqutils.analog_input.DaqServer`) and a client
(:py:class:`pynidaqutils.analog_input.DaqClient`) connected to it as
well as stopping them::

    >>> di = pynidaqutils.analog_input.DaqInterface()

When starting a server, the location, or `host`, of where it is to run
and what is the name of the command to run python called must be
specified. To run on the same computer and say run it on ``python3``,
one would do::

    >>> di.start_server(host='localhost', python_command='python3')
    {'PyDAQmx': '1.2.5.2', 'paramiko': '1.14.0', 'pynidaqutils': '0.2'}

and it then returns the package versions of pynidaqutils,
`PyDAQmx <https://pypi.python.org/pypi/PyDAQmx>`_, and
`paramiko <https://pypi.python.org/pypi/paramiko>`_ on the server. To
instead run it on another machine, `host` must be set to the IP address
of the other machine. That machine must be running an SSH server on port
22 and the `paramiko <https://pypi.python.org/pypi/paramiko>`_ package
is needed to connect to it to start the server. The username and
password to log into the server by SSH must also be provided::

    >>> di.start_server(host='192.168.1.100', python_command='python3'
    ...                 username='daq', password=passwd)
    {'PyDAQmx': '1.2.5.2', 'paramiko': '1.14.0', 'pynidaqutils': '0.2'}

where ``passwd`` holds the password.

A convenience function is provided that will make a client
(:py:class:`pynidaqutils.analog_input.DaqClient`), connect it to the
server, and put store it at ``di.client``::

    >>> di.start_client()
    True

where the ``bool`` returned indicates whether the client was created
and connected successfully or not.

``di.client.daq_list`` contains the output of
``pynidaqutils.list_daqs()`` on the server. ``di.client.scan_daqs()``
will cause the server to re-scan for DAQ's and ``di.client.daq_list`` to
be updated.

In order to acquire, the acquisition must be setup. If we want to
acquire on device ``'Dev1'``, get the value at each channel at a rate
of 100 Hz by averaging 6 measurements together, acquire continuously
till we explicitly tell the server to stop (denoted by ``-1``), and
return the measured values as single precision floating point numbers
(``numpy.float32``) on differential channels 0 and 2 with input ranges
of -10 to +10 V; one would do::

    >>> channels = [{'channel': 0, 'voltage': 10.0, 'termination': b'Diff'},
    ...             {'channel': 2, 'voltage': 10.0, 'termination': b'Diff'}]
    >>> success, = di.client.setup_daq(b'Dev1', frequency=100.0,
    ...                                averaged=6, count=-1,
    ...                                tp='single', channels=channels)
    True

The other outputs of the function indicate what the actual command and
configuration sent to the server looked like in the event that they are
needed. Then, to start acquisition::

    >>> di.client.start_daq()
    True

Whether it was successfully started or not is returned. While it is
acquiring, ``di.client.is_acquiring`` is ``True``. While acquisition is
occurring, the server is transferring it to the client over a socket in
blocks. All the blocks acquired so far are obtained by::

    >>> data, lg = di.client.get_new_data()

``data`` is ``None`` if no blocks have been acquired, and a ``list`` of
``numpy.ndarray`` if there have been. Each bock is a ``numpy.ndarray``
where the columns are the different channels in the order given to
``setup_daq`` and the rows are successive time steps. ``lg`` is a
``list`` of ``tuple`` with a ``tuple`` for each block. The ``tuple``
specify the zero-indexed starting and ending sample number for the
respective block. In the very off chance that the blocks get out of
order or a block is lost, ``lg`` can be used to figure that out and
reorder if necessary.

When done acquiring, call::

    >>> di.client.stop_daq()
    True

to stop the DAQ and transmit the last data blocks. Whether stopping it
was successful or not is returned. At this point, the DAQ can be
reconfigured and acquisition started again.

The client and server are closed by::

    >>> di.stop_client()
    True
    >>> di.stop_server()
    True

Though, calling ``stop_server`` will automatically call ``stop_client``
if the client was started.
