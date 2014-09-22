================
Running Manually
================

The easiest way to do analog input is to make a
:py:class:`pynidaqutils.analog_input.DaqInterface`, which can start a
server (:py:class:`pynidaqutils.analog_input.DaqServer`) and a client
(:py:class:`pynidaqutils.analog_input.DaqClient`) connected to it as
well as stopping them.

However, it is sometimes necessary or desirable to run the server and
client manually.


Server
======

From Command Line
-----------------

The server can be run from the command line by::

    pythonX -m pynidaqutils.analog_input

where ``pythonX`` is the desired python interpreter to run it in. The
output of the ``--help`` option is::

    Usage: analog_input.py [-h] [-v] [--host {localhost, all}] [--port PORT] -d
    
    Starts a server for interacting with a National Instruments DAQ
    listening to the specified port on the indicated host. localhost
    means listening to local connections only, host means listening to
    connections from the network, and both means listen to both. Send
    'close\n' on stdin to close it. When it starts running, the version
    information is returned on stdout. Then, once it is closed, 'Closed'
    is written to stdout.
    
    Options:
      -h, --help     show this help message and exit
      -v, --version  return the version of this program, PyDAQmx, and paramiko
      --host=HOST    where to listen for connections (default is localhost)
      --port=PORT    port to listen to connections on (default is 8163)
      -d, --debug    print communications to stdout

The server can be set to only listen to connections from the same
machine, known as the local loopback network (``--host=localhost``), or
from all networks (``--host=all``). Running it to listen on all
interfaces using the default port (8163)::

    pythonX -m pynidaqutils.analog_input --host=all --port=8163

which returns package version information::

    pynidaqutils: 0.2, PyDAQmx: 1.2.5.2

Stopping the server is just a matter of entering ``'close\n'`` on its
stdin.


From Python
-----------

The server can also be started from a python environment. First, the
necessary packages must be imported::

    >>> import threading
    >>> import asyncore
    >>> import pynidaqutils.analog_input

Then, to start the server listening to the default port (8163) on all
interfaces (host of ``''``)::

    >>> daq_server = pynidaqutils.analog_input.DaqServer(host='', port=8163)
    >>> th = threading.Thread(target=lambda : asyncore.loop(timeout=1.0))
    >>> th.start()

The last two lines are necessary since the socket communication is
handled by the asyncore package, which requires a
:py:class:`threading.Thread` to continually check each socket for data
to transmit and received data.

Then to stop and close the server::

    >>> daq_server.close()
    >>> th.join()


Client
======

The client can only be run from inside a python environment. How it is
run looks just like starting a server from within python. The same
modules must be imported::

    >>> import threading
    >>> import asyncore
    >>> import pynidaqutils.analog_input

Then, the client is started as

    >>> client = DaqClient(host=host, port=8193)
    >>> th = threading.Thread(target=lambda : asyncore.loop(timeout=1.0))
    >>> th.start()

where `host` is the IP address of the machine the server is running
on. If it is the same machine, it should be set to
``'localhost'``. Then, ``client`` can be used as normal. To close
``client``::

    >>> client.exit()
    >>> th.join()
