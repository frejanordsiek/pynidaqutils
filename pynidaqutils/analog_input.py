#!/usr/bin/env python3-32bit

# Copyright (c) 2014, Freja Nordsiek
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

""" Connects to a National Instruments DAQ and reads data from it.

Allows the creation of a server-client pair to acquire analog input data
from an NI DAQ on the server and sent the data to the client over a
socket. The server is accessed by running this module as a script or
explicitly setting up an instance of ``DaqSever``. The client is
``DaqClient``. ``DaqInterface`` is a convenience class for starting and
stopping a server and a client.

"""

import copy
import sys
import ctypes
import argparse
import socket
import asyncore
import asynchat
import threading
import subprocess
import math
import datetime
import binascii

import numpy as np

import pynidaqutils

# Try to import PyDAQmx, which could easily fail, and set a flag
# according to whether it was imported or not.
have_PyDAQmx = True
try:
    import PyDAQmx
except:
    have_PyDAQmx = False

# Need to keep track of the default host and port for the server.
default_host = 'localhost'
default_port = 8163


class ReadAnalogInputThread(threading.Thread):
    """ Thread to read analog input data from the DAQ on the server.

    It calls functions on `server` directly to transmit data and
    messages to the client. Thread is aborted by calling ``abort``.

    Parameters
    ----------
    analog_input : PyDAQmx.Task
        The ``PyDAQmx.Task`` to do the desired analog input. Must not
        be started yet.
    server : DaqServerHandler
        The ``DaqServerHandler`` running the acquisition (the creator
        of this thread).
    device : bytes
        The device label of the NI DAQ being acquired on.
    interval : float
        The time interval of data to grab from the DAQ at a time.
    number_channels : int
        The number of channels being acquired on.
    frequency : float
        The acquisition frequency.
    samples_at_a_time : int
        The number of samples in each channel to transmit in each block
        back to the client.
    convert_to_single : bool
        Whether or not to convert the acquired data to single precision
        (numpy dtype float32) before transmitting to the client.

    Attributes
    ----------
    trigger_time

    """
    def __init__(self, analog_input, server, device, interval,
                 number_channels, frequency, samples_at_a_time,
                 convert_to_single):
        #multiprocessing.Process.__init__(self)
        threading.Thread.__init__(self)
        self.analog_input = analog_input
        self.server = server
        self.device = device
        self.interval = interval
        self.number_channels = number_channels
        self.frequency = frequency
        self.samples_at_a_time = samples_at_a_time
        self.convert_to_single = convert_to_single
        # Start out with an empty trigger time.
        self._trigger_time = []
        # Start out with an empty buffer of read values.
        self._buf = np.empty(shape=(0, number_channels),
                             dtype={True: 'float32',
                             False: 'float64'}[convert_to_single])
        # Keep track of the number of samples read and the index of the
        # sample at the beginning of the buffer.
        self._number_samples = 0
        self._begin_buf_sample_number = 0
        # Need a flag to be able to know when to abort.
        self._abort_now = False

    def run(self):
        """ Start the thread."""
        # Keep track of the number of samples read and the index of the
        # sample at the beginning of the buffer.
        self._number_samples = 0
        self._begin_buf_sample_number = 0
        # Start acquisition and then get the trigger time.
        self.analog_input.StartTask()
        dt = datetime.datetime.utcnow()
        self._trigger_time = list(dt.timetuple())[:5]
        self._trigger_time.append(float(dt.second) \
            + 1e-6 * float(dt.microsecond))

        # Send a message back to the client (using the server) saying
        # when we started.
        self.server.push_line((('Started at {0} {1} {2} {3} {4}' \
            + ' {5}').format(*self._trigger_time)).encode())

        # Read data continuously till either the task is done (acquiring
        # a finite number of samples and all are taken) or we were told
        # to abort. The read data is then written (up to one remaining
        # partial block). Each cycle sleeps for an interval.
        try:
            while True:
                if self._abort_now:
                    raise ValueError('Aborting thread.')
                self.read_data(int(math.ceil( \
                    self.interval * self.frequency)))
                self.transmit_data(False)
        except:
            if self._abort_now:
                # Stop the task as we were told to abort.
                self.server.push_line(b'Aborting acquisition.')
            else:
                self.server.push_line(b'Error: acquisition.')
        finally:
            self.analog_input.StopTask()
            self.transmit_data(True)
            self.analog_input.ClearTask()
            self.analog_input = None
            PyDAQmx.DAQmxResetDevice(self.device)

        # Send a message that acquisition is complete.
        self.server.push_line(b'Finished acquisition.')

    def abort(self):
        """ Abort the thread (non-blocking)."""
        self._abort_now = True

    @property
    def trigger_time(self):
        """ The time on the server that the acquisition started.

        list

        6-element ``list`` of 5 ``int`` and one ``float`` that specify
        the time (UTC) on the server that the acquisition started in
        year, month, day, hour, minute, seconds order.

        .. warning::

           The clock on the server could be very different than the
           clock on the client if they are different machines, so
           the time may not make much sense when looking at the clock
           on the client machine.

        """
        return copy.deepcopy(self._trigger_time)

    def read_data(self, count):
        """ Reads samples from the DAQ and appends them to the buffer.

        Parameters
        ----------
        count : int
            The number of samples per channel to read.

        """
        # We need an integer that will have the number of samples read
        # written to it (DAQmx library call semantics).
        number_read = ctypes.c_int32()

        # Need a buffer to hold all the samples we are going to grab.
        n = count*self.number_channels
        data = np.zeros((count * self.number_channels,), dtype='float64')

        # Read whatever samples are available. They will be ordered by
        # sample time (one channel after another, before the next time)
        # and the number actually read will be stuffed in number_read.
        self.analog_input.ReadAnalogF64(count, 10.0, \
            PyDAQmx.DAQmx_Val_GroupByScanNumber,
            data, n, PyDAQmx.byref(number_read), None)

        # To make life easier, and avoid potential problems, number_read
        # needs to be converted to a normal int.
        number_read = number_read.value

        # If any samples were read, then we need to increment the
        # counter for the number that have been read, change the shape
        # of data so that each row is a sample time and each column is a
        # channel, and then add it to the end of the buffer (converting
        # to single precision if instructed.
        if number_read != 0:
            self._number_samples += number_read
            data.shape = (n//self.number_channels,
                          self.number_channels)
            if self.convert_to_single:
                self._buf = np.vstack((self._buf, \
                    np.float32(data[0:number_read,:])))
            else:
                self._buf = np.vstack((self._buf,
                                      data[0:number_read,:]))

    def transmit_data(self, write_all):
        """ Transmit acquired data to the client.

        Transmits acquired data in the buffer to the client block by
        block.

        Parameters
        ----------
        write_all : bool
            Whether or not to transmit the last partially full block of
            data if there is any.

        """
        # As long as the buffer is longer than the number of samples we
        # are to write at a time, remove a block of the first set of
        # samples, transmit it, and then increment the sample number
        # counter for the buffer.
        while self._buf.shape[0] >= self.samples_at_a_time:
            block = self._buf[:self.samples_at_a_time, :]
            self._buf = self._buf[self.samples_at_a_time:, :]
            self._transmit_block(block, self._begin_buf_sample_number)
            self._begin_buf_sample_number += block.shape[0]

        # If we are writing everything and there is still data in the
        # buffer, write the rest of it and clear the buffer.
        if write_all and self._buf.shape[0] > 0:
            self._transmit_block(self._buf,
                                 self._begin_buf_sample_number)
            self._begin_buf_sample_number += self._buf.shape[0]
            self._buf = np.empty(shape=(0, number_channels),
                             dtype={True: 'float32',
                             False: 'float64'}[self.convert_to_single])

    def _transmit_block(self, block, n):
        """ Transmit one block of data to the client.

        Parameters
        ----------
        block : array_like
            The data to transmit. The columns must correspond to the
            channels and the rows to successive samples on each
            channel.
        n : int
            The index of the starting sample in the block from the
            beginning of acquisition (zero indexed).

        """
        # The data to write looks like
        #
        # b'Data:ENDIANNESS:xN:xNUM: YYYYYYYYYYYYYYYYYYYYY'
        #
        # Where ENDIANNESS is sys.byteorder, which is 'little' or 'big',
        # xN is n written in hexidecimal, xNUM is the number of rows in
        # block written in hexidecimal, and YYYYYYYYYYYYYYYYYYY is all
        # the bytes in block converted to hexidecimal.
        self.server.push_line(b'Data:' + sys.byteorder.encode() + b':' \
            + hex(n).encode() + b':' + hex(block.shape[0]).encode() \
            + b': ' + binascii.hexlify(block.flatten().tostring()))


class DaqAsynchat(asynchat.async_chat):
    """ Base class for the DaqClient and DaqServerHandler.

    Meant to be inherited rather than used directly. Adds a
    ``threading.Lock`` to writing output to the socket compared to
    ``asynchat.async_chat``. Received data is broken up into lines and
    processed one at a time by ``process_input_line``. Data is sent line
    by line as well with ``push_line`` and ``push_lines``.

    Parameters
    ----------
    sock : socket.socket or None
        The ``socket.socket`` to use. ``None`` means use the default
        one obtained from calling
        ``self.create_socket(socket.AF_INET, socket.SOCK_STREAM)``.
    debug_communications : bool
        Whether to debug communications over the socket by printing
        sent and received text to standard output. The first 60
        characters are printed along with a prefix of ``'[sent] '``
        and ``'[received] '`` for sent and received text
        respectively.

    See Also
    --------
    DaqClient
    DaqServerHandler

    """
    def __init__(self, sock=None, debug_communications=False):
        self._debug_communications = debug_communications
        if sock is None:
            asynchat.async_chat.__init__(self)
            self.create_socket(socket.AF_INET, socket.SOCK_STREAM)
        else:
            asynchat.async_chat.__init__(self, sock=sock)
        # Going to be handling termination checking manually in
        # collect_incoming_data.
        self.set_terminator(None)

        # Going to be using \n as the line terminator in all cases.
        self._term = b'\n'

        # Make an empty input buffer.
        self._input_buffer = b''

        # Need a lock for sending data to avoid threading problems.
        self._sending = threading.Lock()

        # Make huge buffers.
        self.ac_in_buffer_size = 10*1024*1024
        self.ac_out_buffer_size = 10*1024*1024

    def initiate_send(self):
        """ Wrapper with locking for async_chat method.

        Wraps ``asynchat.async_chat.initiate_send`` in a lock.
        
        See Also
        --------
        asynchat.async_chat.initiate_send

        """
        # Have to override the parent version and wrap in a lock for
        # thread safety reasons. Found this out on the internet.
        with self._sending:
            asynchat.async_chat.initiate_send(self)

    def collect_incoming_data(self, data):
        """ Collects the incoming data, buffers it, and processes it.

        Collects the incoming `data`, combines it with that already
        collected, splits it into lines based on the newline terminator
        (``'\\n'``), and sends each line off to ``process_input_line``
        for processing. If the ``debug_communications`` option was set,
        the first 60 characters are written to standard out prefixed
        with ``'[received] '``.

        Parameters
        ----------
        data : bytes
            The data just read from the socket to be collected.

        See Also
        --------
        process_input_line

        """
        # Join the whole buffer together and then split at the
        # terminators.
        self._input_buffer += data
        parts = self._input_buffer.split(self._term)

        # The last part is will either be the piece with no newline at
        # the end, or empty. Either way, the buffer should be set to it
        # and it should be removed from parts.
        self._input_buffer = parts.pop()

        # Dispatch input lines till there are none left.
        for line in parts:
            if self._debug_communications:
                print('[received] ' + line[:60].decode())
            self.process_input_line(line)

    def push_line(self, line):
        """ Writes a line of text to the socket.

        Writes the `line` of text to the socket. If the
        ``debug_communications`` option was set, the first 60 characters
        are written to standard out prefixed with ``'[sent] '``.

        Parameters
        ----------
        line : bytes
            The line of text to write to the socket without the
            terminating newline (``'\\n'`` is added onto the end
            automatically).

        See Also
        --------
        push_lines

        """
        if self._debug_communications:
            print('[sent] ' + line[:60].decode())
        self.push(line + self._term)

    def push_lines(self, lines):
        """ Write a bunch of lines of text to the socket one at a time.

        Sends each line through ``push_line``.

        Parameters
        ----------
        lines : iterable of bytes
            Each line of text to write to the socket.

        See Also
        --------
        push_line

        """
        for line in lines:
            self.push_line(line)

    def process_input_line(self, line):
        """ Processes an incoming line on the socket (does nothing now).

        Every received line of text on the socket is processed through
        here one at a time. Currently does nothing, so inheritors need
        to override it for processing.

        Parameters
        ----------
        line : bytes
            The line of recieved text to process. Does not include
            the newline, ``'\\n'``.

        """
        pass


class DaqServer(asyncore.dispatcher):
    """ Dispatcher for the DAQ server ``DaqServerHandler``.

    When a client connects, a ``DaqServerHandler`` is spawned and set to
    handle the request. Only one connected client is allowed at a time.

    Parameters
    ----------
    host : str or None
        The address/interface to listen on. ``''`` and ``None`` mean
        listen to all interfaces. ``'localhost'`` means listen only
        on the local loopback device.
    port : int
        The port to accept connections from. Should be greater than
        1024 to avoid clashing with reserved ports.
    debug_communications : bool
        Whether to debug communications over the socket by printing
        sent and received text to standard output. The first 60
        characters are printed along with a prefix of ``'[sent] '``
        and ``'[received] '`` for sent and received text
        respectively. It is passed on to every spawned
        ``DaqServerHandler``.

    See Also
    --------
    DaqServerHandler
    DaqClient
    DaqInterface

    """
    def __init__(self, host=default_host, port=default_port,
                 debug_communications=False):
        self._debug_communications = debug_communications
        asyncore.dispatcher.__init__(self)
        self.create_socket(socket.AF_INET, socket.SOCK_STREAM)
        self.set_reuse_addr()
        self.bind((host, port))
        self.address = self.socket.getsockname()
        self.srv = None
        self.listen(1)

    def handle_accept(self):
        """ Spawns and hands off to server when client connects."""
        client_info = self.accept()
        self.srv = DaqServerHandler(sock=client_info[0], \
            debug_communications=self._debug_communications)

    def handle_close(self):
        """ Close the daqserverhandler if it is running."""
        if self.srv is not None:
            self.srv.exit_command()
            self.srv = None

    def close(self):
        """ Close the dispatcher."""
        if self.srv is not None:
            self.srv.exit_command()
            self.srv = None
        asyncore.dispatcher.close(self)



class DaqServerHandler(DaqAsynchat):
    """ DAQ server handler that interacts with the client.

    Handles the connections to DAQ clients obtained by
    ``DaqServer``. Listens to the socket for commands and sends back
    responses and acquired data.

    Parameters
    ----------
    sock : socket.socket
        Connected socket to communicate with the client on.
    debug_communications : bool
        Whether to debug communications over the socket by printing
        sent and received text to standard output. The first 60
        characters are printed along with a prefix of ``'[sent] '``
        and ``'[received] '`` for sent and received text
        respectively.

    Raises
    ------
    ImportError
        Not able to import the package PyDAQmx.

    Attributes
    ----------
    acquiring : bool

    See Also
    --------
    DaqServer
    DaqClient
    DaqInterface

    """
    def __init__(self, sock, debug_communications=False):
        # If we don't have PyDAQmx, then nothing can be done other than
        # return error.
        if not have_PyDAQmx:
            raise ImportError('Couldn''t import PyDAQmx.')

        # Scan the available DAQ's.
        self._daq_list = None
        self._scan_daqs()

        # Pass on the arguments to the parent class.
        DaqAsynchat.__init__(self, sock=sock,
                             debug_communications=debug_communications)

        # We don't have a DAQ selected yet, so we start with a blank
        # config.
        self._daq_config = (b'No DAQ config.', b'', 0.0, 0,
                            b'', [])

        # Need to hold the analog input task.
        self._analog_input = None

        # Need to hold on to the acquisition thread.
        self._acquire_thread = None

    @property
    def acquiring(self):
        """ Whether the data is being acquired or not.

        bool

        """
        return (self._acquire_thread is not None
                and self._acquire_thread.is_alive())

    def process_input_line(self, line):
        """ Processes an incoming line on the socket.

        Every received line of text on the socket is processed through
        here one at a time.

        Parameters
        ----------
        line : bytes
            The line of recieved text to process. Does not include
            the newline, ``'\\n'``.

        """
        # Process the different commands.
        if line == b'Scan':
            # If acquisition is going on, the DAQs cannot be scanned and
            # so an error message must be returned. Otherwise, they can
            # be scanned and success reported.
            if self.acquiring:
                self.push_line(b'Error:Scan: acquiring.')
            else:
                self._scan_daqs()
                self.push_line(b'Scan successful.')
        elif line == b'Start':
            # Start aquisition. If acquisition is already happening or
            # the DAQ is not setup yet, return an error message.
            if self.acquiring:
                self.push_line(b'Error:Start: acquiring.')
            else:
                if self._daq_config[0] != b'':
                    self.push_line(b'Error:Start: setup.')
                else:
                    self._start_daq()
        elif line == b'Stop':
            # If acquiring, stop acquisition (daq and acquire thread)
            # and report success. Otherwise report an error.
            if self.acquiring:
                self._stop_daq()
                self._acquire_thread.join()
                self.push_line(b'Stop successful.')
            else:
                self.push_line(b'Error:Stop: not acquiring.')
        elif line == b'Exit':
            self.exit_command()
        elif line.startswith(b'Setup'):
            # Cannot setup the DAQ while acquiring so that is an error.
            if self.acquiring:
                self.push_line(b'Error:Setup: acquiring.')
            else:
                # Parse the line. If it is valid (leading string is
                # blank), acknowledge it and set the current conf to
                # it. If not, send the leading string (its the error
                # message).
                cnf = self._parse_daq_task_command(line)
                if cnf[0] != b'':
                    self.push_line(cnf[0])
                else:
                    self._daq_config = cnf
                    self.push_line(b'Setup successful.')
        else:
            self.push_line(b'Invalid command.')

    def exit_command(self):
        """ Stops DAQ, if acquiring, and closes the connection.

        The text ``b'Exiting'`` is written to the socket before it is
        closed.

        """
        if self.acquiring:
            self._stop_daq()
            self._acquire_thread.join()
        self.push_line(b'Exiting.')
        self.close_when_done()

    def _scan_daqs(self):
        """ Scans for attached NI DAQ's."""
        self._daq_list = pynidaqutils.list_daqs()

    def _stop_daq(self):
        """ Stop acquisition."""
        # If acquiring, the acquisition thread needs to be told to stop.
        if self._acquire_thread is not None:
            self._acquire_thread.abort()

    def _start_daq(self):
        """ Start acquisition."""
        # Clear the analog input task.
        self._analog_input = None

        # Supposedly a scan is necessary to reset a jammed DAQ.
        self._scan_daqs()

        # Make a termination lookup table and grab the config fields.
        terminations = {b'Diff': PyDAQmx.DAQmx_Val_Diff,
                        b'RSE': PyDAQmx.DAQmx_Val_RSE,
                        b'NRSE': PyDAQmx.DAQmx_Val_NRSE}
        device = self._daq_config[1]
        freq = self._daq_config[2]
        count = self._daq_config[3]
        output_type = self._daq_config[4]
        channels = self._daq_config[5]

        # Reset the device.
        PyDAQmx.DAQmxResetDevice(device)

        # Make the analog input task and add all the channels to it with
        # the proper settings.
        self._analog_input = PyDAQmx.Task()
        for ch in channels:
            self._analog_input.CreateAIVoltageChan( \
                device + b'/ai' + str(ch['channel']).encode(), \
                None, terminations[ch['termination']],
                -ch['voltage'], ch['voltage'],
                PyDAQmx.DAQmx_Val_Volts, None)

        # Set the sample frequency and the number of samples to take,
        # which could be infinite.
        if count > 0:
            self._analog_input.CfgSampClkTiming(None, freq, \
                PyDAQmx.DAQmx_Val_Rising, \
                PyDAQmx.DAQmx_Val_FiniteSamps, count)
        else:
            self._analog_input.CfgSampClkTiming(None, freq, \
                PyDAQmx.DAQmx_Val_Rising, \
                PyDAQmx.DAQmx_Val_ContSamps, 1000000)

        # Increase the size of the buffer to hold 10 seconds worth of
        # data (just to be on the safe side) or 10000 samples per
        # channel, whichever is greater.
        self._analog_input.CfgInputBuffer(max(10000, int(
                                          math.ceil(10.0*freq))))

        # We ideally want to write about 1000 samples to the socket at a
        # time, but wait no more than 50 ms before cycles.
        thread_interval = 0.1

        # Make the acquiring thread with the calculated interval.
        self._acquire_thread = ReadAnalogInputThread( \
            self._analog_input, self, device, thread_interval,
                 len(channels), freq,
                 int(math.ceil(thread_interval*freq)),
                 (output_type == b'single'))

        # Start the thread. It will start the acquisition.
        self._acquire_thread.start()

    def _parse_daq_task_command(self, line):
        """ Parses and extracts a DAQ setup command.

        A DAQ task command looks like

        ``b'Setup: DEV FREQUENCY COUNT TYPE CH1 CH2 CH3\\n'``

        where DEV is the Device name, FREQUENCY is the sample frequency,
        COUNT is the number of samples to take (or ``b'-1'`` to take
        continuously), TYPE is the floating point type to send  data in
        (``b'single'`` or ``b'double'``), and CH1 ... are the channel
        specifications, which must look like ``b'N:V:TERM'`` where N is
        the channel number, V is the expected voltage range (+/-), and
        TERM is the input termination and must be either ``b'Diff'``
        (differential), ``b'RSE'`` (Reference Single Ended), or
        ``b'NRSE'`` (Non-Referenced Single Ended).

        Invalid inputs, channels outside of the device's range,
        duplicate channels (including mixed differential and single
        ended), acquisition frequencies out of the range the device can
        do, device availability, etc. are all checked.

        Parameters
        ----------
        line : bytes
            DAQ setup command.

        Returns
        -------
        error_message : bytes
            If the setup command in `line` was valid, it is ``b''``.
            Otherwise, it is the error message.
        device : bytes
            The device label for the NI DAQ such as ``b'Dev1'``.
        frequency : float
            The acquisition frequency the DAQ will be run at (sample
            frequency will be this times the number of channels being
            acquired from).
        count : int
            Number of samples to take from each channel. -1 denotes
            acquiring forever (infinity).
        tp : {b'single', b'double'}
            The floating point precision that the acquired data will be
            sent to the client in. ``b'single'`` and ``b'double'``
            denote ``numpy.float32`` and ``numpy.float64`` respectively.
        channels : list of dicts
            List of the channel configurations in the order specified by
            `line`. Each ``dict`` has the channel number (``int``) in
            field ``'channel'``, the maximum voltage magnitude to
            measure (``float``) in field ``'voltage'``, and the
            termination (``bytes``) in field ``'termination'``.

        """
        # Each part is space separated.
        parts = line.split()

        # If we don't have at least 5 parts and the first one is not
        # 'Setup:', then it is invalid.
        if len(parts) < 6:
            return (b'Invalid: missing parts.', b'', 0.0, 0, b'', [])
        if parts[0] != b'Setup:':
            return (b"Invalid: didn't start with 'Setup:'.", b'',
                    0.0, 0, b'', [])

        # Get the device, read the frequency (handle conversion errors),
        # read the count (handle conversion errors), and the
        # type. Errors mean that it was invalid.
        device = parts[1]
        try:
            frequency = float(parts[2])
        except:
            return (b"Invalid: invalid sample frequency: '"
                    + parts[2] + "'.", b'', 0.0, 0, b'', [])
        try:
            count = int(parts[3])
        except:
            return (b"Invalid: invalid count: '"
                    + parts[3] + "'.", b'', 0.0, 0, b'', [])
        tp = parts[4]
        if tp not in (b'single', b'double'):
            return (b"Invalid: invalid type: '"
                    + parts[4] + "'.", b'', 0.0, 0, b'', [])

        # Check that count is positive or -1.
        if count < 1 and count != -1:
            return (b'Invalid: count ' + str(count).encode()
                    + b' must be positive or -1.', b'', 0.0, 0, b'', [])

        # Parse the provided channels and make a list of them in order
        # one by one. Exceptions are used to say what is wrong.
        #
        # A channel specification looks like 'N:V:TERM' where N is the
        # channel number, V is the expected voltage range (+/-), and
        # TERM is the input termination and must be either 'Diff'
        # (differential), 'RSE' (Reference Single Ended), or 'NRSE'
        # (Non-Referenced Single Ended).
        channels = []
        for channel in parts[5:]:
            # Colons are the separators.
            channel_split = channel.split(b':')

            # If we don't have three parts, it is definitely invalid.
            if len(channel_split) != 3:
                return (b"Invalid: invalid channel: '"
                        + channel + "'.", b'', 0.0, 0, b'', [])

            # Extract the parameters for this channel. An exception
            # means it is invalid.
            ch = dict()
            try:
                ch = {'channel': int(channel_split[0]),
                      'voltage': float(channel_split[1]),
                      'termination': channel_split[2]}
            except:
                return (b"Invalid: invalid channel: '"
                        + channel + "'.", b'', 0.0, 0, b'', [])

            # Check that the channel is non-negative, the voltage is
            # positive, and the termination is one of the allowed values.
            if ch['channel'] < 0 or ch['voltage'] <= 0 \
                    or ch['termination'] not in (b'Diff', b'RSE', \
                    b'NRSE'):
                return (b"Invalid: invalid channel: '"
                        + channel + "'.", b'', 0.0, 0, b'',[])
            channels.append(ch)

        # Check to make sure the given device name is one of them (it is
        # an invalid if not).
        if device.decode() not in self._daq_list:
            return (b"Invalid: device '" + device + b"' not among "
                    b'available devices: '
                    + str(list(self._daq_list.keys())).encode() + b'.',
                    b'', 0.0, 0, b'', [])

        # Grab the hardware info for this DAQ.
        daq_info = pynidaqutils.hw_info[self._daq_list[ \
            device.decode()]['type']]

        # Check to make sure the sample frequency is positive and not
        # too high.
        if len(channels) * frequency \
                > daq_info['max_sample_frequency'] or frequency <= 0.0:
            return (b'Invalid: frequency ' + str(frequency).encode()
                    + b' not in the range (0.0, '
                    + str(daq_info['max_sample_frequency']).encode()
                    + b'].', b'', 0.0, 0, b'', [])

        # Make a list of all the channel numbers on the device that will
        # be used. For single ended channels, it is just the specified
        # channel. For double ended channels, it is the specified
        # channel plus N//2 where N is the total number of analog input
        # channels on the device.
        channel_nums = []
        for ch in channels:
            channel_nums.append(ch['channel'])
            if ch['termination'] == b'Diff':
                channel_nums.append(ch['channel'] + daq_info['ai']//2)

        # Check to see if there are any duplicate channel
        # numbers. Turning it into a set removes extra copies of
        # channels, which will result in a set of a different length.
        if len(channel_nums) != len(set(channel_nums)):
            return (b'Invalid: duplicate channels.', b'', 0.0, 0,
                    b'', [])

        # Check to make sure that none of the channels are negative.
        if min(channel_nums) < 0:
            return (b'Invalid: negative channel '
                    + str(min(channel_nums)).encode() + b'.', b'',
                    0.0, 0, b'', [])

        # Check to make sure that none of the channels are above the
        # maximum range (total available minus 1).
        if max(channel_nums) >= daq_info['ai']:
            return (b'Invalid: channel '
                    + str(max(channel_nums)).encode()
                    + b' when the highest available one is '
                    + str(daq_info['ai'] - 1).encode() + b'.', b'',
                    0.0, 0, b'', [])

        # Check to make sure that none of the maximum expected voltages
        # are greater than what the DAQ can handle.
        max_voltage = max([ch['voltage'] for ch in channels])
        if max(daq_info['voltages']) < max_voltage:
            return (b"Invalid: DAQ can't handle voltage "
                    + str(max_voltage).encode() + b' > '
                    + str(max(daq_info['voltages'])).encode() + b'.',
                    b'', 0.0, 0, b'', [])

        # Its valid, so return everything.
        return (b'', device, frequency, count, tp,
                channels)


class DaqClient(DaqAsynchat):
    """ DAQ acquisition client.

    This is the DAQ client that connects to the server at the specified
    IP address (`host`) and `port`. When data is being acquired, it is
    read from the server in blocks that are obtained by calling
    ``get_new_data``.

    Parameters
    ----------
    host : str
        The host name of the machine the DAQ server is running on. Can
        be a valid IP address, a DNS resolvable address, or
        ``'localhost'`` specifying this machine.
    port : int
        The port to accept connections from. Should be greater than
        1024 to avoid clashing with reserved ports.

    Attributes
    ----------
    is_acquiring : bool
    start_time : list of floats

    See Also
    --------
    DaqServer
    DaqServerHandler
    DaqInterface

    """
    def __init__(self, host='localhost', port=default_port,
                 debug_communications=False):
        DaqAsynchat.__init__(self, \
            debug_communications=debug_communications)
        self.set_reuse_addr()
        self.connect((host, port))

        # Make events for when various commands have finished, various
        # errors were received, or when data has come in.
        self.scan_finished_event = threading.Event()
        self.scan_error_event = threading.Event()
        self.stop_finished_event = threading.Event()
        self.stop_error_event = threading.Event()
        self.exit_finished_event = threading.Event()
        self.setup_finished_event = threading.Event()
        self.setup_error_invalid_event = threading.Event()
        self.setup_error_acquiring_event = threading.Event()
        self.acquisition_finished_event = threading.Event()
        self.aborting_acquisition_event = threading.Event()
        self.error_during_acquisition_event = threading.Event()
        self.invalid_command_event = threading.Event()
        self.acquisition_started_event = threading.Event()
        self.acquisition_error_setup_event = threading.Event()
        self.acquisition_error_acquiring_event = threading.Event()
        self.got_data_event = threading.Event()
        self.acquiring_event = threading.Event()

        # We don't have a DAQ selected yet, so we start with a blank
        # config.
        self._daq_config = (b'No DAQ config.', b'', 0.0, 0,
                            b'', [])

        # We need an array (will start out None) to hold the data, a
        # list to log what data was gotten, as well as a lock for it.
        self._data = None
        self._data_log = []
        self._data_lock = threading.RLock()

        # Need an array to hold the start time.
        self._start_time = None

    @property
    def is_acquiring(self):
        """ Whether data is being acquired or not.

        bool

        """
        return self.acquiring_event.is_set()

    @property
    def start_time(self):
        """ The time on the server that the acquisition started.

        list of floats

        6-element ``list`` of ``float`` that specify the time (UTC) on
        the server that the acquisition started in year, month, day,
        hour, minute, seconds order. All must have no fractional part
        except the seconds field.

        .. warning::

           The clock on the server could be very different than the
           clock on the client if they are different machines, so
           the time may not make much sense when looking at the clock
           on the client machine.

        """
        if self._start_time is None:
            return None
        else:
            return copy.deepcopy(self._start_time)

    def get_new_data(self):
        """ Get all acquired data since the last call.

        The data is acquired in blocks from the server, so it is
        returned in blocks.

        Returns
        -------
        data : list of numpy.ndarray or None
            ``list`` of the acquired data blocks. Each block is a
            ``numpy.ndarray`` of dtype float32 or float64. The
            columns are the channels in the order specified in the
            setup command and the rows are successive samples. ``None``
            if no blocks have been received since the last time this
            function was called.
        lg : list of tuples
            ``list`` of which samples are in each block, which is a
            ``tuple`` having the starting and ending sample number
            (zero indexed).

        """
        with self._data_lock:
            # Grab the data and log into temporary copies, clear the
            # originals, and return the temporaries.
            data = self._data
            lg = self._data_log
            self._data = None
            self._data_log = []
            return (data, lg)
        # Some sort of exception occurred, so return nothing.
        return (None, [])

    def process_input_line(self, line):
        """ Processes an incoming line on the socket.

        Every received line of text on the socket is processed through
        here one at a time.

        Parameters
        ----------
        line : bytes
            The line of recieved text to process. Does not include
            the newline, ``'\\n'``.

        """
        if line == b'Scan successful.':
            self.scan_error_event.clear()
            self.scan_finished_event.set()
        elif line == b'Stop successful.':
            self.stop_error_event.clear()
            self.stop_finished_event.set()
        elif line == b'Exiting.':
            self.exit_finished_event.set()
            self.close_when_done()
        elif line == b'Setup successful.':
            self.setup_error_invalid_event.clear()
            self.setup_error_acquiring_event.clear()
            self.setup_finished_event.set()
        elif line == b'Finished acquisition.':
            self.acquisition_finished_event.set()
            self.acquiring_event.clear()
        elif line == b'Aborting acquisition.':
            self.aborting_acquisition_event.set()
        elif line == b'Error: acquisition.':
            self.error_during_acquisition_event.set()
        elif line == b'Invalid command.':
            self.invalid_command_event.set()
        elif line == b'Error:Start: setup.':
            self.acquisition_error_setup_event.set()
            self.acquisition_error_acquiring_event.clear()
            self.acquisition_started_event.set()
        elif line.startswith(b'Error:') \
                and line.endswith(b'acquiring.'):
            if b':Scan:' in line:
                self.scan_error_event.set()
                self.scan_finished_event.set()
            elif b':Stop:' in line:
                self.stop_error_event.set()
                self.stop_finished_event.set()
            elif b':Setup:' in line:
                self.setup_error_invalid_event.clear()
                self.setup_error_acquiring_event.set()
                self.setup_finished_event.set()
            elif b':Start:' in line:
                self.acquisition_error_setup_event.clear()
                self.acquisition_error_acquiring_event.set()
                self.acquisition_started_event.set()
        elif line.startswith(b'Invalid:'):
            # There was an error with a setup command.
            self.setup_error_invalid_event.set()
            self.setup_finished_event.set()
        elif line.startswith(b'Started at '):
            # Split it into parts based on the spaces, convert all the
            # numeric time ones to floats, and stuff them in
            # _start_time.
            try:
                parts = line.split(b' ')
                self._start_time = [float(s) for s in parts[2:]]
            except:
                self._start_time = None

            self.error_during_acquisition_event.clear()
            self.aborting_acquisition_event.clear()
            self.acquisition_error_setup_event.clear()
            self.acquisition_error_acquiring_event.clear()
            self.acquisition_started_event.set()
            self.acquiring_event.set()
        elif line.startswith(b'Data:'):
            # The data to write looks like
            #
            # b'Data:ENDIANNESS:xN:xNUM: YYYYYYYYYYYYYYYYYYYYY'
            #
            # Where ENDIANNESS is sys.byteorder, which is 'little' or
            # 'big', xN is n (starting sample number) written in
            # hexidecimal, xNUM is the number of rows in block written
            # in hexidecimal, and YYYYYYYYYYYYYYYYYYY is all the bytes
            # in block converted to hexidecimal.
            with self._data_lock:
                # Grab the header with all the config information. A
                # space character separates the header from the data.
                index = line.find(b' ')

                # Split the header into parts and return if the split
                # doesn't give 5 elements back like it should
                parts = line[:index].split(b':')
                if len(parts) != 5:
                    return

                # Grab the endianness, the index of the starting sample
                # of the block, and the number of samples read.
                endianness = parts[1]
                starting_sample_number = int(parts[2], 16)
                number_samples = int(parts[3], 16)

                # Make an array of the appropriate type (single or
                # double precision) of the right shape
                # (number_samples, number_channels) from the
                # unhexilified version of the rest of the line.
                block = np.ndarray(shape=(number_samples \
                    * len(self._daq_config[-1]),), \
                    dtype={b'single': 'float32', \
                    b'double': 'float64'}[self._daq_config[-2]], \
                    buffer=binascii.unhexlify(line[(index+1):]))
                block = block.reshape((number_samples,
                                      len(self._daq_config[-1])))

                # If the endianness doesn't match this machine, swap the
                # bytes.
                if sys.byteorder != endianness.decode():
                    block = block.byteswap()

                # If we are on the first sample, then we need to clear
                # the received data and the log.
                if starting_sample_number == 0:
                    self._data = None
                    self._data_log = []

                # If we don't have any received data yet, then block is
                # all the data we have. Otherwise, block is just
                # appended to the already received data.
                if self._data is None or starting_sample_number == 0:
                    self._data = block
                else:
                    self._data = np.vstack((self._data, block))

                # Add to the log the range of samples that were gotten
                # (not the beginning and then a count like we were given
                # from the server).
                self._data_log.append((starting_sample_number,
                                      starting_sample_number
                                      + number_samples))

                # Successfully got data, so the event needs to be set
                # and we are done processing this line.
                self.got_data_event.set()
                return

            # There was something wrong with the data we received, so we
            # need to clear the got_data_event.
            self.got_data_event.clear()


    def scan_daqs(self, timeout=None):
        """ Scans for DAQs on the server.

        Parameters
        ----------
        timeout : float or None
            The timeout to use when waiting for the response from the
            server that the scan was successful or in error after the
            scan_daq command is sent to the server. ``None``
            denotes an infinite timeout.

        Returns
        -------
        success : bool
            Whether the scan was successful or not.

        """
        self.scan_finished_event.clear()
        self.push_line(b'Scan')
        self.scan_finished_event.wait(timeout)
        return (not self.scan_error_event.is_set())

    def start_daq(self, timeout=None):
        """ Start the DAQ.

        Parameters
        ----------
        timeout : float or None
            The timeout to use when waiting for the acquisition to start
            after the start command is sent to the server. ``None``
            denotes an infinite timeout.

        Returns
        -------
        success : bool
            Whether acquisition was started or not.

        See Also
        --------
        stop_daq

        """
        # If we don't have a valid setup yet, then there is no point
        # even trying as we know it is an error.
        if self._daq_config[0] != b'':
            self.error_during_acquisition_event.clear()
            self.aborting_acquisition_event.clear()
            self.acquisition_error_setup_event.set()
            self.acquisition_error_acquiring_event.clear()
            self.acquisition_started_event.clear()
            return False

        # The command can be done, but first, the existing data buffer
        # and the log need to be cleared.
        with self._data_lock:
            self._data = None
            self._data_log = []

        # Do the command.
        self.got_data_event.clear()
        self.error_during_acquisition_event.clear()
        self.aborting_acquisition_event.clear()
        self.acquisition_started_event.clear()
        self.push_line(b'Start')
        self.acquisition_started_event.wait(timeout)
        return (not self.acquisition_error_setup_event.is_set()
                or not self.acquisition_error_acquiring_event.is_set())

    def stop_daq(self, timeout=None):
        """ Stops the DAQ.

        Parameters
        ----------
        timeout : float or None
            The timeout to use when waiting for the acquisition to stop
            after the stop command is sent to the server. ``None``
            denotes an infinite timeout.

        Returns
        -------
        success : bool
            Whether acquisition was stopped or not.

        See Also
        --------
        start_daq

        """
        self.stop_finished_event.clear()
        self.push_line(b'Stop')
        self.stop_finished_event.wait(timeout)
        return (not self.stop_error_event.is_set())

    def exit(self, timeout=None):
        """ Close and exit this client and the connection to the server.

        Parameters
        ----------
        timeout : float or None
            The timeout to use when waiting for the connection to close
            after the exit command is sent to the server. ``None``
            denotes an infinite timeout.

        """
        self.exit_finished_event.clear()
        self.push_line(b'Exit')
        self.exit_finished_event.wait(timeout)
        self.close_when_done()

    def setup_daq(self, device, frequency, count, tp, channels,
                  timeout=None):
        """ Sends a DAQ setup to the server.

        Sends a DAQ setup to the server.

        Parameters
        ----------
        device : bytes
            The device label for the NI DAQ such as ``b'Dev1'``.
        frequency : float
            The acquisition frequency the DAQ will be run at (sample
            frequency will be this times the number of channels being
            acquired from).
        count : int
            Number of samples to take from each channel. -1 denotes
            acquiring forever (infinity).
        tp : {b'single', b'double'}
            The floating point precision that the acquired data will be
            sent to the read in. ``b'single'`` and ``b'double'``
            denote ``numpy.float32`` and ``numpy.float64`` respectively.
        channels : iterable of dicts
            Iterable of the channel configurations. Each ``dict`` has
            the channel number (``int``) in field ``'channel'``, the
            maximum voltage magnitude to measure (``float``) in field
            ``'voltage'``, and the input termination (``bytes``) in
            field ``'termination'``. The input termination and must be
            either ``b'Diff'`` (differential), ``b'RSE'`` (Reference
            Single Ended), or ``b'NRSE'`` (Non-Referenced Single Ended).
        timeout : float or None
            The timeout to use when waiting for the server to respond
            after the DAQ setup command is sent to the server. ``None``
            denotes an infinite timeout.

        Returns
        -------
        success : bool
            Whether setup was a success (no error) or not.
        command : bytes
            The text of the setup command that was sent to the server.
        daq_conf : list
            The DAQ configuration that was specified and matches what
            ``DaqServerHandler`` would construct. It is
            ``[b'', device, frequency, count, tp, channels]``.

        """
        # Make the daq config structure that all of this implies.
        daq_conf = [b'', device, frequency, count, tp, channels]

        # Construct the channels strings.
        channel_strings = []
        for ch in channels:
            channel_strings.append((':'.join([str(ch['channel']), \
                str(ch['voltage']), \
                ch['termination'].decode()])).encode())

        # Make the command to send. This consists of adding the device,
        # frequency (as a string), count (as a string), the type, and
        # all the channel strings together with a space between them
        # all.
        parts = [b'Setup:']
        parts.extend(daq_conf[1:-1])
        parts.extend(channel_strings)
        parts[2] = str(parts[2]).encode()
        parts[3] = str(parts[3]).encode()
        command = b' '.join(parts)

        # Reset the event, send the command, and wait till we get a
        # response.
        self.setup_finished_event.clear()
        self.push_line(command)
        self.setup_finished_event.wait(timeout)

        # Return whether the command was successful or not. If it was
        # successful, self._daq_config can be set.
        if self.setup_error_invalid_event.is_set() \
                or self.setup_error_acquiring_event.is_set():
            return (False, copy.deepcopy(command),
                    copy.deepcopy(daq_conf))
        else:
            self._daq_config = daq_conf
            return (True, copy.deepcopy(command),
                    copy.deepcopy(daq_conf))


class DaqInterface(object):
    """ Interface to a DAQ server and client on the same machine.

    Provides an interface for starting and stopping a DAQ server and a
    ``DaqClient`` connecting to it on this machine.

    Attributes
    ----------
    client : DaqClient or None
        The running ``DaqClient`` or ``None``.
    daq_program_path : str
        The path to this module, which is needed to start a server.
    is_acquiring : bool
    is_client_running : bool
    is_server_running : bool
    versions : dict
        The versions of pynidaqutils and PyDAQmx.

    See Also
    --------
    DaqClient
    DaqServer
    DaqServerHandler

    """
    def __init__(self):
        #: The path to this module, which is needed to start a server.
        #:
        #: str
        #:
        self.daq_program_path = __file__

        # We need to grab all the version information from daq_program,
        # so we need to run it with the --versions option.
        output = subprocess.check_output([self.daq_program_path,
                                         '--version'],
                                         universal_newlines=True)

        # We need to make a dict of the versions of each part. Each line
        # of output is the a version of the form 'package: version'.

        #: The versions of pynidaqutils and PyDAQmx.
        #:
        #: dict
        #:
        #: The versions of this module and the PyDAQmx module are stored
        #: in the keys ``'pynidaqutils'`` and ``'PyDAQmx'``
        #: respectively.
        #:
        self.versions = dict()
        for line in output.splitlines():
            index = line.find(':')
            if index != -1:
                self.versions[line[:index]] = line[(index + 2):]

        # We need a handle for the daq server.
        self._server = None

        # We need handles for the client and a thread to run
        # asyncore.loop so that the client will operate.

        #: The running ``DaqClient`` or ``None``.
        #:
        #: DaqClient or None
        #:
        #: See Also
        #: --------
        #: DaqClient
        #:
        self.client = None
        self._socket_thread = None

    def __del__(self):
        self.stop_server()

    @property
    def is_server_running(self):
        """ Whether the server is running or not.

        bool

        """
        return (self._server is not None)

    @property
    def is_client_running(self):
        """ Whether the client is running or not.

        bool

        """
        return (self.client is not None and self.client.connected)

    @property
    def is_acquiring(self):
        """ Whether data is being acquired or not.

        Is ``True`` if the ``client`` is started and acquiring.

        bool

        """
        return (self.client is not None and self.client.is_acquiring)

    def start_server(self):
        """ Start the server.

        Returns
        -------
        success : bool
            Whether the server was started or not (was already
            started).

        See Also
        --------
        stop_server
        start_client
        stop_client

        """
        if self._server is not None:
            return False

        self._server = subprocess.Popen([self.daq_program_path],
                                        stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        universal_newlines=True)
        return True

    def stop_server(self, timeout=None):
        """ Stop the server.

        If the client is running, it is stopped first (``stop_client``
        is called) before the server is stopped.

        Parameters
        ----------
        timeout : float or None
            The timeout to use when stopping the server and client.
            ``None`` denotes an infinite timeout.

        Returns
        -------
        could_stop : bool
            Whether the server can be stopped (was running) or not.

        See Also
        --------
        start_server
        start_client
        stop_client

        """
        if self._server is None:
            return True

        # Stop the client if it is there.
        self.stop_client(timeout=timeout)

        # Send the close command on the server's stdin.
        outputs = self._server.communicate(input='close\n',
                                           timeout=timeout)

        self._server = None
        return ('closed' in outputs[0].lower())

    def start_client(self):
        """ Start the client.

        Returns
        -------
        success : bool
            Whether the client was started or not (was already
            started or server has not been started).

        See Also
        --------
        start_server
        stop_server
        stop_client

        """
        if self._server is None or self.client is not None:
            return False
        self.client = DaqClient()
        self._socket_thread = threading.Thread( \
            target=lambda : asyncore.loop(timeout=1.0))
        self._socket_thread.start()
        return True

    def stop_client(self, timeout=None):
        """ Stop the client.

        Parameters
        ----------
        timeout : float or None
            The timeout to use when stopping the client and when
            waiting for the socket to close. ``None`` denotes an
            infinite timeout.

        Returns
        -------
        could_stop : bool
            Whether the client can be stopped (was running) or not.

        See Also
        --------
        start_server
        stop_server
        start_client

        """
        if self.client is None:
            return False
        self.client.exit(timeout=timeout)
        self._socket_thread.join(timeout)
        self.client = None
        self._socket_thread = None
        return True



# Not doing anything unless this is being run as a program.
if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description="""Starts a server
        for interacting with a National Instruments DAQ listening
        to the specified port on the indicated host. localhost means
        listening to local connections only, host means listening to
        connections from the network, and both means listen to
        both. Send 'close\\n' on stdin to close it.
        """)
    parser.add_argument('-v', '--version', action='store_true',
                        help='return the version of this program and '
                        + 'PyDAQmx')
    parser.add_argument('--host', choices=['localhost', 'all'],
                        default=default_host,
                        help='where to listen for connections '
                        + '(default is ' + default_host + ')')
    parser.add_argument('--port', type=int, default=default_port,
                        help='port to listen to connections on '
                        + '(default is ' + str(default_port) + ')')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='print comminications to stdout')
    args = parser.parse_args()

    # Display version information if asked (and then exit). We will need
    # to load the PyDAQmx module if possible and print its version.
    if args.version:
        print('pynidaqutils: ' + pynidaqutils.__version__)
        if have_PyDAQmx:
            print('PyDAQmx: ' + PyDAQmx.__version__)
        exit(0)

    # Run the daq server. If there is an import error, we need to write
    # that to stdout.
    try:
        if args.host == 'all':
            daq_server = DaqServer(host=None, port=args.port,
                                   debug_communications=args.debug)
        else:
            daq_server = DaqServer(host=args.host, port=args.port,
                                   debug_communications=args.debug)
    except ImportError:
        sys.stderr.write('Couldn''t import PyDAQmx.\n')
        sys.stderr.write('Exit 1\n')
        exit(1)

    # Make a thread to run the server, and then wait for the close
    # command to be received on stdin.

    th = threading.Thread(target=lambda : asyncore.loop(timeout=1.0))
    th.start()

    buf = ''
    try:
        while 'close' not in buf.lower():
            buf = sys.stdin.readline()
    except:
        pass
    finally:
        daq_server.close()
        th.join(2.0)
        del daq_server
        sys.stdout.write('Closed\n')
