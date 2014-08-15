# Copyright (c) 2013, Freja Nordsiek
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

""" Module for finding attatched NI DAQ's and describing them.

Attributes
----------
hw_info : dict
    Contains the hardware information for different NI DAQ's stored by
    name (the key). Each DAQ has a ``dict`` that has fields for the
    number of single ended analog input channels ('ai'), number of
    analog output channels ('ao'), number of digital input channels
    ('di'), number of digital output channels ('do'), the resolution of
    the analog input channels in bits ('resolution'), the different
    voltage ranges (+/- the value) the analog input channels can do
    ('voltages'), and the maximum allowed sample frequency
    ('max_sample_frequency').

Version 0.1.1
"""

__version__ = "0.1.1"


import subprocess

def list_daqs():
    """ List the connected National Instruments DAQ's.

    Finds the connected National Instruments DAQ's and returns a list of
    them.

    Returns
    -------
    dict of dicts
        The attached DAQ's with their device ID's (such as ``'Dev1'``)
        as keys. Each one is a ``dict`` that has 'type' and 'hw' which
        are the DAQ type (such as ``'NI USB-6211'``) and additional
        hardware information respectively.

    Notes
    -----
    Just runs the ``lsdaq`` command and processes the output.

    """
    # Run lsdaq to get the DAQ's. If an exception occurs, then we didn't
    # find any.
    try:
        output = subprocess.check_output('lsdaq',
                                         universal_newlines=True)
    except:
        return []

    # Split the output by lines.
    lines = output.splitlines()

    # Check that the first couple lines are correct. If not, then we
    # return nothing.
    if lines[1] != 'Detecting National Instruments DAQ Devices' \
            or lines[2] != 'Found the following DAQ Devices:':
        return []

    # Contstruct the list of devices. Each device's line looks like
    #
    # 'NI USB-6211: "Dev1"    (USB0::0x3923::0x7270::0186F411::RAW)'
    #
    # If it is split about the double quotes, the device type is the
    # first part without the last two characters, the device id is the
    # middle part, and additional hardware information is the part
    # inside the parentheses.
    devices = dict()
    for i in range(3, len(lines) - 1):
        if lines[i].count('"') == 2:
            parts = lines[i].split('"')
            devices[parts[1]] = {'type': parts[0][0:-2],
                                 'hw': parts[2].strip().strip('()')}
    return devices


#: Hardware information for the different NI DAQ's.
#:
#: dict of dicts
#:
#: Contains the hardware information for different NI DAQ's stored by
#: name (the key). Each DAQ has a ``dict`` that has fields for the
#: number of single ended analog input channels ('ai'), number of analog
#: output channels ('ao'), number of digital input channels ('di'),
#: number of digital output channels ('do'), the resolution of the
#: analog input channels in bits ('resolution'), the different voltage
#: ranges (+/- the value) the analog input channels can do ('voltages'),
#: and the maximum allowed sample frequency ('max_sample_frequency').
#:
hw_info = {'NI USB-6210': {'ai': 16, 'ao': 0, 'di': 4, 'do': 4,
           'resolution': 16, 'voltages': (0.2, 1.0, 5.0, 10.0),
           'max_sample_frequency': 250000.0},
           'NI USB-6211': {'ai': 16, 'ao': 2, 'di': 4, 'do': 4,
           'resolution': 16, 'voltages': (0.2, 1.0, 5.0, 10.0),
           'max_sample_frequency': 250000.0},
           'NI USB-6212': {'ai': 16, 'ao': 2, 'di': 8, 'do': 8,
           'resolution': 16, 'voltages': (0.2, 1.0, 5.0, 10.0),
           'max_sample_frequency': 400000.0},
           'NI USB-6215': {'ai': 16, 'ao': 2, 'di': 4, 'do': 4,
           'resolution': 16, 'voltages': (0.2, 1.0, 5.0, 10.0),
           'max_sample_frequency': 250000.0},
           'NI USB-6216': {'ai': 16, 'ao': 2, 'di': 8, 'do': 8,
           'resolution': 16, 'voltages': (0.2, 1.0, 5.0, 10.0),
           'max_sample_frequency': 400000.0},
           'NI USB-6218': {'ai': 32, 'ao': 2, 'di': 8, 'do': 8,
           'resolution': 16, 'voltages': (0.2, 1.0, 5.0, 10.0),
           'max_sample_frequency': 250000.0}}
