# -*- coding: utf-8 -*-
"""
Trace Compressor
"""

import struct
from math import floor
import numpy as np


def trace_compressor(nsamples, nwindows, windowln, trace_format):
    return {'nsamples': nsamples, 'nwindows': nwindows, 'windowln': windowln,
            'trace_format': trace_format}


def get_trace_compressor(nsamples, trace_format):
    if trace_format == "Int16":
        windowln = 100
        nwindows = floor((nsamples - 1.0) / windowln) + 1
        return trace_compressor(nsamples, nwindows, windowln, trace_format)
    elif trace_format == "Float32":
        windowln = nsamples
        nwindows = 1
        return trace_compressor(nsamples, nwindows, windowln, trace_format)
    else:
        raise ValueError("unsupported trace format: {}".format(trace_format))


def get_trace_length(trace_compressor):
    cps = trace_compressor
    if cps['trace_format'] == "Int16":
        if cps['nsamples'] % 2 == 0: # is even
            trclen = 4 * cps['nwindows'] + 2 * cps['nsamples']
        else:
            trclen = 4 * cps['nwindows'] + 2 * (cps['nsamples'] + 1)
    elif cps['trace_format'] == "Float32":
        trclen = 4 * cps['nsamples']
    else:
        raise ValueError("unsupported trace format: {}".format(cps['trace_format']))
    return trclen


def unpack_trace(stream, trace_compressor, trace_offset):
    k1, k2 = 0, 0
    nsamples = trace_compressor['nsamples']
    nwindows = trace_compressor['nwindows']
    windowln = trace_compressor['windowln']
    trace = np.zeros(nsamples, dtype='float32')
    for i in range(nwindows):
        # set sample range
        k1 = k2
        k2 = min(k1 + windowln, nsamples)

        # set inverse scalar
        stream.seek(trace_offset + 4 * i)
        buffer = stream.read(4)
        # TODO big or little endian from file_properties.ByteOrder
        scalar = struct.unpack('<f', buffer)[0] # unpack returns a tuple
        if scalar > 0.0:
            scalar = 1.0 / scalar
        else:
            scalar = 0.0
        #print('i =', i, scalar)

        # scale 16-bit values to float values
        for k in range(k1, k2):
            stream.seek(trace_offset + 4 * nwindows + 2 * k1)
            buffer = stream.read(2)
            val = struct.unpack('<h', buffer)[0] # unpack int16
            val = scalar * (32769.0 + val) # 2^15 = 32768
            #print('k =', k, val)
            trace[k] = val
    return trace


def unpack_frame(stream, frame_offset, trace_compressor, fold):
    if trace_compressor['trace_format'] != 'Int16':
        raise ValueError('This method only works for Int16')
    trclen = get_trace_length(trace_compressor)
    frame = []
    for i in range(fold):
    #for i in range(1):
        trace_offset = i * trclen + frame_offset
        trace = unpack_trace(stream, trace_compressor, trace_offset)
        frame.append(trace)
    frame = np.array(frame, dtype='float32')
    return frame
