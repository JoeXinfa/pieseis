# -*- coding: utf-8 -*-
"""
Trace Compressor
"""

from math import floor

#TraceCompressor = {'nsamples': 0, 'nwindows': 0, 'windowln': 0}


def trace_compressor(nsamples, nwindows, windowln):
    return {'nsamples': nsamples, 'nwindows': nwindows, 'windowln': windowln}


def get_trace_compressor(nsamples, trace_format):
    if trace_format == "Int16":
        windowln = 100
        nwindows = floor((nsamples - 1.0) / windowln) + 1
        return trace_compressor(nsamples, nwindows, windowln)
    elif trace_format == "Float32":
        windowln = nsamples
        nwindows = 1
        return trace_compressor(nsamples, nwindows, windowln)
    else:
        raise ValueError("unsupported trace format: {}".format(trace_format))


def get_trace_length(trace_compressor, trace_format):
    cps = trace_compressor
    if trace_format == "Int16":
        if cps['nsamples'] % 2 == 0: # is even
            trclen = 4 * cps['nwindows'] + 2 * cps['nsamples']
        else:
            trclen = 4 * cps['nwindows'] + 2 * (cps['nsamples'] + 1)
    elif trace_format == "Float32":
        trclen = 4 * cps['nsamples']
    else:
        raise ValueError("unsupported trace format: {}".format(trace_format))
    return trclen
