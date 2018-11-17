# -*- coding: utf-8 -*-
"""
Test write JavaSeis dataset
"""

# cd the example directory
import sys
sys.path.append('..')

import pieseis.io.jsfile as jsfile

#filename = 'C:/Users/xinfa/Documents/181030_test.js'
#jsd = jsfile.JavaSeisDataset.open(filename)

nsample = 101
ntrace = 101
nframe = 101
iframe = 1
fold = 101

#hdrs = jsd.read_frame_hdrs(iframe)
#print('t1 =', len(hdrs), type(hdrs))
#for hdr in hdrs: # namedtuples
#    print(hdr)
#    print(hdr.SEQNO)
#    hdr = hdr._replace(ILINE_NO=600)
#    print(hdr)

#print("ILINE_NO =", jsd.get_trace_header("ILINE_NO", 10, 1))
#print("XLINE_NO =", jsd.get_trace_header("XLINE_NO", 50, 1))
#print("BOS_Z =", jsd.get_trace_header("BOS_Z", 1, 1))
#print("VOLUME =", jsd.get_trace_header("VOLUME", 1, 1))
#print("CDP_X =", jsd.get_trace_header("CDP_X", 1, 1))
#print("CDP_XD =", jsd.get_trace_header("CDP_XD", 1, 1))
#print("CDP_Y =", jsd.get_trace_header("CDP_Y", 1, 1))
#print("CDP_YD =", jsd.get_trace_header("CDP_YD", 1, 1))
#print("TFULL_E =", jsd.get_trace_header("TFULL_E", 1, 1))
#print("TRC_TYPE =", jsd.get_trace_header("TRC_TYPE", 1, 1))
#exit()

#trcs = jsdr.read_frame_trcs(iframe)
#print('trcs1 =', trcs[0, :])
#print('trcs =', trcs.shape)

import numpy as np
trcs = np.ones((101, 101))

filename = 'C:/Users/xinfa/Documents/181115_test.js'
jsd = jsfile.JavaSeisDataset.open(filename, 'w', axis_lengths=[nsample, ntrace, nframe])
print(jsd.trace_length, jsd.header_length)

# prepare header data
headers = {}
SEQNO = np.zeros(ntrace, dtype='int32')
TRC_TYPE = np.ones(ntrace, dtype='int32')
TRACE = np.zeros(ntrace, dtype='int32')
TFULL_E = 3000.0
TLIVE_E = 3000.0
for i in range(ntrace):
    SEQNO[i] = i + 1
    TRACE[i] = i + 1
headers["SEQNO"] = SEQNO
headers["TRC_TYPE"] = TRC_TYPE
headers["TFULL_E"] = TFULL_E
headers["TLIVE_E"] = TLIVE_E
headers["TRACE"] = TRACE

for i in range(1):
    iframe = i + 1
    print('write frame =', iframe)
    headers["FRAME"] = iframe
    jsd.write_frame(trcs, headers, fold, iframe)
