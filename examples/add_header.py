# -*- coding: utf-8 -*-
"""
Add a new trace header to 2020 in a JavaSeis dataset
"""

# cd the example directory
import sys
sys.path.append('..')

import pieseis.io.jsfile as jsfile
from joblib import Parallel, delayed

import os
NCORE = os.cpu_count()

PDH = '/data/my_primary/PromaxDataHome/'
fnsrc = PDH + 'area/line/data.js'
fndst = PDH + 'area/line/data_copy.js'

from collections import OrderedDict as odict
from pieseis.io.properties import TraceHeader
label = "ZHU"
header = TraceHeader(value=(label, "Test new header", "INTEGER", 1, 0))
headers_add = odict()
headers_add[label] = header

jsdsrc = jsfile.JavaSeisDataset.open(fnsrc)
jsddst = jsfile.JavaSeisDataset.open(fndst, mode='w',
    properties_add=headers_add, similar_to=fnsrc)

def write1frame(jsdsrc, jsddst, iframe):
    if iframe % 10 == 0:
        print("Writing frame {} of {}".format(iframe, jsdsrc.nframes))
    idx = jsdsrc.ind2sub(iframe)
    trcs, hdrs = jsdsrc.read_frame(idx)
    fold = jsdsrc.fold(idx)
    if fold > 0:
        hdrs_new = bytearray(jsddst.header_length * fold)
        for i in range(fold):
            itrc = i + 1
            jsddst.set_header_in_frame(header, 1, itrc, hdrs, hou=hdrs_new)
        jsddst.write_frame(trcs, hdrs_new, fold, iframe)

Parallel(n_jobs=NCORE)(delayed(write1frame)(jsdsrc, jsddst, i+1)
    for i in range(jsdsrc.nframes))
