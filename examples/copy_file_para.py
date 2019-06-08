# -*- coding: utf-8 -*-
"""
Copy a JavaSeis dataset using multi threads
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

jsdsrc = jsfile.JavaSeisDataset.open(fnsrc)
jsddst = jsfile.JavaSeisDataset.open(fndst, mode='w', similar_to=fnsrc)

def write1frame(jsdsrc, jsddst, iframe):
    if iframe % 10 == 0:
        print("Writing frame {} of {}".format(iframe, jsdsrc.nframes))
    idx = jsdsrc.ind2sub(iframe)
    trcs, hdrs = jsdsrc.read_frame(idx)
    fold = jsdsrc.fold(idx)
    if fold > 0:
        jsddst.write_frame(trcs, hdrs, fold, iframe)

Parallel(n_jobs=NCORE)(delayed(write1frame)(jsdsrc, jsddst, i+1)
    for i in range(jsdsrc.nframes))
