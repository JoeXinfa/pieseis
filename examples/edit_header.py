# -*- coding: utf-8 -*-
"""
Change trace header YEAR to 2020 in a JavaSeis dataset
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
        print("Reading frame {} of {}".format(iframe, jsdsrc.nframes))
    idx = jsdsrc.ind2sub(iframe)
    trcs, hdrs = jsdsrc.read_frame(idx)
    fold = jsdsrc.fold(idx)
    header = jsdsrc.properties["YEAR"]
    if fold > 0:
        for i in range(fold):
            itrc = i + 1
            jsdsrc.set_header_in_frame(header, 2020, itrc, hdrs)
        jsddst.write_frame(trcs, hdrs, fold, iframe)

Parallel(n_jobs=NCORE)(delayed(write1frame)(jsdsrc, jsddst, i+1)
    for i in range(jsdsrc.nframes))
