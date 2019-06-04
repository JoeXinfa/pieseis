# -*- coding: utf-8 -*-
"""
Copy a JavaSeis dataset
"""

# cd the example directory
import sys
sys.path.append('..')

import pieseis.io.jsfile as jsfile

PDH = '/data/my_primary/PromaxDataHome/'
fnsrc = PDH + 'area/line/data.js'
fndst = PDH + 'area/line/data_copy.js'

jsdsrc = jsfile.JavaSeisDataset.open(fnsrc)
jsddst = jsfile.JavaSeisDataset.open(fndst, mode='w', similar_to=fnsrc)

for i in range(jsdsrc.nframes):
    iframe = i + 1
    if iframe % 10 == 0:
        print("Reading frame {} of {}".format(iframe, jsdsrc.nframes))
    idx = jsdsrc.ind2sub(iframe)
    trcs, hdrs = jsdsrc.read_frame(idx)
    fold = jsdsrc.fold(idx)
    if fold > 0:
        jsddst.write_frame(trcs, hdrs, fold, iframe)
