# -*- coding: utf-8 -*-
"""
Test write JavaSeis dataset
"""

# cd the example directory
import sys
sys.path.append('..')

import pieseis.io.jsfile as jsfile

#filename = 'C:/Users/xinfa/Documents/temp1.js'
#jsdr = jsfile.JavaSeisDataset.open(filename)
#trcs = jsdr.read_frame_trcs(1)
#print('trcs1 =', trcs[0, :])
#print('trcs =', trcs.shape)
#exit()


filename = 'C:/Users/xinfa/Documents/181030_test.js'
jsdr = jsfile.JavaSeisDataset.open(filename)

nframe = 101
iframe = 1
fold = 101

hdrs = jsdr.read_frame_hdrs(iframe)
#jsd.write_frame_hdrs(hdrs, fold, iframe)

trcs = jsdr.read_frame_trcs(iframe)
print('trcs1 =', trcs[0, :])
print('trcs =', trcs.shape)
#import numpy as np
#trcs = np.ones((101, 101))
exit()


filename = 'C:/Users/xinfa/Documents/temp1.js'
jsd = jsfile.JavaSeisDataset.open(filename, 'w', axis_lengths=[101,101,101])
#jsfile.write_file_properties(jsd)

for i in range(nframe):
    iframe = i + 1
    print('write frame =', iframe)
    jsd.write_frame_trcs(trcs, fold, iframe)
    jsd.write_frame_hdrs(hdrs, fold, iframe)

#filename = 'C:/Users/xinfa/Documents/181109_test.js'
#jsfile.JavaSeisDataset.open(filename, mode='w', axis_lengths=[128, 32, 16])
