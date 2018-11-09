# -*- coding: utf-8 -*-
"""
Test write JavaSeis dataset
"""

# cd the example directory
import sys
sys.path.append('..')

import pieseis.io.jsfile as jsfile

filename = 'C:/Users/xinfa/Documents/181109_test.js'

jsfile.JavaSeisDataset.open(filename, mode='w', axis_lengths=[128, 32, 16])
