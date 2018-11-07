# -*- coding: utf-8 -*-
"""
Test read JavaSeis dataset
"""

# cd the example directory
import sys
sys.path.append('..')

import pieseis.io.jsfile as jsfile

#PATH_TO_JS_DATASET = '/home/joe/Documents/181026_test.js'
#PATH_TO_JS_DATASET = '/home/joe/Documents/181030_test.js'
PATH_TO_JS_DATASET = 'C:/Users/xinfa/Documents/181030_test.js'

#import os
#os.environ['JAVASEIS_DATA_HOME'] = '/data/primary/PromaxDataHome'
#PATH_TO_JS_DATASET = '/data/primary/PromaxDataHome/area/line/181030_test.js'

js_reader = jsfile.JSFileReader()
js_reader.open(PATH_TO_JS_DATASET)

dataset = js_reader.dataset
file_properties = dataset.file_properties

print("Dataset {0}".format(PATH_TO_JS_DATASET))
print("Samples: {0}, Traces: {1}, Frames: {2}".format(
    js_reader.nr_samples,
    js_reader.nr_traces,
    js_reader.nr_frames))

print("axis labels =", file_properties.axis_labels)
print("axis lengths =", file_properties.axis_lengths)
print("header entries =", len(dataset._trace_properties._trace_headers))
print("header bytes =", dataset.header_length)
exit()

print(file_properties.logical_origins)
print(file_properties.physical_origins)
print(file_properties.logical_deltas)
print(file_properties.physical_deltas)
print(file_properties.trace_format)
#print(dataset._trc_extents)

iframe = 1
data = js_reader.read_frame_trcs(iframe)
print(type(data))
print(data.shape)
m, n = data.shape
print(data[0, 0])
#print(data[m-1, n-1])
print(data[0,:]) # the first trace
