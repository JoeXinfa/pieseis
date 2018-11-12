# -*- coding: utf-8 -*-
"""
Test read JavaSeis dataset
"""

# cd the example directory
import sys
sys.path.append('..')

import pieseis.io.jsfile as jsfile

#filename = '/home/joe/Documents/181026_test.js'
#filename = '/home/joe/Documents/181030_test.js'
filename = 'C:/Users/xinfa/Documents/181030_test.js'

#import os
#os.environ['JAVASEIS_DATA_HOME'] = '/data/primary/PromaxDataHome'
#filename = '/data/primary/PromaxDataHome/area/line/181030_test.js'

jsd = jsfile.JavaSeisDataset.open(filename)

file_properties = jsd.file_properties

print("Dataset {0}".format(jsd.filename))
print("axis labels =", file_properties.axis_labels)
print("axis lengths =", file_properties.axis_lengths)
print("header entries =", len(jsd._trace_properties._trace_headers))
print("header bytes =", jsd.header_length)
#for prop in jsd.data_properties:
#    print("data property =", prop.label, prop.format, prop.value)

#print(file_properties.logical_origins)
#print(file_properties.physical_origins)
#print(file_properties.logical_deltas)
#print(file_properties.physical_deltas)
#print(file_properties.trace_format)

iframe = 1
#data = jsd.read_frame_trcs(iframe)
#print(type(data))
#print(data.shape)
#m, n = data.shape
#print(data[0, 0])
#print(data[0,:]) # the first trace

#jsd.read_frame_hdrs(iframe)
print("ILINE_NO =", jsd.get_trace_header("ILINE_NO", 1, 1))
print("XLINE_NO =", jsd.get_trace_header("XLINE_NO", 1, 1))
print("BOS_Z =", jsd.get_trace_header("BOS_Z", 1, 1))
print("VOLUME =", jsd.get_trace_header("VOLUME", 1, 1))
print("CDP_X =", jsd.get_trace_header("CDP_X", 1, 1))
print("CDP_XD =", jsd.get_trace_header("CDP_XD", 1, 1))
print("CDP_Y =", jsd.get_trace_header("CDP_Y", 1, 1))
print("CDP_YD =", jsd.get_trace_header("CDP_YD", 1, 1))
print("TFULL_E =", jsd.get_trace_header("TFULL_E", 1, 1))
print("TRC_TYPE =", jsd.get_trace_header("TRC_TYPE", 1, 1))
