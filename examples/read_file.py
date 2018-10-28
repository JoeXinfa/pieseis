# -*- coding: utf-8 -*-
"""
Test read JavaSeis dataset
"""

import pieseis.io.jsfile as jsfile

PATH_TO_JS_DATASET = '/home/joe/Documents/181026_test.js'
js_reader = jsfile.JSFileReader()
js_reader.open(PATH_TO_JS_DATASET)

dataset = js_reader.dataset
file_properties = dataset.file_properties

print("Dataset {0}".format(PATH_TO_JS_DATASET))
print("Samples: {0}, Traces: {1}, Frames: {2}".format(
    js_reader.nr_samples,
    js_reader.nr_traces,
    js_reader.nr_frames))

print(file_properties.axis_labels)
print(file_properties.axis_lengths)
print(file_properties.logical_origins)
print(file_properties.physical_origins)
print(file_properties.logical_deltas)
print(file_properties.physical_deltas)
print(file_properties.trace_format)
