# -*- coding: utf-8 -*-
"""
Test read JavaSeis dataset
"""

import pieseis.io.jsfile as jsfile

PATH_TO_JS_DATASET = '/home/joe/Documents/synth.js'
js_reader = jsfile.JSFileReader()
js_reader.open(PATH_TO_JS_DATASET)

dataset = js_reader.dataset
file_properties = dataset.file_properties

print("Dataset {0}".format(PATH_TO_JS_DATASET))
print("Samples: {0}, Traces: {1}, Frames: {2}".format(
    js_reader.nr_samples,
    js_reader.nr_traces,
    js_reader.nr_frames))
