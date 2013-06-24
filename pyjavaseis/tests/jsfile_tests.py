
"""
With unittests
"""

import unittest

import sys
sys.path.append('..')

import pyjavaseis.io.jsfile as jsfile
from pyjavaseis.io import properties
from pyjavaseis.tests.config_for_test import TEST_DATASET


class TestJSFileReader(unittest.TestCase):
    def setUp(self):
        self.test_dataset = TEST_DATASET
        self.js_reader = jsfile.JSFileReader()
        self.js_reader.open(TEST_DATASET)
        self.js_dataset = self.js_reader.dataset

    def tearDown(self):
        pass

    def test_open_js_dataset(self):
        self.assertTrue(self.js_dataset.is_valid())
        self.assertTrue(self.js_dataset.is_open())
        self.assertTrue(self.js_dataset.close())

    def test_js_file_properties(self):
        self.assertTrue(isinstance(self.js_dataset.file_properties, properties.FileProperties))
        #print("JavaSeisDataset FileProperties: %s" % js_dataset.file_properties)

    def test_trace_properties(self):
        self.assertTrue(isinstance(self.js_dataset.trace_properties, properties.TraceProperties))
        #print("JavaSeisDataset TraceProperties: %s" % js_dataset.trace_properties)
        #print("Header names: {0}".format(js_dataset.trace_properties.header_names))

    def test_nr_of_samples(self):
        self.assertIsInstance(self.js_reader.nr_samples, long, msg="The nr of samples must be of type long.")
        self.assertGreater(self.js_reader.nr_samples, 0, msg="The number of samples must be greater than 0.")
        print("Dataset: {0}, # samples: {1}".format(self.test_dataset, self.js_reader.nr_samples))

    def test_nr_of_traces(self):
        self.assertIsInstance(self.js_reader.nr_traces, long, msg="The nr of traces must be of type long.")
        self.assertGreater(self.js_reader.nr_traces, 0, msg="The number of traces must be greater than 0.")
        print("Dataset: {0}, # traces: {1}".format(self.test_dataset, self.js_reader.nr_traces))

    def test_total_nr_of_frames(self):
        self.assertIsInstance(self.js_reader.total_nr_of_frames, long, msg="The total number of frames must be a long. Got: {0}".format(type(self.js_reader.total_nr_of_frames)))
        self.assertGreater(self.js_reader.total_nr_of_frames, 0, msg="Total number of frames must be 1 or larger")
        print("Dataset: {0}, # of frames: {1}".format(self.test_dataset, self.js_reader.total_nr_of_frames))


class TestFileProperties(unittest.TestCase):
    def setUp(self):
        self.test_dataset = TEST_DATASET
        self.js_dataset = jsfile.JavaSeisDataset(self.test_dataset)
        self.file_properties = self.js_dataset.file_properties

    def tearDown(self):
        pass

    def test_get_nr_dimensions(self):
        self.assertTrue(isinstance(self.file_properties.nr_dimensions, int))
        self.assertTrue(self.file_properties.nr_dimensions > 0)

class TestTraceProperties(unittest.TestCase):
    def setUp(self):
        self.test_dataset = TEST_DATASET
        self.js_dataset = jsfile.JavaSeisDataset(self.test_dataset)
        self.trace_properties = self.js_dataset.trace_properties

    def tearDown(self):
        pass

    def test_get_all_header_names(self):
        self.assertTrue(isinstance(self.trace_properties.header_names, list))
        self.assertTrue(len(self.trace_properties.header_names) > 0)

    def test_get_source_header(self):
        self.assertIsInstance(self.trace_properties.header_values('SOURCE'), properties.TraceHeader)

    def test_header_is_trace_header_object(self):
        source_header = self.trace_properties.header_values('SOURCE')
        self.assertIsInstance(source_header.byte_offset, int)
        self.assertIsInstance(source_header.element_count, int)


class TestCustomProperties(unittest.TestCase):
    def setUp(self):
        self.test_dataset = TEST_DATASET
        self.js_dataset = jsfile.JavaSeisDataset(self.test_dataset)
        self.custom_properties = self.js_dataset.custom_properties

if __name__ == '__main__':
    unittest.main()