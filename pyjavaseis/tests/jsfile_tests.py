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

    def tearDown(self):
        pass

    def test_open_js_dataset(self):
        js_dataset = jsfile.JavaSeisDataset(self.test_dataset)
        self.assertTrue(js_dataset.is_valid())
        self.assertTrue(js_dataset.is_open())
        self.assertTrue(js_dataset.close())

    def test_js_file_properties(self):
        js_dataset = jsfile.JavaSeisDataset(self.test_dataset)
        self.assertTrue(isinstance(js_dataset.file_properties, properties.FileProperties))
        #print("JavaSeisDataset FileProperties: %s" % js_dataset.file_properties)

    def test_trace_properties(self):
        js_dataset = jsfile.JavaSeisDataset(self.test_dataset)
        self.assertTrue(isinstance(js_dataset.trace_properties, properties.TraceProperties))
        #print("JavaSeisDataset TraceProperties: %s" % js_dataset.trace_properties)
        #print("Header names: {0}".format(js_dataset.trace_properties.header_names))


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
        print("SOURCE header: {0}".format(self.trace_properties.header_values('SOURCE')))
        self.assertTrue(isinstance(self.trace_properties.header_values('SOURCE'), dict))



if __name__ == '__main__':
    unittest.main()
