
"""
Unittessts for various parts of the python javaseis implementation.
"""

import unittest

import sys
sys.path.append('..')

import pyjavaseis.io.jsfile as jsfile
from pyjavaseis.io import properties
from pyjavaseis.tests.config_for_test import TEST_DATASET


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