
"""
Unittessts for various parts of the python javaseis implementation.
"""

import unittest

import sys
sys.path.append('..')

import pieseis.io.jsfile as jsfile
from pieseis.io import properties
from pieseis.tests.config_for_test import TEST_DATASET


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

    def tearDown(self):
        pass

    def assert_exists_and_string(self, prop):
        self.assertIsNotNone(prop)
        self.assertIsInstance(prop, str)
        self.assertTrue(len(prop)>0)

    def test_is_synthetic(self):
        self.assertFalse(self.custom_properties.synthetic)

    def test_secondary_key(self):
        self.assertIsInstance(self.custom_properties.secondary_key, str)

    def test_geometry_matches_flag(self):
        self.assertIsInstance(self.custom_properties.geometry_matches_flag, int)

    def test_primary_key(self):
        self.assert_exists_and_string(self.custom_properties.primary_key)
        # TODO: Maybe also check that the value is a VALID header??

    def test_primary_sort(self):
        self.assert_exists_and_string(self.custom_properties.primary_sort)

    def test_trace_no_matches_flag(self):
        self.assertIsInstance(self.custom_properties.trace_no_matches_flag, int)

    def test_stacked(self):
        self.assertIsNotNone(self.custom_properties.stacked)
        self.assertIsInstance(self.custom_properties, bool)

    def test_cookie(self):
        self.assertIsNotNone(self.custom_properties.cookie)
        self.assertIsInstance(self.custom_properties.cookie, int)


if __name__ == '__main__':
    unittest.main()