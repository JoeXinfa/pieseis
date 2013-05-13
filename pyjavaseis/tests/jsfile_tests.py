"""
With unittests
"""

import unittest

import sys
sys.path.append('..')

import pyjavaseis.io.jsfile as jsfile
from pyjavaseis.io import properties

try:
    from config_for_test import TEST_DATASET
except ImportError, ie:
    from tests.config_for_test import TEST_DATASET


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
        self.assertTrue(isinstance(js_dataset._file_properties, properties.FileProperties))
        print("JavaSeisDataset FileProperties: %s" % js_dataset._file_properties)

        # TODO: Check _ALL_ get methods on object

class TestFileProperties(unittest.TestCase):
    def setUp(self):
        self.test_dataset = TEST_DATASET
        self.js_dataset = jsfile.JavaSeisDataset(self.test_dataset)
        self.file_properties = self.js_dataset.get_file_properties()

    def tearDown(self):
        pass

    def test_get_nr_dimensions(self):
        self.assertTrue(isinstance(self.file_properties.get_nr_dimensions(), int))
        self.assertTrue(self.file_properties.get_nr_dimensions() > 0)


if __name__ == '__main__':
    unittest.main()
