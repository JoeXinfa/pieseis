"""
With unittests
"""

import unittest
import sys
sys.path.append('..')

import pyjavaseis.io.jsfile as jsfile
#from io import jsfile
#from io import jsfile
#import io.jsfile as jsfile
from pyjavaseis.io import properties

TEST_DATASET = "/home/asbjorn/datasets/2hots.js"

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
        #print("JavaSeisDataset FileProperties: %s" % js_dataset._file_properties)


if __name__ == '__main__':
    unittest.main()
