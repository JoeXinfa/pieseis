# -*- coding: utf-8 -*-
"""
Unittessts for virtual folders
"""

import unittest

import sys
sys.path.append('..')

import pieseis.io.jsfile as jsfile
from pieseis.tests.config_for_test import TEST_DATASET


class TestVirtualFolders(unittest.TestCase):
    def setUp(self):
        self.test_dataset = TEST_DATASET
        self.js_dataset = jsfile.JavaSeisDataset(self.test_dataset)
        self.virtual_folders = self.js_dataset.virtual_folders

    def tearDown(self):
        pass

    def test_get_nr_dimensions(self):
        #self.assertTrue(isinstance(self.virtual_folders.nr_directories, int))
        self.assertIsInstance(self.virtual_folders.nr_directories, int)
        self.assertTrue(len(self.virtual_folders.secondary_folders) > 0)
        print(self.virtual_folders.secondary_folders)


if __name__ == '__main__':
    unittest.main()
