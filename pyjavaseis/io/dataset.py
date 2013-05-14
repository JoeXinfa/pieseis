#!/usr/bin/env python

import os

class JavaSeisDataset(object):

    def __init__(self, filename):
        if not filename:
            raise Exception("Bad JavaSeisDataset filename input")
        validate_filename(filename)

        self._filename = filename

    def validate_filename(filename):
        if not os.path.isdir(filename):
            raise IOError("JavaSeis dataset does not exists")

        if not os.access(filename, os.R_OK):
            raise IOError("Missing read access for JavaSeis dataset")
