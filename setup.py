#!/usr/bin/env python

from distutils.core import setup

# ATTRIBUTES TO INCLUDE
# name, author
# version
# packages,scripts

setup(
    name     = 'PyJavaSeis',
    version  = '0.1dev',
    packages = ['pyjavaseis'],
    license  = 'LGPL ??',
    long_description = """
PyJavaSeis is a Python implementation of the seismic file format named JavaSeis.
    """
    )