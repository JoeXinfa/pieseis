#!/usr/bin/env python

from distutils.core import setup

# ATTRIBUTES TO INCLUDE
# name, author
# version
# packages,scripts

setup(
    name     = 'PyJavaSeis',
    version  = '0.1dev',
    author   = u"Asbjørn Alexander Fellinghaug",
    author_email = "asbjorn <dot> fellinghaug _dot_ com"
    packages = ['pyjavaseis'],
    license  = 'GNU LGPL',
    long_description = """
PyJavaSeis is a Python implementation of the seismic file format named JavaSeis.
    """
    )