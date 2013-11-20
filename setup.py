#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
    install_requires=[
        "lxml>=3.1.0"
    ]
    )
