# -*- coding: utf-8 -*-

from distutils.core import setup

# ATTRIBUTES TO INCLUDE
# name, author
# version
# packages,scripts

setup(
    name     = 'PieSeis',
    version  = '0.1dev',
    author   = u"Asbjørn Alexander Fellinghaug",
    author_email = "asbjorn <dot> fellinghaug _dot_ com",
    packages = ['pieseis'],
    license  = 'GNU LGPL',
    long_description = """
    PieSeis is a Python library for reading and writing JavaSeis dataset.
    """,
    install_requires=[
        "lxml>=3.1.0"
    ]
    )
