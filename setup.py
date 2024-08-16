#!/usr/bin/env python
from setuptools import setup
from codecs import open
from os import path


ext_modules = []

here = path.abspath(path.dirname(__file__))


setup(name='evetac_software',
      description='software package to read out the Evetac sensor',
      version='1.0.0',
      author='Niklas Funk',
      author_email='niklas@robot-learning.de',
      packages=['utils'],
      )