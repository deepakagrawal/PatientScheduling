import os
from setuptools import setup, find_packages
from Cython.Build import cythonize



setup(
    name = "myapp",
    version= "0.1",
    ext_modules = cythonize("cython_*.pyx"),
)