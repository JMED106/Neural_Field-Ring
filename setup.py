#!python
# cython: boundscheck=False

from distutils.core import setup
from distutils.extension import Extension

from Cython.Build import cythonize

ext_modules = [
    Extension(
        "qifint",
        ["qifint.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    name='qifint-parallel',
    ext_modules=cythonize(ext_modules),
)
