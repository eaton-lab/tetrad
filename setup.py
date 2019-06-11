#!/usr/bin/env python

from setuptools import setup, find_packages
import glob
import re


# Auto-update ipyrad version from git repo tag
# Fetch version from git tags, and write to version.py.
# Also, when git is not available (PyPi package), use stored version.py.
INITFILE = "tetrad/__init__.py"
CUR_VERSION = (
    re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]",
        open(INITFILE, "r").read(),
        re.M)
    .group(1))


setup(
    name="tetrad",
    version=CUR_VERSION,
    url="https://github.com/eaton-lab/tetrad",
    author="Deren Eaton",
    author_email="de2356@columbia.edu",
    description="Quartet inference using phylogenetic invariants",
    long_description=open('README.rst').read(),
    long_description_content_type='text/x-rst',    
    packages=find_packages(),
    install_requires=[
        "future",
        "ipyparallel>=6.2",
        "scipy>1.0",
        "numpy>=1.9",
        "numba>=0.39",
        "pandas>=0.20",
        "h5py>=2.7",
        "mpi4py",
        # "mkl",
        "toytree>=0.1.21",
    ],
    entry_points={'console_scripts': ['tetrad = tetrad.__main__:main']},
    data_files=[('bin', glob.glob("./bin/*"))],
    license='GPLv3',
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
