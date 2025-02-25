#!/usr/bin/env python

"""Setup instructions.

Install superbpp using conda with the command:
    conda install tetrad -c conda-forge -c bioconda

For developers:
    git clone https://github.com/eaton-lab/tetrad
    cd ./tetrad
    conda env create -f environment.yml
    conda activate dev
    pip install -e . --no-deps
"""

import glob
import re
from setuptools import setup

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
    packages=["tetrad"],
    version=CUR_VERSION,
    url="https://github.com/eaton-lab/tetrad",
    author="Deren Eaton",
    author_email="de2356@columbia.edu",
    description="Quartet supertree inference using phylogenetic invariants",
    long_description=open('README.rst').read(),
    long_description_content_type='text/x-rst',
    # packages=find_packages(),
    install_requires=[
        "scipy",
        "numpy",
        "numba",
        "pandas",
        "h5py>=3.0",
        "toytree>=3.0",
        "ipyparallel>=7.0",
    ],
    entry_points={'console_scripts': ['tetrad = tetrad.src.cli:main']},
    data_files=[('bin', glob.glob("./bin/*"))],
    license='GPLv3',
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
)
