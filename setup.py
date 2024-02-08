# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 21:34:34 2023

@author: tjostmou
"""

from setuptools import setup, find_packages
from pathlib import Path


def get_version(rel_path):
    here = Path(__file__).parent.absolute()
    with open(here.joinpath(rel_path), "r") as fp:
        for line in fp.read().splitlines():
            if line.startswith("__version__"):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


setup(
    name="pImage",
    version=get_version(Path("pImage", "__init__.py")),
    packages=find_packages(),
    url="https://github.com/JostTim/pImage",
    license="MIT",
    author="Timothé Jost-MOUSSEAU",
    author_email="timothe.jost-mousseau@pasteur.com",
    description="Image and video management for research use, in an unified easy to use API",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    install_requires=["numpy>=1.23", "opencv-python>=4.6", "ffmpeg>=1.4", "tifffile>=2022.10"],
    entry_points={},
    scripts={},
)
