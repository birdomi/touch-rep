#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="tactile_ssl",
    version="0.0.1",
    author="Meta Research",
    description="SSL for multisensory touch data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/facebookresearch/tactile-ssl",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Attribution-NonCommercial 4.0 International License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)
