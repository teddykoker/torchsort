#!/usr/bin/env python3

import os

from setuptools import setup
from torch.utils import cpp_extension

directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="torchsort",
    version="0.0.1",
    description="Differentiable sorting and ranking in PyTorch",
    author="Teddy Koker",
    url="https://github.com/teddykoker/torchsort",
    license="Apache",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["torchsort"],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    install_requires=["torch"],
    python_requires=">=3.8",
    extras_require={
        "testing": [
            "pytest",
        ],
    },
    ext_modules=[
        cpp_extension.CppExtension("torchsort_cpp", ["torchsort/isotonic.cpp"])
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    include_package_data=True,
)
