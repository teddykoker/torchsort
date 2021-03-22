#!/usr/bin/env python3

import os
import sys

from setuptools import setup
from torch.utils import cpp_extension

directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


def compile_args():
    args = ["-fopenmp", "-ffast-math"]
    if sys.platform == "darwin":
        return ["-Xpreprocessor"] + args
    return args


setup(
    name="torchsort",
    version="0.0.6",
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
            "torch",
            "fast_soft_sort @ git+https://github.com/google-research/fast-soft-sort.git@c3110360d7c94c42027865c71b23e46fa22151e2",
        ],
    },
    ext_modules=[
        cpp_extension.CppExtension(
            "torchsort.isotonic_cpu",
            sources=["torchsort/isotonic_cpu.cpp"],
            extra_compile_args=compile_args(),
        ),
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    include_package_data=True,
)
