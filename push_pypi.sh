#!/bin/bash -e
rm -rf dist
python setup.py sdist
twine upload dist/*


