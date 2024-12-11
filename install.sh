#!/bin/bash
ROMAN_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $ROMAN_DIR

# Install CLIPPER
git submodule update --init --recursive
mkdir dependencies/clipper/build
cd dependencies/clipper/build
cmake .. && make && make pip-install

# pip install
cd $ROMAN_DIR
pip install .