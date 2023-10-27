#!/bin/bash
git submodule update --init --recursive
./clean.sh
cd build
cmake ..
cd ..
./run.sh