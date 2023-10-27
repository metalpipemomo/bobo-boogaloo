#!/bin/bash
cd build/
make

DIR_NAME=$(basename $(pwd))

if [[ -f "./$DIR_NAME" ]]; then
    ./$DIR_NAME "$@"
else
    echo "Executable $DIR_NAME not found!"
    exit 1
fi
