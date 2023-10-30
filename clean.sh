#!/bin/bash

# Define build directory and cache file
BUILD_DIR="./build"

# Check if the build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo "Nothing to clean; $BUILD_DIR directory does not exist."
    exit 1
fi

# Remove build dir
rm -rf "$BUILD_DIR"