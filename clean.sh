#!/bin/bash

# Define build directory and cache file
BUILD_DIR="./build"
CACHE_FILE="CMakeCache.txt"

# Check if the build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo "Error: $BUILD_DIR directory does not exist."
    exit 1
fi

# Path to the cache file in the build directory
CACHE_PATH="$BUILD_DIR/$CACHE_FILE"

# Check if the CMakeCache.txt file exists
if [ -f "$CACHE_PATH" ]; then
    # Delete the CMakeCache.txt file
    rm -rf "$CACHE_PATH"
    rm -rf "CMakeFiles"

    if [ $? -ne 0 ]; then
        echo "Error: Failed to delete $CACHE_PATH."
        exit 1
    else
        echo "$CACHE_PATH deleted successfully."
    fi
else
    echo "Warning: $CACHE_PATH does not exist."
fi

