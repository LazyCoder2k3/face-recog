#!/bin/bash

# Set environment variables from build_wrapper.sh
export ROOT_DIR=/mnt/i/AnhTu/Internship/working
export OPENCV_INCLUDE=$ROOT_DIR/opencv4.10/include
export OPENCV_LIB=$ROOT_DIR/opencv4.10/lib
export CROSS_COMPILE=$ROOT_DIR/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-
export VIVANTE_SDK_DIR=$ROOT_DIR/6.4.15.9

# Clean previous build
rm -rf build

# Create build directory
mkdir -p build
cd build

# Run CMake
echo "Running CMake..."
cmake -DCMAKE_C_COMPILER=${CROSS_COMPILE}gcc \
      -DCMAKE_CXX_COMPILER=${CROSS_COMPILE}g++ \
      -DCMAKE_FIND_ROOT_PATH=${ROOT_DIR} \
      -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY \
      -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
      -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER \
      ..

# Build
echo "Building..."
make -j4 VERBOSE=1

# Check result
if [ -f "face_recognition_pybind.so" ]; then
    echo "✅ Build successful!"
    ls -lh face_recognition_pybind.so
else
    echo "❌ Build failed!"
    exit 1
fi
