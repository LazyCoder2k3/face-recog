#!/bin/bash

# Set environment variables
export ROOT_DIR=/mnt/i/AnhTu/Internship/working
export OPENCV_INCLUDE=$ROOT_DIR/opencv4.10/include
export OPENCV_LIB=$ROOT_DIR/opencv4.10/lib
export CROSS_COMPILE=$ROOT_DIR/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-
export VIVANTE_SDK_DIR=$ROOT_DIR/6.4.15.9

echo "============================================"
echo "Building FaceRecog_wrapper.so"
echo "============================================"
echo "ROOT_DIR: $ROOT_DIR"
echo "OPENCV_INCLUDE: $OPENCV_INCLUDE"
echo "OPENCV_LIB: $OPENCV_LIB"
echo "CROSS_COMPILE: $CROSS_COMPILE"
echo "VIVANTE_SDK_DIR: $VIVANTE_SDK_DIR"
echo "============================================"

# Navigate to build directory
cd /mnt/i/AnhTu/VNEMEX/face-recog/AI_model

# Clean previous build
echo "Cleaning previous build..."
make clean

# Build
echo "Building..."
make

# Check if build succeeded
if [ -f "libFaceRecog_wrapper.so" ]; then
    echo "============================================"
    echo "✅ Build successful!"
    echo "File: libFaceRecog_wrapper.so"
    ls -lh libFaceRecog_wrapper.so
    echo "============================================"
    
    # Copy to server
    echo "Copying to server..."
    scp libFaceRecog_wrapper.so itri@10.60.3.235:/home/itri/Working/NATu/FaceRecognition-FAISS
    
    if [ $? -eq 0 ]; then
        echo "✅ File copied successfully to server!"
    else
        echo "❌ Failed to copy file to server"
    fi
else
    echo "❌ Build failed - libFaceRecog_wrapper.so not found"
    exit 1
fi
