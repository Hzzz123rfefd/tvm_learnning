#!/bin/bash
BUILD_DIR=build_android_arm
mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-28 \
    -DBUILD_ANDROID_PROJECTS=OFF \
    -DUSE_OPENCL=ON \
    -DUSE_CUDA=OFF \
    -Dzstd_LIBRARY=$CONDA_PREFIX/lib/libzstd.so \
    -Dzstd_INCLUDE_DIR=$CONDA_PREFIX/include 
make -j$(nproc)