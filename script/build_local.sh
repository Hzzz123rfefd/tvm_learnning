#!/bin/bash
BUILD_DIR=build
mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_CUDA=ON \
  -DUSE_OPENCL=ON
make -j$(nproc)