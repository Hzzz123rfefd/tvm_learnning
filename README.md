# tvm_learnning


## install tvm
    conda env remove -n tvm
    conda create -n tvm-build-venv -c conda-forge \
        "llvmdev>=15" \
        "cmake>=3.24" \
        git \
        python=3.11
    conda activate tvm-build-venv
    cd 3rd_party/tvm
    rm -rf build && mkdir build && cd build
    cp ../cmake/config.cmake .
    echo "set(CMAKE_BUILD_TYPE RelWithDebInfo)" >> config.cmake
    echo "set(USE_LLVM \"llvm-config --ignore-libllvm --link-static\")" >> config.cmake
    echo "set(HIDE_PRIVATE_SYMBOLS ON)" >> config.cmake
    echo "set(USE_CUDA   ON)" >> config.cmake
    echo "set(USE_OPENCL ON)" >> config.cmake
    cmake .. && cmake --build . --parallel $(nproc)
    export TVM_HOME=/path-to-tvm
    export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH


