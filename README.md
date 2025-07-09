# tvm_learnning
tvm学习仓库，包括了tvm环境搭建、tvm模型转换到各平台各后端、tvm的python、pytorch推理
## installation
### install tvm env
```bash 
    conda env remove -n tvm
    conda create -n tvm -c conda-forge \
        "llvmdev>=15" \
        "cmake>=3.24" \
        git \
        python=3.11
    conda activate tvm
    cd 3rd_party/tvm
    rm -rf build && mkdir build && cd build
    cp ../cmake/config.cmake .
    echo "set(CMAKE_BUILD_TYPE RelWithDebInfo)" >> config.cmake
    echo "set(USE_LLVM \"llvm-config --ignore-libllvm --link-static\")" >> config.cmake
    echo "set(HIDE_PRIVATE_SYMBOLS ON)" >> config.cmake
    echo "set(USE_CUDA   ON)" >> config.cmake
    echo "set(USE_OPENCL ON)" >> config.cmake
    cmake .. && cmake --build . --parallel $(nproc)
    export TVM_HOME=path/tvm_learnning/3rd_party/tvm
    export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH
```
## install others
```bash 
    conda activate tvm
    pip install -r requirments.txt
```
## Step1: convert model to tvm so
你可以通过以下脚本将pytorch模型转为各平台各后端tvm动态库

```bash 
    sh -x script/convert_pytorch_to_linux_cpu.sh
```
```bash 
    sh -x script/convert_pytorch_to_linux_cuda.sh
```
```bash 
    sh -x script/convert_pytorch_to_linux_opencl.sh
```
```bash 
    sh -x script/convert_pytorch_to_android_cpu.sh
```
```bash 
    sh -x script/convert_pytorch_to_android_cuda.sh
```

## Step1: 模型推理
### python 推理
你可以使用下面的python脚本进行推理
```bash 
    sh -x script/inference_pytorch_linux_cpu.sh
```
```bash 
    sh -x script/inference_pytorch_linux_cuda.sh
```
```bash 
    sh -x script/inference_pytorch_linux_opencl.sh
```

### C++推理
#### 编译
编译分为本地（linux）编译和交叉（android）编译
本地编译：
```bash 
    sh -x script/build_local.sh
```
交叉编译：
```bash 
    sh -x script/build_cross_android_arm.sh
```


