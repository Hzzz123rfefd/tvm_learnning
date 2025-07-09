cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)

project(tvm_learnning)

## 添加第三方库环境 tvm
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(USE_LLVM "llvm-config --ignore-libllvm --link-static")
set(HIDE_PRIVATE_SYMBOLS ON)
option(USE_CUDA "Enable CUDA support" OFF)
option(USE_METAL "Enable METAL support" OFF)
option(USE_VULKAN "Enable VULKAN support" OFF)
option(USE_OPENCL "Enable OPENCL support" OFF)
option(USE_CUBLAS "Enable CUBLAS support" OFF)
option(USE_CUDNN "Enable CUDNN support" OFF)
option(USE_CUTLASS "Enable CUTLASS support" OFF)
add_subdirectory(3rd_party/tvm)
include_directories(
    ${CMAKE_SOURCE_DIR}/3rd_party/tvm/include
    ${CMAKE_SOURCE_DIR}/3rd_party/tvm/src
    ${CMAKE_SOURCE_DIR}/3rd_party/tvm/3rdparty/dlpack/include/
    ${CMAKE_SOURCE_DIR}/3rd_party/tvm/3rdparty/dmlc-core/include/
)
add_subdirectory(3rd_party/googletest)
add_subdirectory(inference)