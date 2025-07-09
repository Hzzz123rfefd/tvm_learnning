#!/bin/bash
BUILD_DIR=build_android_arm
REMOTE_DIR="/data/local/tmp/tvm"

MODEL_DIR="build_android_arm/install/output_pytroch_andorid_opencl/"
MODEL_DIR2="build_android_arm/install/output_intervit_android_opencl/"
EXE_PATH="build_android_arm/install/inference"
LIB_PATH="build_android_arm/3rd_party/tvm/libtvm_runtime.so"
LIB_PATH2="/home/cd_hpc_group/group_common_dirs/NDK/android-ndk-r26c/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"

# 构建
sh -x script/build_cross_android_arm.sh

# 创建远程目录
adb shell "mkdir -p $REMOTE_DIR"

# 推送文件
adb push $MODEL_DIR $REMOTE_DIR
adb push $MODEL_DIR2 $REMOTE_DIR
adb push $EXE_PATH $REMOTE_DIR
adb push $LIB_PATH $REMOTE_DIR
adb push $LIB_PATH2 $REMOTE_DIR

# 运行文件
adb shell "cd $REMOTE_DIR && LD_LIBRARY_PATH=$REMOTE_DIR ./inference"

echo "Files pushed successfully."