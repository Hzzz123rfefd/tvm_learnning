python convert/convert_pytorch_demo.py \
    --output ./output_pytroch_andorid_cpu \
    --backend llvm \
    --cross_sm sm_75 \
    --cross_host "llvm -mtriple=aarch64-linux-android" \
    --cross_cc /home/cd_hpc_group/group_common_dirs/NDK/android-ndk-r26c/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android21-clang++