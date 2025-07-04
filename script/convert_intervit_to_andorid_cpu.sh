python convert/convert_pytorch_demo.py \
    --output ./output_intervit_andorid_cpu \
    --pipeline internvit_opt \
    --inject_hyper_params convert/internvit_hyper_params.json \
    --backend llvm \
    --cross_sm sm_75 \
    --cross_host "llvm -mtriple=aarch64-linux-android" \
    --cross_cc /home/cd_hpc_group/group_common_dirs/NDK/android-ndk-r26c/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android21-clang++