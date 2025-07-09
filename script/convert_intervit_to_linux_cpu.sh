python convert/convert_internvit_demo.py \
    --weight OpenGVLab/InternVL3-1B \
    --output ./output_intervit_linux_cpu \
    --pipeline internvit_opt \
    --inject_hyper_params convert/internvit_hyper_params.json \
    --backend llvm 