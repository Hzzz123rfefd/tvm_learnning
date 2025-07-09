#include "tvm_model.h"

ModelBase::ModelBase(std::string model_lib_path, std::string weight_path, std::string backend) {
    // get device type
    if(backend == "llvm") dev.device_type = kDLCPU;
    else if(backend == "cuda") dev.device_type = kDLCUDA;
    else if(backend == "opencl") dev.device_type = kDLOpenCL;
    dev.device_id = 0;

    // get model weight
    tvm::runtime::SimpleBinaryFileStream strm(weight_path, "rb");
    auto param_map = tvm::runtime::LoadParams(&strm);
    size_t num_params = param_map.size();
    params.reserve(num_params);
    for (size_t i = 0; i < num_params; ++i) {
        //! NOTE: naming strategy must match convertsion script
        params.push_back(tvm::runtime::NDArray(
                param_map[tvm::runtime::String(std::to_string(i).c_str())].CopyTo(
                        {dev.device_type, 0})));
    }

    // get model
    mod = tvm::runtime::Module::LoadFromFile(model_lib_path);
    tvm::runtime::PackedFunc vm_load_executable = mod.GetFunction("vm_load_executable");
    if (!vm_load_executable.get()) {
        std::cerr << "vm_load_executable function not found!" << std::endl;
        return;
    }
    vm_exec = vm_load_executable(); 
    tvm::runtime::PackedFunc vm_initialization = vm_exec.GetFunction("vm_initialization");
    if (vm_initialization.get() == nullptr) {
        std::cerr << "vm_initialization function not found!" << std::endl;
        return;
    }
    vm_initialization(
        static_cast<int>(dev.device_type), static_cast<int>(0),
        static_cast<int>(tvm::runtime::AllocatorType::kPooled)
    );
    main = vm_exec.GetFunction("main");
    if (main.get() == nullptr) {
        std::cerr << "mian_func function not found!" << std::endl;
        return;
    }
}

tvm::runtime::NDArray ModelBase::inference(tvm::runtime::NDArray &x_tvm){
    tvm::runtime::NDArray y_tvm;
    x_tvm = x_tvm.CopyTo(dev);
    tvm::runtime::TVMRetValue ret = main(x_tvm, params);
    y_tvm = ret.operator tvm::runtime::NDArray();
    return y_tvm;
}

