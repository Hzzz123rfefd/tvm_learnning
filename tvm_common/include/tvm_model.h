# pragma
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <runtime/file_utils.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/relax_vm/vm.h>
#include <string>
#include <iostream>
#include <chrono>

class ModelBase{
public:
    ModelBase(std::string model_lib_path, std::string weight_path, std::string backend);
    virtual tvm::runtime::NDArray inference(tvm::runtime::NDArray& input);
protected:
    // models
    tvm::runtime::Module mod;
    tvm::runtime::Module vm_exec;
    tvm::runtime::PackedFunc main;
    tvm::runtime::Array<tvm::runtime::NDArray> params;
    DLDevice dev;
};
