#include "tvm_model.h"
#include "gtest/gtest.h"

#if __ANDROID__
    TEST(PYTORCH, OPENCL) {
        ModelBase model("output_pytroch_andorid_opencl/output_pytroch_andorid_opencl.so", "output_pytroch_andorid_opencl/output_pytroch_andorid_opencl.bin", "opencl");
        DLDevice dev{kDLCPU, 0};
        tvm::runtime::NDArray x_tvm = tvm::runtime::NDArray::Empty({1, 784}, {kDLFloat, 32, 1}, dev);
        std::fill((float*)x_tvm->data, (float*)x_tvm->data + 784, 1.0f);

        tvm::runtime::NDArray y_tvm = model.inference(x_tvm).CopyTo(dev);

        float* data = static_cast<float*>(y_tvm->data);
        int64_t total = 1;
        for (int64_t dim : y_tvm.Shape()) {
            total *= dim;
        }
        std::cout << "output (first 10 values): ";
        for (int64_t i = 0; i < std::min<int64_t>(10, total); ++i) {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;
    }
    TEST(INTERVIT, OPENCL) {
        ModelBase model("output_intervit_android_opencl/output_intervit_android_opencl.so", "output_intervit_android_opencl/output_intervit_android_opencl.bin", "opencl");
        DLDevice dev{kDLCPU, 0};
        tvm::runtime::NDArray x_tvm = tvm::runtime::NDArray::Empty({1, 448, 448, 3}, {kDLFloat, 32, 1}, dev);
        std::fill((float*)x_tvm->data, (float*)x_tvm->data + 448 * 448 * 3, 1.0f);

        tvm::runtime::NDArray y_tvm = model.inference(x_tvm).CopyTo(dev);

        float* data = static_cast<float*>(y_tvm->data);
        int64_t total = 1;
        for (int64_t dim : y_tvm.Shape()) {
            total *= dim;
        }
        std::cout << "output (first 10 values): ";
        for (int64_t i = 0; i < std::min<int64_t>(10, total); ++i) {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;
    }
#else
    TEST(PYTORCH, CUDA) {
        ModelBase model("output_pytroch_linux_cuda/output_pytroch_linux_cuda.so", "output_pytroch_linux_cuda/output_pytroch_linux_cuda.bin", "cuda");
        DLDevice dev{kDLCPU, 0};
        tvm::runtime::NDArray x_tvm = tvm::runtime::NDArray::Empty({1, 784}, {kDLFloat, 32, 1}, dev);
        std::fill((float*)x_tvm->data, (float*)x_tvm->data + 784, 1.0f);

        tvm::runtime::NDArray y_tvm = model.inference(x_tvm).CopyTo(dev);

        float* data = static_cast<float*>(y_tvm->data);
        int64_t total = 1;
        for (int64_t dim : y_tvm.Shape()) {
            total *= dim;
        }
        std::cout << "output (first 10 values): ";
        for (int64_t i = 0; i < std::min<int64_t>(10, total); ++i) {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;
    }
    
    TEST(PYTORCH, OPENCL) {
        ModelBase model("output_pytroch_linux_opencl/output_pytroch_linux_opencl.so", "output_pytroch_linux_opencl/output_pytroch_linux_opencl.bin", "opencl");
        DLDevice dev{kDLCPU, 0};
        tvm::runtime::NDArray x_tvm = tvm::runtime::NDArray::Empty({1, 784}, {kDLFloat, 32, 1}, dev);
        std::fill((float*)x_tvm->data, (float*)x_tvm->data + 784, 1.0f);

        tvm::runtime::NDArray y_tvm = model.inference(x_tvm).CopyTo(dev);

        float* data = static_cast<float*>(y_tvm->data);
        int64_t total = 1;
        for (int64_t dim : y_tvm.Shape()) {
            total *= dim;
        }
        std::cout << "output (first 10 values): ";
        for (int64_t i = 0; i < std::min<int64_t>(10, total); ++i) {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;
    }
    
    /**
     * 太慢了，先关了
     */
    // TEST(INTERVIT, CPU) {
    //     ModelBase model("output_intervit_linux_cpu/output_intervit_linux_cpu.so", "output_intervit_linux_cpu/output_intervit_linux_cpu.bin", "llvm");
    //     DLDevice dev{kDLCPU, 0};
    //     tvm::runtime::NDArray x_tvm = tvm::runtime::NDArray::Empty({1, 448, 448, 3}, {kDLFloat, 32, 1}, dev);
    //     std::fill((float*)x_tvm->data, (float*)x_tvm->data + 448 * 448 * 3, 1.0f);

    //     tvm::runtime::NDArray y_tvm = model.inference(x_tvm).CopyTo(dev);

    //     float* data = static_cast<float*>(y_tvm->data);
    //     int64_t total = 1;
    //     for (int64_t dim : y_tvm.Shape()) {
    //         total *= dim;
    //     }
    //     std::cout << "output (first 10 values): ";
    //     for (int64_t i = 0; i < std::min<int64_t>(10, total); ++i) {
    //         std::cout << data[i] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    
    TEST(INTERVIT, CUDA) {
        ModelBase model("output_intervit_linux_cuda/output_intervit_linux_cuda.so", "output_intervit_linux_cuda/output_intervit_linux_cuda.bin", "cuda");
        DLDevice dev{kDLCPU, 0};
        tvm::runtime::NDArray x_tvm = tvm::runtime::NDArray::Empty({1, 448, 448, 3}, {kDLFloat, 32, 1}, dev);
        std::fill((float*)x_tvm->data, (float*)x_tvm->data + 448 * 448 * 3, 1.0f);

        tvm::runtime::NDArray y_tvm = model.inference(x_tvm).CopyTo(dev);

        float* data = static_cast<float*>(y_tvm->data);
        int64_t total = 1;
        for (int64_t dim : y_tvm.Shape()) {
            total *= dim;
        }
        std::cout << "output (first 10 values): ";
        for (int64_t i = 0; i < std::min<int64_t>(10, total); ++i) {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;
    }
   
    TEST(INTERVIT, OPENCL) {
        ModelBase model("output_intervit_linux_opencl/output_intervit_linux_opencl.so", "output_intervit_linux_opencl/output_intervit_linux_opencl.bin", "opencl");
        DLDevice dev{kDLCPU, 0};
        tvm::runtime::NDArray x_tvm = tvm::runtime::NDArray::Empty({1, 448, 448, 3}, {kDLFloat, 32, 1}, dev);
        std::fill((float*)x_tvm->data, (float*)x_tvm->data + 448 * 448 * 3, 1.0f);

        tvm::runtime::NDArray y_tvm = model.inference(x_tvm).CopyTo(dev);

        float* data = static_cast<float*>(y_tvm->data);
        int64_t total = 1;
        for (int64_t dim : y_tvm.Shape()) {
            total *= dim;
        }
        std::cout << "output (first 10 values): ";
        for (int64_t i = 0; i < std::min<int64_t>(10, total); ++i) {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;
    }
#endif
