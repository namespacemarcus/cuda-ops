#include "elementwise_add.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <torch/extension.h>
#include <torch/types.h>

#define STRINGFY(str) #str

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
    if (((T).options().dtype() != (th_type))) {                                \
        std::cout << "Tensor Info: " << (T).options() << std::endl;            \
        throw std::runtime_error("values must be " #th_type);                  \
    }

#define TORCH_BINDING_ELEMENTWISE_ADD(packed_type, th_type, element_type,      \
                                      n_elements)                              \
    void elementwise_add_##packed_type(torch::Tensor a, torch::Tensor b,       \
                                       torch::Tensor c) {                      \
        CHECK_TORCH_TENSOR_DTYPE(a, th_type);                                  \
        CHECK_TORCH_TENSOR_DTYPE(b, th_type);                                  \
        CHECK_TORCH_TENSOR_DTYPE(c, th_type);                                  \
        const int ndim = a.dim();                                              \
        if (ndim != 2) {                                                       \
            int N = 1;                                                         \
            for (int i = 0; i < ndim; ++i) {                                   \
                N *= a.size(i);                                                \
            }                                                                  \
            dim3 block(256 / (n_elements));                                    \
            dim3 grid((N + 256 - 1) / 256);                                    \
            elementwise_add_##packed_type##_kernel<<<grid, block>>>(           \
                reinterpret_cast<element_type *>(a.data_ptr()),                \
                reinterpret_cast<element_type *>(b.data_ptr()),                \
                reinterpret_cast<element_type *>(c.data_ptr()), N);            \
        } else {                                                               \
            const int S = a.size(0);                                           \
            const int K = a.size(1);                                           \
            const int N = S * K;                                               \
            if ((K / (n_elements)) <= 1024) {                                  \
                dim3 block(K / (n_elements));                                  \
                dim3 grid(S);                                                  \
                elementwise_add_##packed_type##_kernel<<<grid, block>>>(       \
                    reinterpret_cast<element_type *>(a.data_ptr()),            \
                    reinterpret_cast<element_type *>(b.data_ptr()),            \
                    reinterpret_cast<element_type *>(c.data_ptr()), N);        \
            } else {                                                           \
                int N = 1;                                                     \
                for (int i = 0; i < ndim; ++i) {                               \
                    N *= a.size(i);                                            \
                }                                                              \
                dim3 block(256 / (n_elements));                                \
                dim3 grid((N + 256 - 1) / 256);                                \
                elementwise_add_##packed_type##_kernel<<<grid, block>>>(       \
                    reinterpret_cast<element_type *>(a.data_ptr()),            \
                    reinterpret_cast<element_type *>(b.data_ptr()),            \
                    reinterpret_cast<element_type *>(c.data_ptr()), N);        \
            }                                                                  \
        }                                                                      \
    }

TORCH_BINDING_ELEMENTWISE_ADD(f32, torch::kFloat32, float, 1)
TORCH_BINDING_ELEMENTWISE_ADD(f32x4, torch::kFloat32, float, 4)
TORCH_BINDING_ELEMENTWISE_ADD(f16, torch::kHalf, half, 1)
TORCH_BINDING_ELEMENTWISE_ADD(f16x2, torch::kHalf, half, 2)
TORCH_BINDING_ELEMENTWISE_ADD(f16x8, torch::kHalf, half, 8)
TORCH_BINDING_ELEMENTWISE_ADD(f16x8_pack, torch::kHalf, half, 8)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Element-wise add CUDA kernels.";

    m.def("elementwise_add_f32", &elementwise_add_f32, "F32 element-wise add: c = a + b (1 thread / 1 element).",
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"));
    m.def("elementwise_add_f32x4", &elementwise_add_f32x4,
          "F32 element-wise add: c = a + b (1 threads / 4 elements).", pybind11::arg("a"), pybind11::arg("b"),
          pybind11::arg("c"));
    m.def("elementwise_add_f16", &elementwise_add_f16, "F16 element-wise add: c = a + b (1 thread / 1 half).",
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"));
    m.def("elementwise_add_f16x2", &elementwise_add_f16x2,
          "F16x2 element-wise add: c = a + b (1 threads / 2 half).", pybind11::arg("a"), pybind11::arg("b"),
          pybind11::arg("c"));
    m.def("elementwise_add_f16x8", &elementwise_add_f16x8,
          "F16 element-wise add: c = a + b (1 threads / 8 halves, unpacked).", pybind11::arg("a"), pybind11::arg("b"),
          pybind11::arg("c"));
    m.def("elementwise_add_f16x8_pack", &elementwise_add_f16x8_pack,
          "F16_pack element-wise add: c = a + b , 128-bit LDST (1 threads / 8 halves).", pybind11::arg("a"), pybind11::arg("b"),
          pybind11::arg("c"));
}
