#include <torch/extension.h>

#include <iostream>
#include <cassert>

void lookupmatmul_d4_k8(
    torch::Tensor X,        // k x n
    torch::Tensor YIs,      // m x (n/4)
    torch::Tensor CB,       // 256 x 4
    torch::Tensor Z         // k x m
);

void lookupmatmul_d4_k16(
    torch::Tensor X,        // k x n
    torch::Tensor YIs,      // m x (n/4)
    torch::Tensor CB,       // 256 x 4
    torch::Tensor Z         // k x m
);

void lookupmatmul_d4_k32(
    torch::Tensor X,        // k x n
    torch::Tensor YIs,      // m x (n/4)
    torch::Tensor CB,       // 256 x 4
    torch::Tensor Z         // k x m
);

void decompress_d4(
    torch::Tensor YIs,      // m x (n/4)
    torch::Tensor CB,       // 256 x 4
    torch::Tensor Y         // m x n
);

void decompress_d4_origorder(
    torch::Tensor YIs,      // m x (n/4)
    torch::Tensor CB,       // 256 x 4
    torch::Tensor Y         // m x n
);

void decompress_e8p_origorder(
    torch::Tensor YIs,      // m x (n/8)
    torch::Tensor CB,       // 256 x 8
    torch::Tensor CB_even_flips, // 256
    torch::Tensor &Y         // m x n
);

torch::Tensor decode_matmul_e8p(
    torch::Tensor x,
    torch::Tensor weights_compressed,
    torch::Tensor codebook_abs
);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("lookupmatmul_d4_k8", &lookupmatmul_d4_k8, "lookupmatmul_d4_k8");
  m.def("lookupmatmul_d4_k16", &lookupmatmul_d4_k16, "lookupmatmul_d4_k16");
  m.def("lookupmatmul_d4_k32", &lookupmatmul_d4_k32, "lookupmatmul_d4_k32");
  m.def("decompress_d4", &decompress_d4, "decompress_d4");
  m.def("decompress_d4_origorder", &decompress_d4_origorder, "decompress_d4_origorder");
  m.def("decompress_e8p_origorder", &decompress_e8p_origorder, "decompress_e8p_origorder");
  m.def("decode_matmul_e8p", &decode_matmul_e8p, "decode_matmul_e8p");
}

