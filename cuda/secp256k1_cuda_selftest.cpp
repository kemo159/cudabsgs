#include "secp256k1_cuda.cuh"

#include <cstdio>

int main() {
  if (!cuda_secp256k1_selftest()) {
    std::fprintf(stderr, "CUDA secp256k1 field selftest FAILED\n");
    return 1;
  }
  std::printf("CUDA secp256k1 field selftest OK\n");
  return 0;
}
