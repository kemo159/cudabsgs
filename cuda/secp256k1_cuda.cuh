#ifndef KEYHUNT_CUDA_SECP256K1_CUH
#define KEYHUNT_CUDA_SECP256K1_CUH

#include <stdint.h>

#ifndef __CUDACC__
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#endif

typedef struct cuda_u256 {
  uint64_t v[4];
} cuda_u256;

typedef struct cuda_point_affine {
  cuda_u256 x;
  cuda_u256 y;
  int infinity;
} cuda_point_affine;

__host__ __device__ cuda_u256 cuda_secp_p();
__host__ __device__ int cuda_u256_cmp(const cuda_u256 &a, const cuda_u256 &b);
__host__ __device__ cuda_u256 cuda_u256_add_mod_p(const cuda_u256 &a, const cuda_u256 &b);
__host__ __device__ cuda_u256 cuda_u256_sub_mod_p(const cuda_u256 &a, const cuda_u256 &b);
__host__ __device__ cuda_u256 cuda_u256_mul_mod_p(const cuda_u256 &a, const cuda_u256 &b);
__host__ __device__ cuda_u256 cuda_u256_square_mod_p(const cuda_u256 &a);
__host__ __device__ cuda_u256 cuda_u256_inv_mod_p(const cuda_u256 &a);
__host__ __device__ cuda_point_affine cuda_secp256k1_generator();
__host__ __device__ int cuda_point_on_curve(const cuda_point_affine &p);
__host__ __device__ cuda_point_affine cuda_point_add_affine(const cuda_point_affine &a,
                                                            const cuda_point_affine &b);
__host__ __device__ cuda_point_affine cuda_point_double_affine(const cuda_point_affine &p);

extern "C" int cuda_secp256k1_selftest(void);

#endif
