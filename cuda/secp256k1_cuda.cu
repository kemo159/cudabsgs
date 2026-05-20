#include "secp256k1_cuda.cuh"

#include <cuda_runtime.h>
#include <stdio.h>

__host__ __device__ cuda_u256 cuda_secp_p() {
  cuda_u256 p;
  p.v[0] = 0xFFFFFFFEFFFFFC2FULL;
  p.v[1] = 0xFFFFFFFFFFFFFFFFULL;
  p.v[2] = 0xFFFFFFFFFFFFFFFFULL;
  p.v[3] = 0xFFFFFFFFFFFFFFFFULL;
  return p;
}

__host__ __device__ int cuda_u256_cmp(const cuda_u256 &a, const cuda_u256 &b) {
  for (int i = 3; i >= 0; i--) {
    if (a.v[i] > b.v[i]) return 1;
    if (a.v[i] < b.v[i]) return -1;
  }
  return 0;
}

__host__ __device__ static cuda_u256 raw_add(const cuda_u256 &a, const cuda_u256 &b, uint64_t *carry_out) {
  cuda_u256 r;
  uint64_t carry = 0;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    uint64_t s = a.v[i] + b.v[i];
    uint64_t c1 = (s < a.v[i]);
    uint64_t s2 = s + carry;
    uint64_t c2 = (s2 < s);
    r.v[i] = s2;
    carry = c1 | c2;
  }
  *carry_out = carry;
  return r;
}

__host__ __device__ static cuda_u256 raw_sub(const cuda_u256 &a, const cuda_u256 &b, uint64_t *borrow_out) {
  cuda_u256 r;
  uint64_t borrow = 0;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    uint64_t bi = b.v[i] + borrow;
    uint64_t bcarry = (bi < b.v[i]);
    r.v[i] = a.v[i] - bi;
    borrow = (a.v[i] < bi) | bcarry;
  }
  *borrow_out = borrow;
  return r;
}

__host__ __device__ cuda_u256 cuda_u256_add_mod_p(const cuda_u256 &a, const cuda_u256 &b) {
  const cuda_u256 p = cuda_secp_p();
  uint64_t carry = 0;
  cuda_u256 r = raw_add(a, b, &carry);
  if (carry) {
    cuda_u256 fold = {};
    fold.v[0] = 0x1000003D1ULL;
    r = raw_add(r, fold, &carry);
  }
  while (carry || cuda_u256_cmp(r, p) >= 0) {
    uint64_t borrow = 0;
    r = raw_sub(r, p, &borrow);
    carry = 0;
  }
  return r;
}

__host__ __device__ cuda_u256 cuda_u256_sub_mod_p(const cuda_u256 &a, const cuda_u256 &b) {
  const cuda_u256 p = cuda_secp_p();
  uint64_t borrow = 0;
  cuda_u256 r = raw_sub(a, b, &borrow);
  if (borrow) {
    uint64_t carry = 0;
    r = raw_add(r, p, &carry);
  }
  return r;
}

__host__ __device__ static int cuda_u256_get_bit(const cuda_u256 &a, int bit) {
  return (int)((a.v[bit >> 6] >> (bit & 63)) & 1ULL);
}

__host__ __device__ static cuda_u256 cuda_u256_mul_mod_p_slow(const cuda_u256 &a, const cuda_u256 &b) {
  cuda_u256 acc = {};
  cuda_u256 addend = a;

  #pragma unroll 1
  for (int bit = 0; bit < 256; bit++) {
    if (cuda_u256_get_bit(b, bit)) {
      acc = cuda_u256_add_mod_p(acc, addend);
    }
    addend = cuda_u256_add_mod_p(addend, addend);
  }

  return acc;
}

__host__ __device__ static void u256_to_u32(const cuda_u256 &a, uint32_t out[8]) {
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    out[i * 2] = (uint32_t)(a.v[i] & 0xFFFFFFFFULL);
    out[i * 2 + 1] = (uint32_t)(a.v[i] >> 32);
  }
}

__host__ __device__ static cuda_u256 u32_to_u256(const uint32_t in[8]) {
  cuda_u256 r;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    r.v[i] = ((uint64_t)in[i * 2 + 1] << 32) | (uint64_t)in[i * 2];
  }
  return r;
}

__host__ __device__ static int cmp_u32_8(const uint32_t a[8], const uint32_t b[8]) {
  for (int i = 7; i >= 0; i--) {
    if (a[i] > b[i]) return 1;
    if (a[i] < b[i]) return -1;
  }
  return 0;
}

__host__ __device__ static void sub_u32_8(uint32_t a[8], const uint32_t b[8]) {
  uint64_t borrow = 0;
  #pragma unroll
  for (int i = 0; i < 8; i++) {
    uint64_t bi = (uint64_t)b[i] + borrow;
    uint64_t ai = (uint64_t)a[i];
    a[i] = (uint32_t)(ai - bi);
    borrow = (ai < bi);
  }
}

__host__ __device__ static void normalize_u64_12(uint64_t r[12]) {
  #pragma unroll
  for (int i = 0; i < 11; i++) {
    uint64_t carry = r[i] >> 32;
    r[i] &= 0xFFFFFFFFULL;
    r[i + 1] += carry;
  }
}

__host__ __device__ static void fold_high_u64_12(uint64_t r[12]) {
  #pragma unroll
  for (int k = 8; k < 12; k++) {
    uint64_t q = r[k];
    r[k] = 0;
    r[k - 8] += q * 977ULL;
    r[k - 7] += q;
  }
}

__host__ __device__ static cuda_u256 canonicalize_u32_8(uint32_t r[8]) {
  const uint32_t p[8] = {
      0xFFFFFC2FU, 0xFFFFFFFEU, 0xFFFFFFFFU, 0xFFFFFFFFU,
      0xFFFFFFFFU, 0xFFFFFFFFU, 0xFFFFFFFFU, 0xFFFFFFFFU};
  while (cmp_u32_8(r, p) >= 0) {
    sub_u32_8(r, p);
  }
  return u32_to_u256(r);
}

__host__ __device__ cuda_u256 cuda_u256_mul_mod_p(const cuda_u256 &a, const cuda_u256 &b) {
  uint32_t aa[8], bb[8], t[16];
  u256_to_u32(a, aa);
  u256_to_u32(b, bb);
  #pragma unroll
  for (int i = 0; i < 16; i++) {
    t[i] = 0;
  }

  #pragma unroll
  for (int i = 0; i < 8; i++) {
    uint64_t carry = 0;
    #pragma unroll
    for (int j = 0; j < 8; j++) {
      uint64_t acc = (uint64_t)t[i + j] + (uint64_t)aa[i] * (uint64_t)bb[j] + carry;
      t[i + j] = (uint32_t)acc;
      carry = acc >> 32;
    }
    int k = i + 8;
    while (carry != 0 && k < 16) {
      uint64_t acc = (uint64_t)t[k] + carry;
      t[k] = (uint32_t)acc;
      carry = acc >> 32;
      k++;
    }
  }

  uint64_t r[12];
  #pragma unroll
  for (int i = 0; i < 12; i++) {
    r[i] = 0;
  }
  #pragma unroll
  for (int i = 0; i < 8; i++) {
    r[i] += t[i];
    r[i] += (uint64_t)t[i + 8] * 977ULL;
    r[i + 1] += (uint64_t)t[i + 8];
  }

  #pragma unroll
  for (int pass = 0; pass < 8; pass++) {
    normalize_u64_12(r);
    fold_high_u64_12(r);
  }
  normalize_u64_12(r);
  fold_high_u64_12(r);
  normalize_u64_12(r);

  uint32_t out[8];
  #pragma unroll
  for (int i = 0; i < 8; i++) {
    out[i] = (uint32_t)r[i];
  }
  return canonicalize_u32_8(out);
}

__host__ __device__ cuda_u256 cuda_u256_square_mod_p(const cuda_u256 &a) {
  return cuda_u256_mul_mod_p(a, a);
}

__host__ __device__ static cuda_u256 cuda_u256_from_u64(uint64_t x) {
  cuda_u256 r = {};
  r.v[0] = x;
  return r;
}

__host__ __device__ static int cuda_u256_is_zero(const cuda_u256 &a) {
  return (a.v[0] | a.v[1] | a.v[2] | a.v[3]) == 0;
}

__host__ __device__ cuda_u256 cuda_u256_inv_mod_p(const cuda_u256 &a) {
  const cuda_u256 p = cuda_secp_p();
  cuda_u256 exponent = cuda_u256_sub_mod_p(p, cuda_u256_from_u64(2));
  cuda_u256 result = cuda_u256_from_u64(1);

  for (int bit = 255; bit >= 0; bit--) {
    result = cuda_u256_square_mod_p(result);
    if (cuda_u256_get_bit(exponent, bit)) {
      result = cuda_u256_mul_mod_p(result, a);
    }
  }
  return result;
}

__host__ __device__ cuda_point_affine cuda_secp256k1_generator() {
  cuda_point_affine g;
  g.x.v[0] = 0x59F2815B16F81798ULL;
  g.x.v[1] = 0x029BFCDB2DCE28D9ULL;
  g.x.v[2] = 0x55A06295CE870B07ULL;
  g.x.v[3] = 0x79BE667EF9DCBBACULL;
  g.y.v[0] = 0x9C47D08FFB10D4B8ULL;
  g.y.v[1] = 0xFD17B448A6855419ULL;
  g.y.v[2] = 0x5DA4FBFC0E1108A8ULL;
  g.y.v[3] = 0x483ADA7726A3C465ULL;
  g.infinity = 0;
  return g;
}

__host__ __device__ int cuda_point_on_curve(const cuda_point_affine &p) {
  if (p.infinity) return 1;
  cuda_u256 y2 = cuda_u256_square_mod_p(p.y);
  cuda_u256 x2 = cuda_u256_square_mod_p(p.x);
  cuda_u256 x3 = cuda_u256_mul_mod_p(x2, p.x);
  cuda_u256 rhs = cuda_u256_add_mod_p(x3, cuda_u256_from_u64(7));
  return cuda_u256_cmp(y2, rhs) == 0;
}

__host__ __device__ cuda_point_affine cuda_point_double_affine(const cuda_point_affine &p) {
  if (p.infinity || cuda_u256_is_zero(p.y)) {
    cuda_point_affine r = {};
    r.infinity = 1;
    return r;
  }
  cuda_u256 three = cuda_u256_from_u64(3);
  cuda_u256 two = cuda_u256_from_u64(2);
  cuda_u256 s_num = cuda_u256_mul_mod_p(three, cuda_u256_square_mod_p(p.x));
  cuda_u256 s_den = cuda_u256_inv_mod_p(cuda_u256_mul_mod_p(two, p.y));
  cuda_u256 s = cuda_u256_mul_mod_p(s_num, s_den);
  cuda_u256 rx = cuda_u256_sub_mod_p(cuda_u256_square_mod_p(s),
                                     cuda_u256_add_mod_p(p.x, p.x));
  cuda_u256 ry = cuda_u256_sub_mod_p(cuda_u256_mul_mod_p(s, cuda_u256_sub_mod_p(p.x, rx)), p.y);
  cuda_point_affine r;
  r.x = rx;
  r.y = ry;
  r.infinity = 0;
  return r;
}

__host__ __device__ cuda_point_affine cuda_point_add_affine(const cuda_point_affine &a,
                                                            const cuda_point_affine &b) {
  if (a.infinity) return b;
  if (b.infinity) return a;
  if (cuda_u256_cmp(a.x, b.x) == 0) {
    if (cuda_u256_cmp(a.y, b.y) == 0) {
      return cuda_point_double_affine(a);
    }
    cuda_point_affine r = {};
    r.infinity = 1;
    return r;
  }

  cuda_u256 s_num = cuda_u256_sub_mod_p(b.y, a.y);
  cuda_u256 s_den = cuda_u256_inv_mod_p(cuda_u256_sub_mod_p(b.x, a.x));
  cuda_u256 s = cuda_u256_mul_mod_p(s_num, s_den);
  cuda_u256 rx = cuda_u256_sub_mod_p(cuda_u256_sub_mod_p(cuda_u256_square_mod_p(s), a.x), b.x);
  cuda_u256 ry = cuda_u256_sub_mod_p(cuda_u256_mul_mod_p(s, cuda_u256_sub_mod_p(a.x, rx)), a.y);
  cuda_point_affine r;
  r.x = rx;
  r.y = ry;
  r.infinity = 0;
  return r;
}

__host__ __device__ static cuda_u256 selftest_value(uint32_t seed) {
  cuda_u256 x;
  uint64_t s = (uint64_t)seed * 0x9E3779B97F4A7C15ULL + 0xD1B54A32D192ED03ULL;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    s ^= s >> 30;
    s *= 0xBF58476D1CE4E5B9ULL;
    s ^= s >> 27;
    s *= 0x94D049BB133111EBULL;
    s ^= s >> 31;
    x.v[i] = s;
  }
  const cuda_u256 p = cuda_secp_p();
  while (cuda_u256_cmp(x, p) >= 0) {
    uint64_t borrow = 0;
    x = raw_sub(x, p, &borrow);
  }
  return x;
}

__global__ static void secp_field_selftest_kernel(int *ok) {
  const cuda_u256 p = cuda_secp_p();
  cuda_u256 one = {};
  one.v[0] = 1;
  cuda_u256 two = {};
  two.v[0] = 2;
  cuda_u256 three = {};
  three.v[0] = 3;

  cuda_u256 p_minus_one = cuda_u256_sub_mod_p(p, one);
  cuda_u256 zero = cuda_u256_add_mod_p(p_minus_one, one);
  cuda_u256 wrapped = cuda_u256_add_mod_p(p_minus_one, two);
  cuda_u256 back = cuda_u256_sub_mod_p(one, two);
  cuda_u256 six = cuda_u256_mul_mod_p(two, three);
  cuda_u256 nine = cuda_u256_square_mod_p(three);
  cuda_u256 neg_one_squared = cuda_u256_square_mod_p(p_minus_one);
  cuda_u256 wrap_mul = cuda_u256_mul_mod_p(p_minus_one, two);
  cuda_point_affine g = cuda_secp256k1_generator();
  cuda_point_affine two_g_a = cuda_point_double_affine(g);
  cuda_point_affine two_g_b = cuda_point_add_affine(g, g);
  cuda_point_affine three_g = cuda_point_add_affine(two_g_a, g);

  int local_ok = 1;
  int fail_code = 0;
  if (!(zero.v[0] == 0 && zero.v[1] == 0 && zero.v[2] == 0 && zero.v[3] == 0)) fail_code = 10;
  if (!fail_code && !(wrapped.v[0] == 1 && wrapped.v[1] == 0 && wrapped.v[2] == 0 && wrapped.v[3] == 0)) fail_code = 11;
  if (!fail_code && !(cuda_u256_cmp(back, p_minus_one) == 0)) fail_code = 12;
  if (!fail_code && !(six.v[0] == 6 && six.v[1] == 0 && six.v[2] == 0 && six.v[3] == 0)) fail_code = 13;
  if (!fail_code && !(nine.v[0] == 9 && nine.v[1] == 0 && nine.v[2] == 0 && nine.v[3] == 0)) fail_code = 14;
  if (!fail_code && !(neg_one_squared.v[0] == 1 && neg_one_squared.v[1] == 0 &&
               neg_one_squared.v[2] == 0 && neg_one_squared.v[3] == 0)) {
    fail_code = 15;
  }
  if (!fail_code && !(cuda_u256_cmp(wrap_mul, cuda_u256_sub_mod_p(p, two)) == 0)) fail_code = 16;
  if (!fail_code && !cuda_point_on_curve(g)) fail_code = 20;
  if (!fail_code && !cuda_point_on_curve(two_g_a)) fail_code = 21;
  if (!fail_code && !cuda_point_on_curve(three_g)) fail_code = 22;
  if (!fail_code && !(cuda_u256_cmp(two_g_a.x, two_g_b.x) == 0 &&
                      cuda_u256_cmp(two_g_a.y, two_g_b.y) == 0 &&
                      two_g_a.infinity == two_g_b.infinity)) fail_code = 23;
  local_ok &= (fail_code == 0);
  #pragma unroll 1
  for (int i = 0; i < 64; i++) {
    cuda_u256 a = selftest_value((uint32_t)(0x12340000U + i * 17U));
    cuda_u256 b = selftest_value((uint32_t)(0xABC00000U + i * 29U));
    cuda_u256 fast = cuda_u256_mul_mod_p(a, b);
    cuda_u256 slow = cuda_u256_mul_mod_p_slow(a, b);
    cuda_u256 sq_fast = cuda_u256_square_mod_p(a);
    cuda_u256 sq_slow = cuda_u256_mul_mod_p_slow(a, a);
    if (!fail_code && !(cuda_u256_cmp(fast, slow) == 0)) fail_code = 1000 + i;
    if (!fail_code && !(cuda_u256_cmp(sq_fast, sq_slow) == 0)) fail_code = 2000 + i;
    if (!fail_code && !(cuda_u256_cmp(fast, p) < 0)) fail_code = 3000 + i;
    if (!fail_code && !(cuda_u256_cmp(sq_fast, p) < 0)) fail_code = 4000 + i;
  }
  *ok = fail_code ? -fail_code : local_ok;
}

extern "C" int cuda_secp256k1_selftest(void) {
  int *d_ok = NULL;
  int h_ok = 0;
  cudaError_t err = cudaMalloc((void **)&d_ok, sizeof(int));
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
    return 0;
  }
  err = cudaMemset(d_ok, 0, sizeof(int));
  if (err == cudaSuccess) {
    secp_field_selftest_kernel<<<1, 1>>>(d_ok);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) {
    err = cudaDeviceSynchronize();
  }
  if (err == cudaSuccess) {
    err = cudaMemcpy(&h_ok, d_ok, sizeof(int), cudaMemcpyDeviceToHost);
  }
  cudaFree(d_ok);
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA secp256k1 selftest failed: %s\n", cudaGetErrorString(err));
    return 0;
  }
  if (h_ok != 1) {
    fprintf(stderr, "CUDA secp256k1 selftest code: %d\n", h_ok);
  }
  return h_ok == 1;
}
