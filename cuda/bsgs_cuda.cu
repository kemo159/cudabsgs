#include "bsgs_cuda.h"
#include "secp256k1_cuda.cuh"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CUDA_BSGS_SHARDS 256
#define CUDA_BSGS_GSN_COUNT 512
#define CUDA_BSGS_CENTER_POINTS 1024
#define CUDA_BSGS_CENTER_INVERSES 512
#define CUDA_BSGS_CENTER_SCAN 512
#define CUDA_BSGS_CENTER_SEQUENCE_MAX 512
#define CUDA_BSGS_CENTER_OFFSET_COUNT (CUDA_BSGS_CENTER_SEQUENCE_MAX + 1)
#define CUDA_BSGS_CACHE_MAGIC 0x484742535543474BULL
#define CUDA_BSGS_CACHE_VERSION 1U

typedef struct cuda_bsgs_cache_header {
  uint64_t magic;
  uint32_t version;
  uint32_t shard_count;
  uint64_t table_count;
} cuda_bsgs_cache_header;

typedef struct cuda_bsgs_cache_bloom_meta {
  uint64_t bits;
  uint64_t bytes;
  uint8_t hashes;
  uint8_t reserved[7];
} cuda_bsgs_cache_bloom_meta;

typedef struct cuda_bloom_device_view {
  uint64_t bits;
  uint64_t bytes;
  uint8_t hashes;
  uint8_t *bf;
} cuda_bloom_device_view;

struct cuda_bsgs_context {
  cuda_bloom_device_view bloom1[CUDA_BSGS_SHARDS];
  cuda_bloom_device_view bloom2[CUDA_BSGS_SHARDS];
  cuda_bloom_device_view bloom3[CUDA_BSGS_SHARDS];
  cuda_bloom_device_view *d_bloom1;
  cuda_bloom_device_view *d_bloom2;
  cuda_bloom_device_view *d_bloom3;
  uint8_t *d_bloom1_data;
  uint8_t *d_bloom2_data;
  uint8_t *d_bloom3_data;
  size_t bloom_shard_count;
  cuda_bsgs_xvalue *table;
  uint8_t *table_values_soa;
  uint64_t *table_indices;
  uint64_t table_count;
  cuda_bsgs_affine_point *d_gsn;
  cuda_bsgs_affine_point *d_2gsn;
  cuda_bsgs_affine_point *d_original_points;
  uint32_t original_point_count;
  cuda_bsgs_affine_point *d_amp2;
  cuda_bsgs_affine_point *d_amp3;
  cuda_bsgs_scalar h_m_double;
  cuda_bsgs_scalar h_m2_double;
  cuda_bsgs_scalar h_m3;
  cuda_bsgs_scalar h_m3_double;
  cuda_bsgs_scalar h_order;
  cuda_bsgs_scalar *d_m_double;
  cuda_bsgs_scalar *d_m2_double;
  cuda_bsgs_scalar *d_m3;
  cuda_bsgs_scalar *d_m3_double;
  cuda_bsgs_scalar *d_order;
  cuda_bsgs_affine_point *d_center_offsets;
  cuda_bsgs_affine_point *d_center_next_seed;
  int search_points_uploaded;
  cuda_bsgs_center_work *d_center_works;
  uint32_t center_work_capacity;
  cuda_bsgs_center_hit *d_center_hits;
  uint32_t center_hit_capacity;
  uint32_t *d_center_hit_count;
  cuda_bsgs_center_hit *d_filter_input_hits;
  uint32_t filter_input_capacity;
  cuda_bsgs_filter_hit *d_filter_hits;
  uint32_t filter_hit_capacity;
  uint32_t *d_filter_hit_count;
  uint8_t *d_probe_xpoints;
  uint32_t probe_xpoint_capacity;
  cuda_bsgs_hit *d_probe_hits;
  uint32_t probe_hit_capacity;
  uint32_t *d_probe_hit_count;
  char last_error[256];
};

struct cuda_bsgs_probe_workspace {
  cudaStream_t stream;
  uint8_t *d_xpoints;
  uint32_t xpoint_capacity;
  cuda_bsgs_hit *d_hits;
  uint32_t hit_capacity;
  uint32_t *d_hit_count;
  uint32_t h_hit_count;
  int pending;
};

struct cuda_bsgs_center_workspace {
  cudaStream_t stream;
  cuda_bsgs_center_work *d_works;
  uint32_t work_capacity;
  cuda_bsgs_center_hit *d_hits;
  uint32_t hit_capacity;
  uint32_t *d_hit_count;
  cuda_bsgs_affine_point *d_next_seed;
};

static void set_error(cuda_bsgs_context *ctx, const char *msg, cudaError_t err) {
  if (ctx == NULL) {
    return;
  }
  if (err == cudaSuccess) {
    snprintf(ctx->last_error, sizeof(ctx->last_error), "%s", msg);
  } else {
    snprintf(ctx->last_error, sizeof(ctx->last_error), "%s: %s", msg, cudaGetErrorString(err));
  }
}

static int check_cuda(cuda_bsgs_context *ctx, const char *msg, cudaError_t err) {
  if (err != cudaSuccess) {
    set_error(ctx, msg, err);
    return 0;
  }
  return 1;
}

static int upload_one_bloom_set(cuda_bsgs_context *ctx,
                                cuda_bloom_device_view *dst,
                                const struct bloom *src,
                                uint8_t **data_out,
                                size_t shard_count) {
  uint64_t total_bytes = 0;
  for (size_t i = 0; i < shard_count; i++) {
    if (src[i].ready == 0 || src[i].bf == NULL || src[i].bytes == 0) {
      set_error(ctx, "invalid bloom shard", cudaSuccess);
      return 0;
    }
    total_bytes += src[i].bytes;
  }

  if (*data_out != NULL) {
    cudaFree(*data_out);
    *data_out = NULL;
  }
  if (!check_cuda(ctx, "cudaMalloc bloom data", cudaMalloc((void **)data_out, (size_t)total_bytes))) {
    return 0;
  }

  uint64_t offset = 0;
  for (size_t i = 0; i < shard_count; i++) {
    dst[i].bits = src[i].bits;
    dst[i].bytes = src[i].bytes;
    dst[i].hashes = src[i].hashes;
    dst[i].bf = *data_out + offset;

    if (!check_cuda(ctx, "cudaMemcpy bloom shard",
                    cudaMemcpy(dst[i].bf, src[i].bf, src[i].bytes, cudaMemcpyHostToDevice))) {
      return 0;
    }
    offset += src[i].bytes;
  }
  return 1;
}

static void free_one_bloom_set(cuda_bloom_device_view *views) {
  for (int i = 0; i < CUDA_BSGS_SHARDS; i++) {
    views[i].bf = NULL;
  }
}

static int upload_bloom_views(cuda_bsgs_context *ctx,
                              cuda_bloom_device_view **dst,
                              const cuda_bloom_device_view *src,
                              size_t shard_count,
                              const char *label) {
  if (*dst != NULL) {
    cudaFree(*dst);
    *dst = NULL;
  }
  const size_t bytes = shard_count * sizeof(cuda_bloom_device_view);
  if (!check_cuda(ctx, label, cudaMalloc((void **)dst, bytes))) {
    return 0;
  }
  return check_cuda(ctx, label, cudaMemcpy(*dst, src, bytes, cudaMemcpyHostToDevice));
}

static int write_exact(FILE *file, const void *data, size_t bytes) {
  return bytes == 0 || fwrite(data, bytes, 1, file) == 1;
}

static int read_exact(FILE *file, void *data, size_t bytes) {
  return bytes == 0 || fread(data, bytes, 1, file) == 1;
}

static int save_one_native_bloom_set(cuda_bsgs_context *ctx,
                                     FILE *file,
                                     const cuda_bloom_device_view *views,
                                     const uint8_t *device_data,
                                     size_t shard_count) {
  uint64_t total_bytes = 0;
  for (size_t i = 0; i < shard_count; i++) {
    cuda_bsgs_cache_bloom_meta meta = {};
    meta.bits = views[i].bits;
    meta.bytes = views[i].bytes;
    meta.hashes = views[i].hashes;
    if (!write_exact(file, &meta, sizeof(meta))) {
      set_error(ctx, "write native bloom metadata", cudaSuccess);
      return 0;
    }
    total_bytes += views[i].bytes;
  }

  uint8_t *host_data = (uint8_t *)malloc((size_t)total_bytes);
  if (host_data == NULL) {
    set_error(ctx, "malloc native bloom staging", cudaSuccess);
    return 0;
  }
  if (!check_cuda(ctx, "copy native bloom from device",
                  cudaMemcpy(host_data, device_data, (size_t)total_bytes, cudaMemcpyDeviceToHost))) {
    free(host_data);
    return 0;
  }
  const int ok = write_exact(file, host_data, (size_t)total_bytes);
  free(host_data);
  if (!ok) {
    set_error(ctx, "write native bloom data", cudaSuccess);
    return 0;
  }
  return 1;
}

static int load_one_native_bloom_set(cuda_bsgs_context *ctx,
                                     FILE *file,
                                     cuda_bloom_device_view *views,
                                     cuda_bloom_device_view **device_views,
                                     uint8_t **device_data,
                                     size_t shard_count,
                                     const char *label) {
  uint64_t total_bytes = 0;
  for (size_t i = 0; i < shard_count; i++) {
    cuda_bsgs_cache_bloom_meta meta = {};
    if (!read_exact(file, &meta, sizeof(meta)) || meta.bytes == 0 || meta.hashes == 0) {
      set_error(ctx, "read native bloom metadata", cudaSuccess);
      return 0;
    }
    views[i].bits = meta.bits;
    views[i].bytes = meta.bytes;
    views[i].hashes = meta.hashes;
    total_bytes += meta.bytes;
  }

  uint8_t *host_data = (uint8_t *)malloc((size_t)total_bytes);
  if (host_data == NULL) {
    set_error(ctx, "malloc native bloom load staging", cudaSuccess);
    return 0;
  }
  if (!read_exact(file, host_data, (size_t)total_bytes)) {
    free(host_data);
    set_error(ctx, "read native bloom data", cudaSuccess);
    return 0;
  }

  if (*device_data != NULL) {
    cudaFree(*device_data);
    *device_data = NULL;
  }
  if (!check_cuda(ctx, "cudaMalloc native bloom data",
                  cudaMalloc((void **)device_data, (size_t)total_bytes)) ||
      !check_cuda(ctx, "cudaMemcpy native bloom data",
                  cudaMemcpy(*device_data, host_data, (size_t)total_bytes, cudaMemcpyHostToDevice))) {
    free(host_data);
    return 0;
  }
  free(host_data);

  uint64_t offset = 0;
  for (size_t i = 0; i < shard_count; i++) {
    views[i].bf = *device_data + offset;
    offset += views[i].bytes;
  }
  return upload_bloom_views(ctx, device_views, views, shard_count, label);
}

static int ensure_center_buffers(cuda_bsgs_context *ctx, uint32_t work_count, uint32_t max_hits) {
  if (ctx->center_work_capacity < work_count) {
    cuda_bsgs_center_work *new_works = NULL;
    const size_t work_bytes = (size_t)work_count * sizeof(cuda_bsgs_center_work);
    if (!check_cuda(ctx, "cudaMalloc center works", cudaMalloc((void **)&new_works, work_bytes))) {
      return 0;
    }
    if (ctx->d_center_works != NULL) {
      cudaFree(ctx->d_center_works);
    }
    ctx->d_center_works = new_works;
    ctx->center_work_capacity = work_count;
  }

  if (ctx->center_hit_capacity < max_hits) {
    cuda_bsgs_center_hit *new_hits = NULL;
    const size_t hit_bytes = (size_t)max_hits * sizeof(cuda_bsgs_center_hit);
    if (max_hits > 0 &&
        !check_cuda(ctx, "cudaMalloc center hits", cudaMalloc((void **)&new_hits, hit_bytes))) {
      return 0;
    }
    if (ctx->d_center_hits != NULL) {
      cudaFree(ctx->d_center_hits);
    }
    ctx->d_center_hits = new_hits;
    ctx->center_hit_capacity = max_hits;
  }

  if (ctx->d_center_hit_count == NULL &&
      !check_cuda(ctx, "cudaMalloc center hit count",
                  cudaMalloc((void **)&ctx->d_center_hit_count, sizeof(uint32_t)))) {
    return 0;
  }
  return 1;
}

static int ensure_filter_buffers(cuda_bsgs_context *ctx, uint32_t input_count, uint32_t max_hits) {
  if (ctx->filter_input_capacity < input_count) {
    cuda_bsgs_center_hit *new_hits = NULL;
    const size_t bytes = (size_t)input_count * sizeof(cuda_bsgs_center_hit);
    if (!check_cuda(ctx, "cudaMalloc filter input hits",
                    cudaMalloc((void **)&new_hits, bytes))) {
      return 0;
    }
    if (ctx->d_filter_input_hits != NULL) {
      cudaFree(ctx->d_filter_input_hits);
    }
    ctx->d_filter_input_hits = new_hits;
    ctx->filter_input_capacity = input_count;
  }
  if (ctx->filter_hit_capacity < max_hits) {
    cuda_bsgs_filter_hit *new_hits = NULL;
    const size_t bytes = (size_t)max_hits * sizeof(cuda_bsgs_filter_hit);
    if (max_hits > 0 &&
        !check_cuda(ctx, "cudaMalloc filter output hits",
                    cudaMalloc((void **)&new_hits, bytes))) {
      return 0;
    }
    if (ctx->d_filter_hits != NULL) {
      cudaFree(ctx->d_filter_hits);
    }
    ctx->d_filter_hits = new_hits;
    ctx->filter_hit_capacity = max_hits;
  }
  if (ctx->d_filter_hit_count == NULL &&
      !check_cuda(ctx, "cudaMalloc filter hit count",
                  cudaMalloc((void **)&ctx->d_filter_hit_count, sizeof(uint32_t)))) {
    return 0;
  }
  return 1;
}

static int ensure_probe_buffers(cuda_bsgs_context *ctx, uint32_t xpoint_count, uint32_t max_hits) {
  if (ctx->probe_xpoint_capacity < xpoint_count) {
    uint8_t *new_xpoints = NULL;
    const size_t bytes = (size_t)xpoint_count * 32;
    if (!check_cuda(ctx, "cudaMalloc probe xpoints",
                    cudaMalloc((void **)&new_xpoints, bytes))) {
      return 0;
    }
    if (ctx->d_probe_xpoints != NULL) {
      cudaFree(ctx->d_probe_xpoints);
    }
    ctx->d_probe_xpoints = new_xpoints;
    ctx->probe_xpoint_capacity = xpoint_count;
  }
  if (ctx->probe_hit_capacity < max_hits) {
    cuda_bsgs_hit *new_hits = NULL;
    const size_t bytes = (size_t)max_hits * sizeof(cuda_bsgs_hit);
    if (max_hits > 0 &&
        !check_cuda(ctx, "cudaMalloc probe hits",
                    cudaMalloc((void **)&new_hits, bytes))) {
      return 0;
    }
    if (ctx->d_probe_hits != NULL) {
      cudaFree(ctx->d_probe_hits);
    }
    ctx->d_probe_hits = new_hits;
    ctx->probe_hit_capacity = max_hits;
  }
  if (ctx->d_probe_hit_count == NULL &&
      !check_cuda(ctx, "cudaMalloc probe hit count",
                  cudaMalloc((void **)&ctx->d_probe_hit_count, sizeof(uint32_t)))) {
    return 0;
  }
  return 1;
}

static int ensure_probe_workspace_buffers(cuda_bsgs_context *ctx,
                                          cuda_bsgs_probe_workspace *workspace,
                                          uint32_t xpoint_count,
                                          uint32_t max_hits) {
  if (workspace->xpoint_capacity < xpoint_count) {
    uint8_t *new_xpoints = NULL;
    const size_t bytes = (size_t)xpoint_count * 32;
    if (!check_cuda(ctx, "cudaMalloc workspace probe xpoints",
                    cudaMalloc((void **)&new_xpoints, bytes))) {
      return 0;
    }
    if (workspace->d_xpoints != NULL) {
      cudaFree(workspace->d_xpoints);
    }
    workspace->d_xpoints = new_xpoints;
    workspace->xpoint_capacity = xpoint_count;
  }
  if (workspace->hit_capacity < max_hits) {
    cuda_bsgs_hit *new_hits = NULL;
    const size_t bytes = (size_t)max_hits * sizeof(cuda_bsgs_hit);
    if (max_hits > 0 &&
        !check_cuda(ctx, "cudaMalloc workspace probe hits",
                    cudaMalloc((void **)&new_hits, bytes))) {
      return 0;
    }
    if (workspace->d_hits != NULL) {
      cudaFree(workspace->d_hits);
    }
    workspace->d_hits = new_hits;
    workspace->hit_capacity = max_hits;
  }
  if (workspace->d_hit_count == NULL &&
      !check_cuda(ctx, "cudaMalloc workspace probe hit count",
                  cudaMalloc((void **)&workspace->d_hit_count, sizeof(uint32_t)))) {
    return 0;
  }
  return 1;
}

static int ensure_center_workspace_buffers(cuda_bsgs_context *ctx,
                                           cuda_bsgs_center_workspace *workspace,
                                           uint32_t work_count,
                                           uint32_t max_hits) {
  if (workspace->work_capacity < work_count) {
    cuda_bsgs_center_work *new_works = NULL;
    const size_t bytes = (size_t)work_count * sizeof(cuda_bsgs_center_work);
    if (!check_cuda(ctx, "cudaMalloc center workspace works",
                    cudaMalloc((void **)&new_works, bytes))) {
      return 0;
    }
    if (workspace->d_works != NULL) cudaFree(workspace->d_works);
    workspace->d_works = new_works;
    workspace->work_capacity = work_count;
  }
  if (workspace->hit_capacity < max_hits) {
    cuda_bsgs_center_hit *new_hits = NULL;
    const size_t bytes = (size_t)max_hits * sizeof(cuda_bsgs_center_hit);
    if (max_hits > 0 &&
        !check_cuda(ctx, "cudaMalloc center workspace hits",
                    cudaMalloc((void **)&new_hits, bytes))) {
      return 0;
    }
    if (workspace->d_hits != NULL) cudaFree(workspace->d_hits);
    workspace->d_hits = new_hits;
    workspace->hit_capacity = max_hits;
  }
  if (workspace->d_hit_count == NULL &&
      !check_cuda(ctx, "cudaMalloc center workspace hit count",
                  cudaMalloc((void **)&workspace->d_hit_count, sizeof(uint32_t)))) {
    return 0;
  }
  if (workspace->d_next_seed == NULL &&
      !check_cuda(ctx, "cudaMalloc center workspace next seed",
                  cudaMalloc((void **)&workspace->d_next_seed, sizeof(cuda_bsgs_affine_point)))) {
    return 0;
  }
  return 1;
}

__device__ __forceinline__ uint64_t rotl64(uint64_t x, int r) {
  return (x << r) | (x >> (64 - r));
}

__device__ __forceinline__ uint64_t read64le(const uint8_t *p) {
  uint64_t v;
  memcpy(&v, p, sizeof(v));
  return v;
}

__device__ __forceinline__ uint64_t xxh64_round(uint64_t acc, uint64_t input) {
  const uint64_t PRIME64_2 = 14029467366897019727ULL;
  const uint64_t PRIME64_1 = 11400714785074694791ULL;
  acc += input * PRIME64_2;
  acc = rotl64(acc, 31);
  acc *= PRIME64_1;
  return acc;
}

__device__ __forceinline__ uint64_t xxh64_merge_round(uint64_t acc, uint64_t val) {
  const uint64_t PRIME64_1 = 11400714785074694791ULL;
  const uint64_t PRIME64_4 = 9650029242287828579ULL;
  val = xxh64_round(0, val);
  acc ^= val;
  acc = acc * PRIME64_1 + PRIME64_4;
  return acc;
}

__device__ uint64_t xxh64_32(const uint8_t *input, uint64_t seed) {
  const uint64_t PRIME64_1 = 11400714785074694791ULL;
  const uint64_t PRIME64_2 = 14029467366897019727ULL;
  const uint64_t PRIME64_3 = 1609587929392839161ULL;
  uint64_t v1 = seed + PRIME64_1 + PRIME64_2;
  uint64_t v2 = seed + PRIME64_2;
  uint64_t v3 = seed + 0;
  uint64_t v4 = seed - PRIME64_1;

  v1 = xxh64_round(v1, read64le(input + 0));
  v2 = xxh64_round(v2, read64le(input + 8));
  v3 = xxh64_round(v3, read64le(input + 16));
  v4 = xxh64_round(v4, read64le(input + 24));

  uint64_t h64 = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);
  h64 = xxh64_merge_round(h64, v1);
  h64 = xxh64_merge_round(h64, v2);
  h64 = xxh64_merge_round(h64, v3);
  h64 = xxh64_merge_round(h64, v4);

  h64 += 32;
  h64 ^= h64 >> 33;
  h64 *= PRIME64_2;
  h64 ^= h64 >> 29;
  h64 *= PRIME64_3;
  h64 ^= h64 >> 32;
  return h64;
}

__device__ __forceinline__ int cuda_bloom_check32(const cuda_bloom_device_view *bloom,
                                                  const uint8_t *xpoint) {
  const uint64_t a = xxh64_32(xpoint, 0x59f2815b16f81798ULL);
  const uint64_t b = xxh64_32(xpoint, a);
  for (uint8_t i = 0; i < bloom->hashes; i++) {
    const uint64_t bit = (a + b * i) % bloom->bits;
    const uint8_t c = bloom->bf[bit >> 3];
    const uint8_t mask = (uint8_t)(1u << (bit & 7));
    if ((c & mask) == 0) {
      return 0;
    }
  }
  return 1;
}

__device__ __forceinline__ int cmp6(const uint8_t *a, const uint8_t *b) {
  #pragma unroll
  for (int i = 0; i < 6; i++) {
    if (a[i] < b[i]) return -1;
    if (a[i] > b[i]) return 1;
  }
  return 0;
}

__device__ int cuda_table_search6(const cuda_bsgs_xvalue *table,
                                  uint64_t table_count,
                                  const uint8_t *xpoint,
                                  uint64_t *index_out) {
  uint64_t low = 0;
  uint64_t high = table_count;
  const uint8_t *needle = xpoint + 16;
  while (low < high) {
    const uint64_t mid = low + ((high - low) >> 1);
    const int c = cmp6(needle, table[mid].value);
    if (c == 0) {
      *index_out = table[mid].index;
      return 1;
    }
    if (c < 0) {
      high = mid;
    } else {
      low = mid + 1;
    }
  }
  return 0;
}

__device__ __forceinline__ int cmp6_soa(const uint8_t *values_soa,
                                        uint64_t table_count,
                                        uint64_t index,
                                        const uint8_t *needle) {
  #pragma unroll
  for (int i = 0; i < 6; i++) {
    const uint8_t v = values_soa[(uint64_t)i * table_count + index];
    if (needle[i] < v) return -1;
    if (needle[i] > v) return 1;
  }
  return 0;
}

__device__ int cuda_table_search6_soa(const uint8_t *values_soa,
                                      const uint64_t *indices,
                                      uint64_t table_count,
                                      const uint8_t *xpoint,
                                      uint64_t *index_out) {
  uint64_t low = 0;
  uint64_t high = table_count;
  const uint8_t *needle = xpoint + 16;
  while (low < high) {
    const uint64_t mid = low + ((high - low) >> 1);
    const int c = cmp6_soa(values_soa, table_count, mid, needle);
    if (c == 0) {
      *index_out = indices[mid];
      return 1;
    }
    if (c < 0) {
      high = mid;
    } else {
      low = mid + 1;
    }
  }
  return 0;
}

__device__ __forceinline__ cuda_bsgs_u256 center_p() {
  cuda_bsgs_u256 p;
  p.v[0] = 0xFFFFFFFEFFFFFC2FULL;
  p.v[1] = 0xFFFFFFFFFFFFFFFFULL;
  p.v[2] = 0xFFFFFFFFFFFFFFFFULL;
  p.v[3] = 0xFFFFFFFFFFFFFFFFULL;
  return p;
}

__device__ __forceinline__ int center_u256_cmp(const cuda_bsgs_u256 &a, const cuda_bsgs_u256 &b) {
  for (int i = 3; i >= 0; i--) {
    if (a.v[i] > b.v[i]) return 1;
    if (a.v[i] < b.v[i]) return -1;
  }
  return 0;
}

__device__ __forceinline__ cuda_bsgs_u256 center_raw_add(const cuda_bsgs_u256 &a,
                                                         const cuda_bsgs_u256 &b,
                                                         uint64_t *carry_out) {
  cuda_bsgs_u256 r;
  uint64_t carry = 0;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    const uint64_t s = a.v[i] + b.v[i];
    const uint64_t c1 = (s < a.v[i]);
    const uint64_t s2 = s + carry;
    const uint64_t c2 = (s2 < s);
    r.v[i] = s2;
    carry = c1 | c2;
  }
  *carry_out = carry;
  return r;
}

__device__ __forceinline__ cuda_bsgs_u256 center_raw_sub(const cuda_bsgs_u256 &a,
                                                         const cuda_bsgs_u256 &b,
                                                         uint64_t *borrow_out) {
  cuda_bsgs_u256 r;
  uint64_t borrow = 0;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    const uint64_t bi = b.v[i] + borrow;
    const uint64_t bcarry = (bi < b.v[i]);
    r.v[i] = a.v[i] - bi;
    borrow = (a.v[i] < bi) | bcarry;
  }
  *borrow_out = borrow;
  return r;
}

__device__ __forceinline__ cuda_bsgs_u256 center_add_mod_p(const cuda_bsgs_u256 &a,
                                                           const cuda_bsgs_u256 &b) {
  const cuda_bsgs_u256 p = center_p();
  uint64_t carry = 0;
  cuda_bsgs_u256 r = center_raw_add(a, b, &carry);
  if (carry) {
    cuda_bsgs_u256 fold = {};
    fold.v[0] = 0x1000003D1ULL;
    r = center_raw_add(r, fold, &carry);
  }
  while (carry || center_u256_cmp(r, p) >= 0) {
    uint64_t borrow = 0;
    r = center_raw_sub(r, p, &borrow);
    carry = 0;
  }
  return r;
}

__device__ __forceinline__ cuda_bsgs_u256 center_sub_mod_p(const cuda_bsgs_u256 &a,
                                                           const cuda_bsgs_u256 &b) {
  const cuda_bsgs_u256 p = center_p();
  uint64_t borrow = 0;
  cuda_bsgs_u256 r = center_raw_sub(a, b, &borrow);
  if (borrow) {
    uint64_t carry = 0;
    r = center_raw_add(r, p, &carry);
  }
  return r;
}

__device__ __forceinline__ int center_u256_get_bit(const cuda_bsgs_u256 &a, int bit) {
  return (int)((a.v[bit >> 6] >> (bit & 63)) & 1ULL);
}

__device__ __forceinline__ void center_u256_to_u32(const cuda_bsgs_u256 &a, uint32_t out[8]) {
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    out[i * 2] = (uint32_t)(a.v[i] & 0xFFFFFFFFULL);
    out[i * 2 + 1] = (uint32_t)(a.v[i] >> 32);
  }
}

__device__ __forceinline__ cuda_bsgs_u256 center_u32_to_u256(const uint32_t in[8]) {
  cuda_bsgs_u256 r;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    r.v[i] = ((uint64_t)in[i * 2 + 1] << 32) | (uint64_t)in[i * 2];
  }
  return r;
}

__device__ __forceinline__ int center_cmp_u32_8(const uint32_t a[8], const uint32_t b[8]) {
  for (int i = 7; i >= 0; i--) {
    if (a[i] > b[i]) return 1;
    if (a[i] < b[i]) return -1;
  }
  return 0;
}

__device__ __forceinline__ void center_sub_u32_8(uint32_t a[8], const uint32_t b[8]) {
  uint64_t borrow = 0;
  #pragma unroll
  for (int i = 0; i < 8; i++) {
    const uint64_t bi = (uint64_t)b[i] + borrow;
    const uint64_t ai = (uint64_t)a[i];
    a[i] = (uint32_t)(ai - bi);
    borrow = (ai < bi);
  }
}

__device__ __forceinline__ void center_normalize_u64_12(uint64_t r[12]) {
  #pragma unroll
  for (int i = 0; i < 11; i++) {
    const uint64_t carry = r[i] >> 32;
    r[i] &= 0xFFFFFFFFULL;
    r[i + 1] += carry;
  }
}

__device__ __forceinline__ void center_fold_high_u64_12(uint64_t r[12]) {
  #pragma unroll
  for (int k = 8; k < 12; k++) {
    const uint64_t q = r[k];
    r[k] = 0;
    r[k - 8] += q * 977ULL;
    r[k - 7] += q;
  }
}

__device__ __forceinline__ cuda_bsgs_u256 center_canonicalize_u32_8(uint32_t r[8]) {
  const uint32_t p[8] = {
      0xFFFFFC2FU, 0xFFFFFFFEU, 0xFFFFFFFFU, 0xFFFFFFFFU,
      0xFFFFFFFFU, 0xFFFFFFFFU, 0xFFFFFFFFU, 0xFFFFFFFFU};
  while (center_cmp_u32_8(r, p) >= 0) {
    center_sub_u32_8(r, p);
  }
  return center_u32_to_u256(r);
}

__device__ cuda_bsgs_u256 center_mul_mod_p(const cuda_bsgs_u256 &a, const cuda_bsgs_u256 &b) {
  uint32_t aa[8], bb[8], t[16];
  center_u256_to_u32(a, aa);
  center_u256_to_u32(b, bb);
  #pragma unroll
  for (int i = 0; i < 16; i++) {
    t[i] = 0;
  }

  #pragma unroll
  for (int i = 0; i < 8; i++) {
    uint64_t carry = 0;
    #pragma unroll
    for (int j = 0; j < 8; j++) {
      const uint64_t acc = (uint64_t)t[i + j] + (uint64_t)aa[i] * (uint64_t)bb[j] + carry;
      t[i + j] = (uint32_t)acc;
      carry = acc >> 32;
    }
    int k = i + 8;
    while (carry != 0 && k < 16) {
      const uint64_t acc = (uint64_t)t[k] + carry;
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

  center_normalize_u64_12(r);
  center_fold_high_u64_12(r);
  center_normalize_u64_12(r);

  uint32_t out[8];
  #pragma unroll
  for (int i = 0; i < 8; i++) {
    out[i] = (uint32_t)r[i];
  }
  return center_canonicalize_u32_8(out);
}

__device__ __forceinline__ cuda_bsgs_u256 center_square_mod_p(const cuda_bsgs_u256 &a) {
  return center_mul_mod_p(a, a);
}

__device__ __forceinline__ cuda_bsgs_u256 center_from_u64(uint64_t x) {
  cuda_bsgs_u256 r = {};
  r.v[0] = x;
  return r;
}

__device__ cuda_bsgs_u256 center_inv_mod_p(const cuda_bsgs_u256 &a) {
  const cuda_bsgs_u256 p = center_p();
  const cuda_bsgs_u256 two = center_from_u64(2);
  const cuda_bsgs_u256 exponent = center_sub_mod_p(p, two);
  cuda_bsgs_u256 result = center_from_u64(1);

  for (int bit = 255; bit >= 0; bit--) {
    result = center_square_mod_p(result);
    if (center_u256_get_bit(exponent, bit)) {
      result = center_mul_mod_p(result, a);
    }
  }
  return result;
}

__device__ __forceinline__ cuda_bsgs_u256 center_neg_mod_p(const cuda_bsgs_u256 &a) {
  cuda_bsgs_u256 zero = {};
  return center_sub_mod_p(zero, a);
}

__device__ __forceinline__ void center_x_to_bytes(const cuda_bsgs_u256 &x, uint8_t out[32]) {
  int pos = 0;
  #pragma unroll
  for (int limb = 3; limb >= 0; limb--) {
    #pragma unroll
    for (int shift = 56; shift >= 0; shift -= 8) {
      out[pos++] = (uint8_t)(x.v[limb] >> shift);
    }
  }
}

__device__ __forceinline__ cuda_bsgs_u256 center_add_x_from_slope(const cuda_bsgs_u256 &start_x,
                                                                  const cuda_bsgs_u256 &gsn_x,
                                                                  const cuda_bsgs_u256 &slope) {
  cuda_bsgs_u256 rx = center_neg_mod_p(start_x);
  rx = center_add_mod_p(rx, center_square_mod_p(slope));
  rx = center_sub_mod_p(rx, gsn_x);
  return rx;
}

__device__ __forceinline__ cuda_bsgs_u256 center_plus_x(const cuda_bsgs_affine_point &startP,
                                                        const cuda_bsgs_affine_point &gsn,
                                                        const cuda_bsgs_u256 &dx_inv) {
  const cuda_bsgs_u256 dy = center_sub_mod_p(gsn.y, startP.y);
  const cuda_bsgs_u256 slope = center_mul_mod_p(dy, dx_inv);
  return center_add_x_from_slope(startP.x, gsn.x, slope);
}

__device__ __forceinline__ cuda_bsgs_u256 center_minus_x(const cuda_bsgs_affine_point &startP,
                                                         const cuda_bsgs_affine_point &gsn,
                                                         const cuda_bsgs_u256 &dx_inv) {
  cuda_bsgs_u256 dyn = center_neg_mod_p(gsn.y);
  dyn = center_sub_mod_p(dyn, startP.y);
  const cuda_bsgs_u256 slope = center_mul_mod_p(dyn, dx_inv);
  return center_add_x_from_slope(startP.x, gsn.x, slope);
}

__device__ __forceinline__ cuda_bsgs_affine_point center_point_zero(void) {
  cuda_bsgs_affine_point r = {};
  r.infinity = 1U;
  return r;
}

__device__ __forceinline__ cuda_bsgs_affine_point center_point_add_with_inv(
    const cuda_bsgs_affine_point &a,
    const cuda_bsgs_affine_point &b,
    const cuda_bsgs_u256 &dx_inv) {
  cuda_bsgs_affine_point r = {};
  const cuda_bsgs_u256 dy = center_sub_mod_p(b.y, a.y);
  const cuda_bsgs_u256 slope = center_mul_mod_p(dy, dx_inv);
  r.x = center_add_x_from_slope(a.x, b.x, slope);
  const cuda_bsgs_u256 x_delta = center_sub_mod_p(a.x, r.x);
  r.y = center_sub_mod_p(center_mul_mod_p(slope, x_delta), a.y);
  r.infinity = 0U;
  return r;
}

__device__ __forceinline__ cuda_bsgs_affine_point center_point_double_slow(
    const cuda_bsgs_affine_point &a) {
  if (a.infinity) {
    return a;
  }
  const cuda_bsgs_u256 two_y = center_add_mod_p(a.y, a.y);
  if ((two_y.v[0] | two_y.v[1] | two_y.v[2] | two_y.v[3]) == 0ULL) {
    return center_point_zero();
  }
  const cuda_bsgs_u256 inv = center_inv_mod_p(two_y);
  cuda_bsgs_u256 three_x2 = center_square_mod_p(a.x);
  three_x2 = center_add_mod_p(three_x2, three_x2);
  three_x2 = center_add_mod_p(three_x2, center_square_mod_p(a.x));
  const cuda_bsgs_u256 slope = center_mul_mod_p(three_x2, inv);
  cuda_bsgs_affine_point r = {};
  r.x = center_sub_mod_p(center_square_mod_p(slope), center_add_mod_p(a.x, a.x));
  const cuda_bsgs_u256 x_delta = center_sub_mod_p(a.x, r.x);
  r.y = center_sub_mod_p(center_mul_mod_p(slope, x_delta), a.y);
  r.infinity = 0U;
  return r;
}

__device__ cuda_bsgs_affine_point center_point_add_slow(const cuda_bsgs_affine_point &a,
                                                        const cuda_bsgs_affine_point &b) {
  if (a.infinity) return b;
  if (b.infinity) return a;
  if (center_u256_cmp(a.x, b.x) == 0) {
    if (center_u256_cmp(a.y, b.y) == 0) {
      return center_point_double_slow(a);
    }
    return center_point_zero();
  }
  const cuda_bsgs_u256 dx = center_sub_mod_p(b.x, a.x);
  return center_point_add_with_inv(a, b, center_inv_mod_p(dx));
}

__device__ __forceinline__ cuda_point_affine center_to_secp_point(const cuda_bsgs_affine_point &p) {
  cuda_point_affine r;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    r.x.v[i] = p.x.v[i];
    r.y.v[i] = p.y.v[i];
  }
  r.infinity = (int)p.infinity;
  return r;
}

__device__ __forceinline__ cuda_bsgs_affine_point center_neg_point(const cuda_bsgs_affine_point &p) {
  cuda_bsgs_affine_point r = p;
  if (!r.infinity) {
    r.y = center_neg_mod_p(r.y);
  }
  return r;
}

__device__ __forceinline__ cuda_bsgs_scalar scalar_from_u32(uint32_t x) {
  cuda_bsgs_scalar r = {};
  r.v[0] = x;
  return r;
}

__device__ __forceinline__ uint32_t scalar_get_bit(const cuda_bsgs_scalar &a, int bit) {
  return (uint32_t)((a.v[bit >> 6] >> (bit & 63)) & 1ULL);
}

__device__ cuda_bsgs_scalar scalar_add(const cuda_bsgs_scalar &a, const cuda_bsgs_scalar &b) {
  cuda_bsgs_scalar r;
  uint64_t carry = 0;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    const uint64_t s = a.v[i] + carry;
    const uint64_t c1 = (s < carry);
    const uint64_t s2 = s + b.v[i];
    const uint64_t c2 = (s2 < s);
    r.v[i] = s2;
    carry = c1 | c2;
  }
  return r;
}

__device__ cuda_bsgs_scalar scalar_mul_u32(const cuda_bsgs_scalar &a, uint32_t m) {
  cuda_bsgs_scalar r;
  unsigned __int128 carry = 0;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    const unsigned __int128 product = (unsigned __int128)a.v[i] * (unsigned __int128)m + carry;
    r.v[i] = (uint64_t)product;
    carry = product >> 64;
  }
  return r;
}

__device__ cuda_bsgs_affine_point scalar_public_key(const cuda_bsgs_scalar &scalar) {
  cuda_point_affine acc = {};
  acc.infinity = 1;
  cuda_point_affine addend = cuda_secp256k1_generator();
  for (int bit = 0; bit < 256; bit++) {
    if (scalar_get_bit(scalar, bit)) {
      acc = cuda_point_add_affine(acc, addend);
    }
    addend = cuda_point_double_affine(addend);
  }
  cuda_bsgs_affine_point r = {};
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    r.x.v[i] = acc.x.v[i];
    r.y.v[i] = acc.y.v[i];
  }
  r.infinity = (uint32_t)acc.infinity;
  return r;
}

__device__ __forceinline__ void scalar_x_to_bytes(const cuda_bsgs_u256 &x, uint8_t out[32]) {
  center_x_to_bytes(x, out);
}

__device__ __forceinline__ cuda_bsgs_affine_point secp_to_center_point(const cuda_point_affine &p) {
  cuda_bsgs_affine_point r = {};
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    r.x.v[i] = p.x.v[i];
    r.y.v[i] = p.y.v[i];
  }
  r.infinity = (uint32_t)p.infinity;
  return r;
}

__global__ void generate_center_offsets_kernel(const cuda_bsgs_affine_point *_2gsn,
                                               cuda_bsgs_affine_point *offsets) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  offsets[0] = center_point_zero();
  offsets[1] = *_2gsn;
  cuda_point_affine step = center_to_secp_point(*_2gsn);
  cuda_point_affine current = step;
  for (int i = 2; i < CUDA_BSGS_CENTER_OFFSET_COUNT; i++) {
    current = cuda_point_add_affine(current, step);
    offsets[i] = secp_to_center_point(current);
  }
}

__global__ void generate_center_sequence_kernel(const cuda_bsgs_affine_point seed,
                                                uint64_t first_user_index,
                                                const cuda_bsgs_affine_point *offsets,
                                                cuda_bsgs_center_work *works,
                                                cuda_bsgs_affine_point *next_seed,
                                                uint32_t work_count) {
  const uint32_t i = blockIdx.x;
  if (threadIdx.x == 0 && i < work_count) {
    works[i].user_index = first_user_index + (uint64_t)i;
    if (i == 0 || offsets[i].infinity) {
      works[i].startP = seed;
    } else {
      cuda_point_affine secp_seed = center_to_secp_point(seed);
      cuda_point_affine secp_offset = center_to_secp_point(offsets[i]);
      works[i].startP = secp_to_center_point(cuda_point_add_affine(secp_seed, secp_offset));
    }
  }

  if (threadIdx.x == 0 && blockIdx.x == 0 && next_seed != NULL) {
    cuda_point_affine secp_seed = center_to_secp_point(seed);
    cuda_point_affine secp_offset = center_to_secp_point(offsets[work_count]);
    *next_seed = secp_to_center_point(cuda_point_add_affine(secp_seed, secp_offset));
  }
}

__device__ __forceinline__ void center_record_bloom1_hit(const cuda_bloom_device_view *bloom1,
                                                         const cuda_bsgs_u256 &x,
                                                         uint32_t work_index,
                                                         uint64_t user_index,
                                                         uint32_t point_index,
                                                         cuda_bsgs_center_hit *hits,
                                                         uint32_t max_hits,
                                                         uint32_t *hit_count) {
  uint8_t xbytes[32];
  center_x_to_bytes(x, xbytes);
  const uint8_t shard = xbytes[0];
  if (!cuda_bloom_check32(&bloom1[shard], xbytes)) {
    return;
  }

  const uint32_t slot = atomicAdd(hit_count, 1u);
  if (slot < max_hits && hits != NULL) {
    hits[slot].work_index = work_index;
    hits[slot].point_index = point_index;
    hits[slot].table_index = user_index;
    hits[slot].shard = shard;
    hits[slot].stage = 1;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
      hits[slot].x[i] = xbytes[i];
    }
    #pragma unroll
    for (int i = 0; i < 7; i++) {
      hits[slot].reserved[i] = 0;
    }
  }
}

__device__ __forceinline__ void record_filter_hit(uint32_t work_index,
                                                  uint32_t point_index,
                                                  uint32_t second_index,
                                                  uint32_t third_index,
                                                  uint64_t table_index,
                                                  uint8_t stage,
                                                  cuda_bsgs_filter_hit *hits,
                                                  uint32_t max_hits,
                                                  uint32_t *hit_count) {
  const uint32_t slot = atomicAdd(hit_count, 1u);
  if (slot < max_hits && hits != NULL) {
    hits[slot].work_index = work_index;
    hits[slot].point_index = point_index;
    hits[slot].second_index = second_index;
    hits[slot].third_index = third_index;
    hits[slot].table_index = table_index;
    hits[slot].stage = stage;
    #pragma unroll
    for (int i = 0; i < 7; i++) {
      hits[slot].reserved[i] = 0;
    }
  }
}

__global__ void filter_center_hits_kernel(cuda_bloom_device_view *bloom2,
                                          cuda_bloom_device_view *bloom3,
                                          const uint8_t *table_values_soa,
                                          const uint64_t *table_indices,
                                          uint64_t table_count,
                                          const cuda_bsgs_affine_point *original_points,
                                          uint32_t original_point_count,
                                          const cuda_bsgs_affine_point *amp2,
                                          const cuda_bsgs_affine_point *amp3,
                                          const cuda_bsgs_scalar base_key,
                                          const cuda_bsgs_scalar m_double,
                                          const cuda_bsgs_scalar m2_double,
                                          uint32_t k_index,
                                          const cuda_bsgs_center_hit *hits_in,
                                          uint32_t hit_count_in,
                                          cuda_bsgs_filter_hit *hits_out,
                                          uint32_t max_hits_out,
                                          uint32_t *hit_count_out) {
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= hit_count_in || k_index >= original_point_count) {
    return;
  }

  const cuda_bsgs_center_hit hit = hits_in[tid];
  const uint32_t a = (uint32_t)(hit.table_index * CUDA_BSGS_CENTER_POINTS +
                                (uint64_t)hit.point_index);
  cuda_bsgs_scalar first_scalar = scalar_add(base_key, scalar_mul_u32(m_double, a));
  cuda_bsgs_affine_point first_point = scalar_public_key(first_scalar);
  cuda_bsgs_affine_point q = center_point_add_slow(original_points[k_index],
                                                   center_neg_point(first_point));

  uint8_t xbytes[32];
  for (uint32_t second = 0; second < 32; second++) {
    const cuda_bsgs_affine_point q2 = center_point_add_slow(q, amp2[second]);
    center_x_to_bytes(q2.x, xbytes);
    const uint8_t shard2 = xbytes[0];
    if (!cuda_bloom_check32(&bloom2[shard2], xbytes)) {
      continue;
    }

    cuda_bsgs_scalar second_scalar = scalar_add(first_scalar, scalar_mul_u32(m2_double, second));
    cuda_bsgs_affine_point second_point = scalar_public_key(second_scalar);
    cuda_bsgs_affine_point q3base = center_point_add_slow(original_points[k_index],
                                                          center_neg_point(second_point));

    for (uint32_t third = 0; third < 32; third++) {
      const cuda_bsgs_affine_point q3 = center_point_add_slow(q3base, amp3[third]);
      center_x_to_bytes(q3.x, xbytes);
      const uint8_t shard3 = xbytes[0];
      if (cuda_bloom_check32(&bloom3[shard3], xbytes)) {
        uint64_t table_index = 0;
        if (table_values_soa != NULL && table_indices != NULL &&
            cuda_table_search6_soa(table_values_soa, table_indices, table_count, xbytes, &table_index)) {
          record_filter_hit(hit.work_index, hit.point_index, second, third, table_index, 3,
                            hits_out, max_hits_out, hit_count_out);
        }
      } else if (center_u256_cmp(q3base.x, amp3[third].x) == 0) {
        record_filter_hit(hit.work_index, hit.point_index, second, third, 0, 4,
                          hits_out, max_hits_out, hit_count_out);
      }
    }
  }
}

__global__ void probe_xpoints_kernel(cuda_bloom_device_view *bloom1,
                                     cuda_bloom_device_view *bloom3,
                                     cuda_bsgs_xvalue *table,
                                     const uint8_t *table_values_soa,
                                     const uint64_t *table_indices,
                                     uint64_t table_count,
                                     const uint8_t *xpoints32,
                                     uint32_t count,
                                     cuda_bsgs_hit *hits,
                                     uint32_t max_hits,
                                     uint32_t *hit_count) {
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= count) {
    return;
  }

  const uint8_t *xpoint = xpoints32 + ((size_t)tid * 32);
  const uint8_t shard = xpoint[0];
  if (!cuda_bloom_check32(&bloom1[shard], xpoint)) {
    return;
  }

  uint64_t table_index = 0;
  uint8_t stage = 1;
  if (bloom3 != NULL && table_values_soa != NULL && table_indices != NULL) {
    if (!cuda_bloom_check32(&bloom3[shard], xpoint)) {
      return;
    }
    if (!cuda_table_search6_soa(table_values_soa, table_indices, table_count, xpoint, &table_index)) {
      return;
    }
    stage = 3;
  } else if (bloom3 != NULL && table != NULL) {
    if (!cuda_bloom_check32(&bloom3[shard], xpoint)) {
      return;
    }
    if (!cuda_table_search6(table, table_count, xpoint, &table_index)) {
      return;
    }
    stage = 3;
  }

  const uint32_t slot = atomicAdd(hit_count, 1u);
  if (slot < max_hits) {
    hits[slot].xpoint_index = tid;
    hits[slot].table_index = table_index;
    hits[slot].stage = stage;
  }
}

__global__ void probe_center_batches_kernel(cuda_bloom_device_view *bloom1,
                                            const cuda_bsgs_affine_point *gsn,
                                            const cuda_bsgs_affine_point *_2gsn,
                                            const cuda_bsgs_center_work *works,
                                            uint32_t work_count,
                                            cuda_bsgs_center_hit *hits,
                                            uint32_t max_hits,
                                            uint32_t *hit_count) {
  const uint32_t work_index = blockIdx.x;
  if (work_index >= work_count) {
    return;
  }

  const cuda_bsgs_affine_point startP = works[work_index].startP;
  if (startP.infinity) {
    return;
  }

  /* One block owns one center batch. Denominators and prefix/suffix products
     are built cooperatively; only the final field inverse is scalar. */
  extern __shared__ cuda_bsgs_u256 shared_u256[];
  cuda_bsgs_u256 *dx = shared_u256;
  cuda_bsgs_u256 *prefix = shared_u256 + CUDA_BSGS_CENTER_SCAN;
  cuda_bsgs_u256 *suffix = prefix + CUDA_BSGS_CENTER_SCAN;
  __shared__ cuda_bsgs_u256 inv_total;
  const cuda_bsgs_u256 one = center_from_u64(1);

  for (int i = (int)threadIdx.x; i < CUDA_BSGS_CENTER_SCAN; i += blockDim.x) {
    if (i < CUDA_BSGS_GSN_COUNT) {
      dx[i] = center_sub_mod_p(gsn[i].x, startP.x);
    } else {
      dx[i] = one;
    }
  }
  __syncthreads();

  for (int i = (int)threadIdx.x; i < CUDA_BSGS_CENTER_SCAN; i += blockDim.x) {
    prefix[i] = dx[i];
    suffix[CUDA_BSGS_CENTER_SCAN - 1 - i] = dx[i];
  }
  __syncthreads();

  for (int stride = 1; stride < CUDA_BSGS_CENTER_SCAN; stride <<= 1) {
    const int idx = ((int)threadIdx.x + 1) * (stride << 1) - 1;
    if (idx < CUDA_BSGS_CENTER_SCAN) {
      prefix[idx] = center_mul_mod_p(prefix[idx - stride], prefix[idx]);
      suffix[idx] = center_mul_mod_p(suffix[idx - stride], suffix[idx]);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    inv_total = center_inv_mod_p(prefix[CUDA_BSGS_CENTER_SCAN - 1]);
    prefix[CUDA_BSGS_CENTER_SCAN - 1] = one;
    suffix[CUDA_BSGS_CENTER_SCAN - 1] = one;
  }
  __syncthreads();

  for (int stride = CUDA_BSGS_CENTER_SCAN >> 1; stride > 0; stride >>= 1) {
    const int idx = ((int)threadIdx.x + 1) * (stride << 1) - 1;
    if (idx < CUDA_BSGS_CENTER_SCAN) {
      const cuda_bsgs_u256 left_product = prefix[idx - stride];
      const cuda_bsgs_u256 suffix_left_product = suffix[idx - stride];
      prefix[idx - stride] = prefix[idx];
      prefix[idx] = center_mul_mod_p(prefix[idx], left_product);
      suffix[idx - stride] = suffix[idx];
      suffix[idx] = center_mul_mod_p(suffix[idx], suffix_left_product);
    }
    __syncthreads();
  }

  for (int i = (int)threadIdx.x; i < CUDA_BSGS_CENTER_INVERSES; i += blockDim.x) {
    const cuda_bsgs_u256 product_except_i =
        center_mul_mod_p(prefix[i], suffix[CUDA_BSGS_CENTER_SCAN - 1 - i]);
    dx[i] = center_mul_mod_p(product_except_i, inv_total);
  }

  __syncthreads();

  for (int point_index = (int)threadIdx.x; point_index < CUDA_BSGS_CENTER_POINTS; point_index += blockDim.x) {
    cuda_bsgs_u256 x;
    if (point_index < CUDA_BSGS_GSN_COUNT) {
      const int gsn_index = CUDA_BSGS_GSN_COUNT - 1 - point_index;
      x = center_minus_x(startP, gsn[gsn_index], dx[gsn_index]);
    } else if (point_index == CUDA_BSGS_GSN_COUNT) {
      x = startP.x;
    } else {
      const int gsn_index = point_index - CUDA_BSGS_GSN_COUNT - 1;
      x = center_plus_x(startP, gsn[gsn_index], dx[gsn_index]);
    }
    center_record_bloom1_hit(bloom1, x, work_index, works[work_index].user_index,
                             (uint32_t)point_index, hits, max_hits, hit_count);
  }
}

extern "C" int cuda_bsgs_available(void) {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

extern "C" int cuda_bsgs_create(cuda_bsgs_context **ctx) {
  if (ctx == NULL) {
    return 0;
  }
  cuda_bsgs_context *tmp = (cuda_bsgs_context *)calloc(1, sizeof(cuda_bsgs_context));
  if (tmp == NULL) {
    return 0;
  }
  tmp->last_error[0] = 0;
  *ctx = tmp;
  return 1;
}

extern "C" void cuda_bsgs_destroy(cuda_bsgs_context *ctx) {
  if (ctx == NULL) {
    return;
  }
  free_one_bloom_set(ctx->bloom1);
  free_one_bloom_set(ctx->bloom2);
  free_one_bloom_set(ctx->bloom3);
  if (ctx->d_bloom1 != NULL) cudaFree(ctx->d_bloom1);
  if (ctx->d_bloom2 != NULL) cudaFree(ctx->d_bloom2);
  if (ctx->d_bloom3 != NULL) cudaFree(ctx->d_bloom3);
  if (ctx->d_bloom1_data != NULL) cudaFree(ctx->d_bloom1_data);
  if (ctx->d_bloom2_data != NULL) cudaFree(ctx->d_bloom2_data);
  if (ctx->d_bloom3_data != NULL) cudaFree(ctx->d_bloom3_data);
  if (ctx->table != NULL) {
    cudaFree(ctx->table);
  }
  if (ctx->table_values_soa != NULL) cudaFree(ctx->table_values_soa);
  if (ctx->table_indices != NULL) cudaFree(ctx->table_indices);
  if (ctx->d_gsn != NULL) cudaFree(ctx->d_gsn);
  if (ctx->d_2gsn != NULL) cudaFree(ctx->d_2gsn);
  if (ctx->d_original_points != NULL) cudaFree(ctx->d_original_points);
  if (ctx->d_amp2 != NULL) cudaFree(ctx->d_amp2);
  if (ctx->d_amp3 != NULL) cudaFree(ctx->d_amp3);
  if (ctx->d_m_double != NULL) cudaFree(ctx->d_m_double);
  if (ctx->d_m2_double != NULL) cudaFree(ctx->d_m2_double);
  if (ctx->d_m3 != NULL) cudaFree(ctx->d_m3);
  if (ctx->d_m3_double != NULL) cudaFree(ctx->d_m3_double);
  if (ctx->d_order != NULL) cudaFree(ctx->d_order);
  if (ctx->d_center_offsets != NULL) cudaFree(ctx->d_center_offsets);
  if (ctx->d_center_next_seed != NULL) cudaFree(ctx->d_center_next_seed);
  if (ctx->d_center_works != NULL) cudaFree(ctx->d_center_works);
  if (ctx->d_center_hits != NULL) cudaFree(ctx->d_center_hits);
  if (ctx->d_center_hit_count != NULL) cudaFree(ctx->d_center_hit_count);
  if (ctx->d_filter_input_hits != NULL) cudaFree(ctx->d_filter_input_hits);
  if (ctx->d_filter_hits != NULL) cudaFree(ctx->d_filter_hits);
  if (ctx->d_filter_hit_count != NULL) cudaFree(ctx->d_filter_hit_count);
  if (ctx->d_probe_xpoints != NULL) cudaFree(ctx->d_probe_xpoints);
  if (ctx->d_probe_hits != NULL) cudaFree(ctx->d_probe_hits);
  if (ctx->d_probe_hit_count != NULL) cudaFree(ctx->d_probe_hit_count);
  free(ctx);
}

extern "C" int cuda_bsgs_host_alloc(void **ptr, size_t bytes) {
  if (ptr == NULL || bytes == 0) {
    return 0;
  }
  *ptr = NULL;
  return cudaHostAlloc(ptr, bytes, cudaHostAllocDefault) == cudaSuccess;
}

extern "C" void cuda_bsgs_host_free(void *ptr) {
  if (ptr != NULL) {
    cudaFreeHost(ptr);
  }
}

extern "C" int cuda_bsgs_probe_workspace_create(cuda_bsgs_probe_workspace **workspace) {
  if (workspace == NULL) {
    return 0;
  }
  *workspace = NULL;
  cuda_bsgs_probe_workspace *tmp =
      (cuda_bsgs_probe_workspace *)calloc(1, sizeof(cuda_bsgs_probe_workspace));
  if (tmp == NULL) {
    return 0;
  }
  if (cudaStreamCreateWithFlags(&tmp->stream, cudaStreamNonBlocking) != cudaSuccess) {
    free(tmp);
    return 0;
  }
  *workspace = tmp;
  return 1;
}

extern "C" void cuda_bsgs_probe_workspace_destroy(cuda_bsgs_probe_workspace *workspace) {
  if (workspace == NULL) {
    return;
  }
  if (workspace->d_xpoints != NULL) cudaFree(workspace->d_xpoints);
  if (workspace->d_hits != NULL) cudaFree(workspace->d_hits);
  if (workspace->d_hit_count != NULL) cudaFree(workspace->d_hit_count);
  if (workspace->stream != NULL) cudaStreamDestroy(workspace->stream);
  free(workspace);
}

extern "C" int cuda_bsgs_center_workspace_create(cuda_bsgs_center_workspace **workspace) {
  if (workspace == NULL) {
    return 0;
  }
  *workspace = NULL;
  cuda_bsgs_center_workspace *tmp =
      (cuda_bsgs_center_workspace *)calloc(1, sizeof(cuda_bsgs_center_workspace));
  if (tmp == NULL) {
    return 0;
  }
  if (cudaStreamCreateWithFlags(&tmp->stream, cudaStreamNonBlocking) != cudaSuccess) {
    free(tmp);
    return 0;
  }
  *workspace = tmp;
  return 1;
}

extern "C" void cuda_bsgs_center_workspace_destroy(cuda_bsgs_center_workspace *workspace) {
  if (workspace == NULL) {
    return;
  }
  if (workspace->d_works != NULL) cudaFree(workspace->d_works);
  if (workspace->d_hits != NULL) cudaFree(workspace->d_hits);
  if (workspace->d_hit_count != NULL) cudaFree(workspace->d_hit_count);
  if (workspace->d_next_seed != NULL) cudaFree(workspace->d_next_seed);
  if (workspace->stream != NULL) cudaStreamDestroy(workspace->stream);
  free(workspace);
}

extern "C" int cuda_bsgs_upload_blooms(cuda_bsgs_context *ctx,
                                        const struct bloom *bloom1,
                                        const struct bloom *bloom2,
                                        const struct bloom *bloom3,
                                        size_t shard_count) {
  if (ctx == NULL || bloom1 == NULL || bloom2 == NULL || bloom3 == NULL ||
      shard_count == 0 || shard_count > CUDA_BSGS_SHARDS) {
    set_error(ctx, "invalid bloom upload arguments", cudaSuccess);
    return 0;
  }
  free_one_bloom_set(ctx->bloom1);
  free_one_bloom_set(ctx->bloom2);
  free_one_bloom_set(ctx->bloom3);
  ctx->bloom_shard_count = 0;
  const int ok = upload_one_bloom_set(ctx, ctx->bloom1, bloom1, &ctx->d_bloom1_data, shard_count) &&
                 upload_one_bloom_set(ctx, ctx->bloom2, bloom2, &ctx->d_bloom2_data, shard_count) &&
                 upload_one_bloom_set(ctx, ctx->bloom3, bloom3, &ctx->d_bloom3_data, shard_count) &&
                 upload_bloom_views(ctx, &ctx->d_bloom1, ctx->bloom1, shard_count, "cuda upload bloom1 views") &&
                 upload_bloom_views(ctx, &ctx->d_bloom2, ctx->bloom2, shard_count, "cuda upload bloom2 views") &&
                 upload_bloom_views(ctx, &ctx->d_bloom3, ctx->bloom3, shard_count, "cuda upload bloom3 views");
  if (ok) {
    ctx->bloom_shard_count = shard_count;
  }
  return ok;
}

extern "C" int cuda_bsgs_upload_table(cuda_bsgs_context *ctx,
                                       const cuda_bsgs_xvalue *table,
                                       uint64_t table_count) {
  if (ctx == NULL || table == NULL || table_count == 0) {
    set_error(ctx, "invalid table upload arguments", cudaSuccess);
    return 0;
  }
  if (ctx->table != NULL) {
    cudaFree(ctx->table);
    ctx->table = NULL;
  }
  if (ctx->table_values_soa != NULL) {
    cudaFree(ctx->table_values_soa);
    ctx->table_values_soa = NULL;
  }
  if (ctx->table_indices != NULL) {
    cudaFree(ctx->table_indices);
    ctx->table_indices = NULL;
  }
  const size_t bytes = (size_t)table_count * sizeof(cuda_bsgs_xvalue);
  if (!check_cuda(ctx, "cudaMalloc bP table", cudaMalloc((void **)&ctx->table, bytes))) {
    return 0;
  }
  if (!check_cuda(ctx, "cudaMemcpy bP table", cudaMemcpy(ctx->table, table, bytes, cudaMemcpyHostToDevice))) {
    return 0;
  }
  uint8_t *values_soa = (uint8_t *)malloc((size_t)table_count * 6);
  uint64_t *indices = (uint64_t *)malloc((size_t)table_count * sizeof(uint64_t));
  if (values_soa == NULL || indices == NULL) {
    free(values_soa);
    free(indices);
    set_error(ctx, "malloc bP SoA staging", cudaSuccess);
    return 0;
  }
  for (uint64_t i = 0; i < table_count; i++) {
    #pragma unroll
    for (int b = 0; b < 6; b++) {
      values_soa[(uint64_t)b * table_count + i] = table[i].value[b];
    }
    indices[i] = table[i].index;
  }
  if (!check_cuda(ctx, "cudaMalloc bP table values SoA",
                  cudaMalloc((void **)&ctx->table_values_soa, (size_t)table_count * 6)) ||
      !check_cuda(ctx, "cudaMalloc bP table indices",
                  cudaMalloc((void **)&ctx->table_indices, (size_t)table_count * sizeof(uint64_t))) ||
      !check_cuda(ctx, "cudaMemcpy bP table values SoA",
                  cudaMemcpy(ctx->table_values_soa, values_soa, (size_t)table_count * 6, cudaMemcpyHostToDevice)) ||
      !check_cuda(ctx, "cudaMemcpy bP table indices",
                  cudaMemcpy(ctx->table_indices, indices, (size_t)table_count * sizeof(uint64_t), cudaMemcpyHostToDevice))) {
    free(values_soa);
    free(indices);
    return 0;
  }
  free(values_soa);
  free(indices);
  ctx->table_count = table_count;
  return 1;
}

extern "C" int cuda_bsgs_save_native_cache(cuda_bsgs_context *ctx,
                                            const char *filename) {
  if (ctx == NULL || filename == NULL || ctx->d_bloom1_data == NULL ||
      ctx->d_bloom2_data == NULL || ctx->d_bloom3_data == NULL ||
      ctx->table_values_soa == NULL || ctx->table_indices == NULL ||
      ctx->table_count == 0 || ctx->bloom_shard_count == 0) {
    set_error(ctx, "invalid native cache save state", cudaSuccess);
    return 0;
  }

  FILE *file = fopen(filename, "wb");
  if (file == NULL) {
    set_error(ctx, "open native cache for write", cudaSuccess);
    return 0;
  }

  cuda_bsgs_cache_header header = {};
  header.magic = CUDA_BSGS_CACHE_MAGIC;
  header.version = CUDA_BSGS_CACHE_VERSION;
  header.shard_count = (uint32_t)ctx->bloom_shard_count;
  header.table_count = ctx->table_count;

  int ok = write_exact(file, &header, sizeof(header)) &&
           save_one_native_bloom_set(ctx, file, ctx->bloom1, ctx->d_bloom1_data, ctx->bloom_shard_count) &&
           save_one_native_bloom_set(ctx, file, ctx->bloom2, ctx->d_bloom2_data, ctx->bloom_shard_count) &&
           save_one_native_bloom_set(ctx, file, ctx->bloom3, ctx->d_bloom3_data, ctx->bloom_shard_count);

  if (ok) {
    const size_t values_bytes = (size_t)ctx->table_count * 6;
    const size_t indices_bytes = (size_t)ctx->table_count * sizeof(uint64_t);
    uint8_t *values = (uint8_t *)malloc(values_bytes);
    uint64_t *indices = (uint64_t *)malloc(indices_bytes);
    if (values == NULL || indices == NULL) {
      free(values);
      free(indices);
      set_error(ctx, "malloc native table save staging", cudaSuccess);
      ok = 0;
    } else if (!check_cuda(ctx, "copy native table values from device",
                           cudaMemcpy(values, ctx->table_values_soa, values_bytes, cudaMemcpyDeviceToHost)) ||
               !check_cuda(ctx, "copy native table indices from device",
                           cudaMemcpy(indices, ctx->table_indices, indices_bytes, cudaMemcpyDeviceToHost))) {
      ok = 0;
    } else {
      ok = write_exact(file, values, values_bytes) &&
           write_exact(file, indices, indices_bytes);
      if (!ok) {
        set_error(ctx, "write native table", cudaSuccess);
      }
    }
    free(values);
    free(indices);
  }

  fclose(file);
  return ok;
}

extern "C" int cuda_bsgs_load_native_cache(cuda_bsgs_context *ctx,
                                            const char *filename) {
  if (ctx == NULL || filename == NULL) {
    set_error(ctx, "invalid native cache load arguments", cudaSuccess);
    return 0;
  }

  FILE *file = fopen(filename, "rb");
  if (file == NULL) {
    set_error(ctx, "native cache not found", cudaSuccess);
    return 0;
  }

  cuda_bsgs_cache_header header = {};
  if (!read_exact(file, &header, sizeof(header)) ||
      header.magic != CUDA_BSGS_CACHE_MAGIC ||
      header.version != CUDA_BSGS_CACHE_VERSION ||
      header.shard_count == 0 || header.shard_count > CUDA_BSGS_SHARDS ||
      header.table_count == 0) {
    fclose(file);
    set_error(ctx, "invalid native cache header", cudaSuccess);
    return 0;
  }

  free_one_bloom_set(ctx->bloom1);
  free_one_bloom_set(ctx->bloom2);
  free_one_bloom_set(ctx->bloom3);
  ctx->bloom_shard_count = 0;

  int ok = load_one_native_bloom_set(ctx, file, ctx->bloom1, &ctx->d_bloom1,
                                     &ctx->d_bloom1_data, header.shard_count,
                                     "cuda upload native bloom1 views") &&
           load_one_native_bloom_set(ctx, file, ctx->bloom2, &ctx->d_bloom2,
                                     &ctx->d_bloom2_data, header.shard_count,
                                     "cuda upload native bloom2 views") &&
           load_one_native_bloom_set(ctx, file, ctx->bloom3, &ctx->d_bloom3,
                                     &ctx->d_bloom3_data, header.shard_count,
                                     "cuda upload native bloom3 views");

  if (ok) {
    const size_t values_bytes = (size_t)header.table_count * 6;
    const size_t indices_bytes = (size_t)header.table_count * sizeof(uint64_t);
    uint8_t *values = (uint8_t *)malloc(values_bytes);
    uint64_t *indices = (uint64_t *)malloc(indices_bytes);
    if (values == NULL || indices == NULL) {
      free(values);
      free(indices);
      set_error(ctx, "malloc native table load staging", cudaSuccess);
      ok = 0;
    } else if (!read_exact(file, values, values_bytes) ||
               !read_exact(file, indices, indices_bytes)) {
      set_error(ctx, "read native table", cudaSuccess);
      ok = 0;
    } else {
      if (ctx->table != NULL) {
        cudaFree(ctx->table);
        ctx->table = NULL;
      }
      if (ctx->table_values_soa != NULL) cudaFree(ctx->table_values_soa);
      if (ctx->table_indices != NULL) cudaFree(ctx->table_indices);
      ctx->table_values_soa = NULL;
      ctx->table_indices = NULL;
      if (!check_cuda(ctx, "cudaMalloc native table values",
                      cudaMalloc((void **)&ctx->table_values_soa, values_bytes)) ||
          !check_cuda(ctx, "cudaMalloc native table indices",
                      cudaMalloc((void **)&ctx->table_indices, indices_bytes)) ||
          !check_cuda(ctx, "cudaMemcpy native table values",
                      cudaMemcpy(ctx->table_values_soa, values, values_bytes, cudaMemcpyHostToDevice)) ||
          !check_cuda(ctx, "cudaMemcpy native table indices",
                      cudaMemcpy(ctx->table_indices, indices, indices_bytes, cudaMemcpyHostToDevice))) {
        ok = 0;
      }
    }
    free(values);
    free(indices);
  }

  fclose(file);
  if (ok) {
    ctx->bloom_shard_count = header.shard_count;
    ctx->table_count = header.table_count;
  }
  return ok;
}

extern "C" int cuda_bsgs_probe_xpoints(cuda_bsgs_context *ctx,
                                        const uint8_t *xpoints32,
                                        uint32_t xpoint_count,
                                        cuda_bsgs_hit *hits,
                                        uint32_t max_hits,
                                        uint32_t *hit_count) {
  if (ctx == NULL || xpoints32 == NULL || hits == NULL || hit_count == NULL || xpoint_count == 0) {
    set_error(ctx, "invalid probe arguments", cudaSuccess);
    return 0;
  }

  uint8_t *d_xpoints = NULL;
  cuda_bsgs_hit *d_hits = NULL;
  uint32_t *d_hit_count = NULL;
  const size_t xpoint_bytes = (size_t)xpoint_count * 32;
  const size_t hit_bytes = (size_t)max_hits * sizeof(cuda_bsgs_hit);

  if (!check_cuda(ctx, "cudaMalloc xpoints", cudaMalloc((void **)&d_xpoints, xpoint_bytes)) ||
      !check_cuda(ctx, "cudaMalloc hits", cudaMalloc((void **)&d_hits, hit_bytes)) ||
      !check_cuda(ctx, "cudaMalloc hit count", cudaMalloc((void **)&d_hit_count, sizeof(uint32_t))) ||
      !check_cuda(ctx, "cudaMemcpy xpoints", cudaMemcpy(d_xpoints, xpoints32, xpoint_bytes, cudaMemcpyHostToDevice)) ||
      !check_cuda(ctx, "cudaMemset hit count", cudaMemset(d_hit_count, 0, sizeof(uint32_t)))) {
    cudaFree(d_xpoints);
    cudaFree(d_hits);
    cudaFree(d_hit_count);
    return 0;
  }

  const int block = 512;
  const int grid = (xpoint_count + block - 1) / block;
  probe_xpoints_kernel<<<grid, block>>>(ctx->d_bloom1, ctx->d_bloom3, ctx->table,
                                        ctx->table_values_soa, ctx->table_indices, ctx->table_count,
                                        d_xpoints, xpoint_count, d_hits, max_hits, d_hit_count);
  if (!check_cuda(ctx, "probe_xpoints_kernel", cudaGetLastError()) ||
      !check_cuda(ctx, "probe synchronize", cudaDeviceSynchronize()) ||
      !check_cuda(ctx, "copy hit count", cudaMemcpy(hit_count, d_hit_count, sizeof(uint32_t), cudaMemcpyDeviceToHost))) {
    cudaFree(d_xpoints);
    cudaFree(d_hits);
    cudaFree(d_hit_count);
    return 0;
  }

  const uint32_t copy_hits = (*hit_count < max_hits) ? *hit_count : max_hits;
  if (copy_hits > 0 &&
      !check_cuda(ctx, "copy hits", cudaMemcpy(hits, d_hits, (size_t)copy_hits * sizeof(cuda_bsgs_hit),
                                               cudaMemcpyDeviceToHost))) {
    cudaFree(d_xpoints);
    cudaFree(d_hits);
    cudaFree(d_hit_count);
    return 0;
  }

  cudaFree(d_xpoints);
  cudaFree(d_hits);
  cudaFree(d_hit_count);
  return 1;
}

extern "C" int cuda_bsgs_probe_xpoints_bloom1(cuda_bsgs_context *ctx,
                                               const uint8_t *xpoints32,
                                               uint32_t xpoint_count,
                                               cuda_bsgs_hit *hits,
                                               uint32_t max_hits,
                                               uint32_t *hit_count) {
  if (ctx == NULL || xpoints32 == NULL || hits == NULL || hit_count == NULL || xpoint_count == 0) {
    set_error(ctx, "invalid bloom1 probe arguments", cudaSuccess);
    return 0;
  }

  if (!ensure_probe_buffers(ctx, xpoint_count, max_hits)) {
    return 0;
  }

  const size_t xpoint_bytes = (size_t)xpoint_count * 32;
  if (!check_cuda(ctx, "cudaMemcpy bloom1 xpoints",
                  cudaMemcpy(ctx->d_probe_xpoints, xpoints32, xpoint_bytes,
                             cudaMemcpyHostToDevice)) ||
      !check_cuda(ctx, "cudaMemset bloom1 hit count",
                  cudaMemset(ctx->d_probe_hit_count, 0, sizeof(uint32_t)))) {
    return 0;
  }

  const int block = 512;
  const int grid = (xpoint_count + block - 1) / block;
  probe_xpoints_kernel<<<grid, block>>>(ctx->d_bloom1, NULL, NULL, NULL, NULL, 0,
                                        ctx->d_probe_xpoints, xpoint_count,
                                        ctx->d_probe_hits, max_hits,
                                        ctx->d_probe_hit_count);
  if (!check_cuda(ctx, "probe_xpoints_bloom1_kernel", cudaGetLastError()) ||
      !check_cuda(ctx, "probe bloom1 synchronize", cudaDeviceSynchronize()) ||
      !check_cuda(ctx, "copy bloom1 hit count",
                  cudaMemcpy(hit_count, ctx->d_probe_hit_count, sizeof(uint32_t),
                             cudaMemcpyDeviceToHost))) {
    return 0;
  }

  const uint32_t copy_hits = (*hit_count < max_hits) ? *hit_count : max_hits;
  if (copy_hits > 0 &&
      !check_cuda(ctx, "copy bloom1 hits", cudaMemcpy(hits, ctx->d_probe_hits,
                                                      (size_t)copy_hits * sizeof(cuda_bsgs_hit),
                                                      cudaMemcpyDeviceToHost))) {
    return 0;
  }
  return 1;
}

extern "C" int cuda_bsgs_probe_xpoints_bloom1_workspace(cuda_bsgs_context *ctx,
                                                         cuda_bsgs_probe_workspace *workspace,
                                                         const uint8_t *xpoints32,
                                                         uint32_t xpoint_count,
                                                         cuda_bsgs_hit *hits,
                                                         uint32_t max_hits,
                                                         uint32_t *hit_count) {
  if (ctx == NULL || workspace == NULL || xpoints32 == NULL || hits == NULL ||
      hit_count == NULL || xpoint_count == 0) {
    set_error(ctx, "invalid workspace bloom1 probe arguments", cudaSuccess);
    return 0;
  }
  if (ctx->d_bloom1 == NULL || ctx->bloom_shard_count != CUDA_BSGS_SHARDS) {
    set_error(ctx, "Bloom1 must be uploaded before workspace probe", cudaSuccess);
    return 0;
  }
  if (!ensure_probe_workspace_buffers(ctx, workspace, xpoint_count, max_hits)) {
    return 0;
  }

  const size_t xpoint_bytes = (size_t)xpoint_count * 32;
  if (!check_cuda(ctx, "cudaMemcpyAsync workspace bloom1 xpoints",
                  cudaMemcpyAsync(workspace->d_xpoints, xpoints32, xpoint_bytes,
                                  cudaMemcpyHostToDevice, workspace->stream)) ||
      !check_cuda(ctx, "cudaMemsetAsync workspace bloom1 hit count",
                  cudaMemsetAsync(workspace->d_hit_count, 0, sizeof(uint32_t),
                                  workspace->stream))) {
    return 0;
  }

  const int block = 512;
  const int grid = (xpoint_count + block - 1) / block;
  probe_xpoints_kernel<<<grid, block, 0, workspace->stream>>>(
      ctx->d_bloom1, NULL, NULL, NULL, NULL, 0,
      workspace->d_xpoints, xpoint_count, workspace->d_hits, max_hits,
      workspace->d_hit_count);
  if (!check_cuda(ctx, "probe_xpoints_bloom1_workspace_kernel", cudaGetLastError()) ||
      !check_cuda(ctx, "copy workspace bloom1 hit count",
                  cudaMemcpyAsync(&workspace->h_hit_count, workspace->d_hit_count,
                                  sizeof(uint32_t), cudaMemcpyDeviceToHost,
                                  workspace->stream)) ||
      !check_cuda(ctx, "workspace bloom1 stream synchronize",
                  cudaStreamSynchronize(workspace->stream))) {
    return 0;
  }

  *hit_count = workspace->h_hit_count;
  const uint32_t copy_hits = (*hit_count < max_hits) ? *hit_count : max_hits;
  if (copy_hits > 0) {
    if (!check_cuda(ctx, "copy workspace bloom1 hits",
                    cudaMemcpyAsync(hits, workspace->d_hits,
                                    (size_t)copy_hits * sizeof(cuda_bsgs_hit),
                                    cudaMemcpyDeviceToHost, workspace->stream)) ||
        !check_cuda(ctx, "workspace bloom1 hit synchronize",
                    cudaStreamSynchronize(workspace->stream))) {
      return 0;
    }
  }
  return 1;
}

extern "C" int cuda_bsgs_probe_xpoints_bloom1_workspace_submit(cuda_bsgs_context *ctx,
                                                                cuda_bsgs_probe_workspace *workspace,
                                                                const uint8_t *xpoints32,
                                                                uint32_t xpoint_count,
                                                                uint32_t max_hits) {
  if (ctx == NULL || workspace == NULL || xpoints32 == NULL || xpoint_count == 0) {
    set_error(ctx, "invalid async workspace bloom1 submit arguments", cudaSuccess);
    return 0;
  }
  if (workspace->pending) {
    set_error(ctx, "workspace probe already pending", cudaSuccess);
    return 0;
  }
  if (ctx->d_bloom1 == NULL || ctx->bloom_shard_count != CUDA_BSGS_SHARDS) {
    set_error(ctx, "Bloom1 must be uploaded before async workspace probe", cudaSuccess);
    return 0;
  }
  if (!ensure_probe_workspace_buffers(ctx, workspace, xpoint_count, max_hits)) {
    return 0;
  }

  const size_t xpoint_bytes = (size_t)xpoint_count * 32;
  if (!check_cuda(ctx, "cudaMemcpyAsync submit bloom1 xpoints",
                  cudaMemcpyAsync(workspace->d_xpoints, xpoints32, xpoint_bytes,
                                  cudaMemcpyHostToDevice, workspace->stream)) ||
      !check_cuda(ctx, "cudaMemsetAsync submit bloom1 hit count",
                  cudaMemsetAsync(workspace->d_hit_count, 0, sizeof(uint32_t),
                                  workspace->stream))) {
    return 0;
  }

  const int block = 512;
  const int grid = (xpoint_count + block - 1) / block;
  probe_xpoints_kernel<<<grid, block, 0, workspace->stream>>>(
      ctx->d_bloom1, NULL, NULL, NULL, NULL, 0,
      workspace->d_xpoints, xpoint_count, workspace->d_hits, max_hits,
      workspace->d_hit_count);
  if (!check_cuda(ctx, "probe_xpoints_bloom1_submit_kernel", cudaGetLastError()) ||
      !check_cuda(ctx, "copy submit bloom1 hit count",
                  cudaMemcpyAsync(&workspace->h_hit_count, workspace->d_hit_count,
                                  sizeof(uint32_t), cudaMemcpyDeviceToHost,
                                  workspace->stream))) {
    return 0;
  }

  workspace->pending = 1;
  return 1;
}

extern "C" int cuda_bsgs_probe_xpoints_bloom1_workspace_finish(cuda_bsgs_context *ctx,
                                                                cuda_bsgs_probe_workspace *workspace,
                                                                cuda_bsgs_hit *hits,
                                                                uint32_t max_hits,
                                                                uint32_t *hit_count) {
  if (ctx == NULL || workspace == NULL || hits == NULL || hit_count == NULL) {
    set_error(ctx, "invalid async workspace bloom1 finish arguments", cudaSuccess);
    return 0;
  }
  if (!workspace->pending) {
    *hit_count = 0;
    return 1;
  }
  if (!check_cuda(ctx, "async workspace bloom1 stream synchronize",
                  cudaStreamSynchronize(workspace->stream))) {
    return 0;
  }

  *hit_count = workspace->h_hit_count;
  const uint32_t copy_hits = (*hit_count < max_hits) ? *hit_count : max_hits;
  if (copy_hits > 0) {
    if (!check_cuda(ctx, "copy async workspace bloom1 hits",
                    cudaMemcpyAsync(hits, workspace->d_hits,
                                    (size_t)copy_hits * sizeof(cuda_bsgs_hit),
                                    cudaMemcpyDeviceToHost, workspace->stream)) ||
        !check_cuda(ctx, "async workspace bloom1 hit synchronize",
                    cudaStreamSynchronize(workspace->stream))) {
      return 0;
    }
  }
  workspace->pending = 0;
  return 1;
}

extern "C" int cuda_bsgs_upload_search_points(cuda_bsgs_context *ctx,
                                               const cuda_bsgs_affine_point *GSn512,
                                               const cuda_bsgs_affine_point *_2GSn) {
  if (ctx == NULL || GSn512 == NULL || _2GSn == NULL) {
    set_error(ctx, "invalid search point upload arguments", cudaSuccess);
    return 0;
  }

  if (ctx->d_gsn == NULL &&
      !check_cuda(ctx, "cudaMalloc GSn", cudaMalloc((void **)&ctx->d_gsn,
                                                    CUDA_BSGS_GSN_COUNT * sizeof(cuda_bsgs_affine_point)))) {
    return 0;
  }
  if (ctx->d_2gsn == NULL &&
      !check_cuda(ctx, "cudaMalloc _2GSn", cudaMalloc((void **)&ctx->d_2gsn,
                                                      sizeof(cuda_bsgs_affine_point)))) {
    return 0;
  }
  if (ctx->d_center_offsets == NULL &&
      !check_cuda(ctx, "cudaMalloc center offsets",
                  cudaMalloc((void **)&ctx->d_center_offsets,
                             CUDA_BSGS_CENTER_OFFSET_COUNT * sizeof(cuda_bsgs_affine_point)))) {
    return 0;
  }
  if (ctx->d_center_next_seed == NULL &&
      !check_cuda(ctx, "cudaMalloc center next seed",
                  cudaMalloc((void **)&ctx->d_center_next_seed,
                             sizeof(cuda_bsgs_affine_point)))) {
    return 0;
  }
  if (!check_cuda(ctx, "cudaMemcpy GSn",
                  cudaMemcpy(ctx->d_gsn, GSn512,
                             CUDA_BSGS_GSN_COUNT * sizeof(cuda_bsgs_affine_point),
                             cudaMemcpyHostToDevice)) ||
      !check_cuda(ctx, "cudaMemcpy _2GSn",
                  cudaMemcpy(ctx->d_2gsn, _2GSn, sizeof(cuda_bsgs_affine_point),
                             cudaMemcpyHostToDevice))) {
    return 0;
  }

  generate_center_offsets_kernel<<<1, 1>>>(ctx->d_2gsn, ctx->d_center_offsets);
  if (!check_cuda(ctx, "generate_center_offsets_kernel", cudaGetLastError()) ||
      !check_cuda(ctx, "generate center offsets synchronize", cudaDeviceSynchronize())) {
    return 0;
  }

  ctx->search_points_uploaded = 1;
  return 1;
}

extern "C" int cuda_bsgs_upload_filter_points(cuda_bsgs_context *ctx,
                                               const cuda_bsgs_affine_point *original_points,
                                               uint32_t original_point_count,
                                               const cuda_bsgs_affine_point *amp2_32,
                                               const cuda_bsgs_affine_point *amp3_32) {
  if (ctx == NULL || original_points == NULL || original_point_count == 0 ||
      amp2_32 == NULL || amp3_32 == NULL) {
    set_error(ctx, "invalid filter point upload arguments", cudaSuccess);
    return 0;
  }

  if (ctx->d_original_points != NULL) {
    cudaFree(ctx->d_original_points);
    ctx->d_original_points = NULL;
  }
  if (ctx->d_amp2 == NULL &&
      !check_cuda(ctx, "cudaMalloc AMP2",
                  cudaMalloc((void **)&ctx->d_amp2, 32 * sizeof(cuda_bsgs_affine_point)))) {
    return 0;
  }
  if (ctx->d_amp3 == NULL &&
      !check_cuda(ctx, "cudaMalloc AMP3",
                  cudaMalloc((void **)&ctx->d_amp3, 32 * sizeof(cuda_bsgs_affine_point)))) {
    return 0;
  }

  const size_t original_bytes = (size_t)original_point_count * sizeof(cuda_bsgs_affine_point);
  if (!check_cuda(ctx, "cudaMalloc original points",
                  cudaMalloc((void **)&ctx->d_original_points, original_bytes)) ||
      !check_cuda(ctx, "cudaMemcpy original points",
                  cudaMemcpy(ctx->d_original_points, original_points, original_bytes,
                             cudaMemcpyHostToDevice)) ||
      !check_cuda(ctx, "cudaMemcpy AMP2",
                  cudaMemcpy(ctx->d_amp2, amp2_32, 32 * sizeof(cuda_bsgs_affine_point),
                             cudaMemcpyHostToDevice)) ||
      !check_cuda(ctx, "cudaMemcpy AMP3",
                  cudaMemcpy(ctx->d_amp3, amp3_32, 32 * sizeof(cuda_bsgs_affine_point),
                             cudaMemcpyHostToDevice))) {
    return 0;
  }

  ctx->original_point_count = original_point_count;
  return 1;
}

static int upload_scalar(cuda_bsgs_context *ctx,
                         cuda_bsgs_scalar **dst,
                         const cuda_bsgs_scalar *src,
                         const char *label) {
  if (*dst == NULL &&
      !check_cuda(ctx, label, cudaMalloc((void **)dst, sizeof(cuda_bsgs_scalar)))) {
    return 0;
  }
  return check_cuda(ctx, label, cudaMemcpy(*dst, src, sizeof(cuda_bsgs_scalar),
                                           cudaMemcpyHostToDevice));
}

extern "C" int cuda_bsgs_upload_filter_scalars(cuda_bsgs_context *ctx,
                                                const cuda_bsgs_scalar *m_double,
                                                const cuda_bsgs_scalar *m2_double,
                                                const cuda_bsgs_scalar *m3,
                                                const cuda_bsgs_scalar *m3_double,
                                                const cuda_bsgs_scalar *order) {
  if (ctx == NULL || m_double == NULL || m2_double == NULL ||
      m3 == NULL || m3_double == NULL || order == NULL) {
    set_error(ctx, "invalid filter scalar upload arguments", cudaSuccess);
    return 0;
  }

  ctx->h_m_double = *m_double;
  ctx->h_m2_double = *m2_double;
  ctx->h_m3 = *m3;
  ctx->h_m3_double = *m3_double;
  ctx->h_order = *order;

  return upload_scalar(ctx, &ctx->d_m_double, m_double, "cuda upload M double") &&
         upload_scalar(ctx, &ctx->d_m2_double, m2_double, "cuda upload M2 double") &&
         upload_scalar(ctx, &ctx->d_m3, m3, "cuda upload M3") &&
         upload_scalar(ctx, &ctx->d_m3_double, m3_double, "cuda upload M3 double") &&
         upload_scalar(ctx, &ctx->d_order, order, "cuda upload order");
}

extern "C" int cuda_bsgs_probe_center_batches(cuda_bsgs_context *ctx,
                                               const cuda_bsgs_center_work *works,
                                               uint32_t work_count,
                                               cuda_bsgs_center_hit *hits,
                                               uint32_t max_hits,
                                               uint32_t *hit_count) {
  if (ctx == NULL || works == NULL || hit_count == NULL || work_count == 0 ||
      (max_hits > 0 && hits == NULL)) {
    set_error(ctx, "invalid center probe arguments", cudaSuccess);
    return 0;
  }
  if (ctx->d_bloom1 == NULL || ctx->bloom_shard_count != CUDA_BSGS_SHARDS) {
    set_error(ctx, "Bloom1 must be uploaded with 256 shards before center probe", cudaSuccess);
    return 0;
  }
  if (!ctx->search_points_uploaded || ctx->d_gsn == NULL || ctx->d_2gsn == NULL) {
    set_error(ctx, "search points must be uploaded before center probe", cudaSuccess);
    return 0;
  }
  if (!ensure_center_buffers(ctx, work_count, max_hits)) {
    return 0;
  }

  const size_t work_bytes = (size_t)work_count * sizeof(cuda_bsgs_center_work);
  if (!check_cuda(ctx, "cudaMemcpy center works",
                  cudaMemcpy(ctx->d_center_works, works, work_bytes, cudaMemcpyHostToDevice)) ||
      !check_cuda(ctx, "cudaMemset center hit count",
                  cudaMemset(ctx->d_center_hit_count, 0, sizeof(uint32_t)))) {
    return 0;
  }

  const int block = 512;
  const int grid = (int)work_count;
  const size_t shared_bytes = 3 * CUDA_BSGS_CENTER_SCAN * sizeof(cuda_bsgs_u256);
  cudaFuncSetAttribute(probe_center_batches_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       (int)shared_bytes);
  probe_center_batches_kernel<<<grid, block, shared_bytes>>>(ctx->d_bloom1,
                                                             ctx->d_gsn,
                                                             ctx->d_2gsn,
                                                             ctx->d_center_works,
                                                             work_count,
                                                             ctx->d_center_hits,
                                                             max_hits,
                                                             ctx->d_center_hit_count);
  if (!check_cuda(ctx, "probe_center_batches_kernel", cudaGetLastError()) ||
      !check_cuda(ctx, "center probe synchronize", cudaDeviceSynchronize()) ||
      !check_cuda(ctx, "copy center hit count",
                  cudaMemcpy(hit_count, ctx->d_center_hit_count, sizeof(uint32_t),
                             cudaMemcpyDeviceToHost))) {
    return 0;
  }

  const uint32_t copy_hits = (*hit_count < max_hits) ? *hit_count : max_hits;
  if (copy_hits > 0 &&
      !check_cuda(ctx, "copy center hits",
                  cudaMemcpy(hits, ctx->d_center_hits,
                             (size_t)copy_hits * sizeof(cuda_bsgs_center_hit),
                             cudaMemcpyDeviceToHost))) {
    return 0;
  }

  return 1;
}

extern "C" int cuda_bsgs_probe_center_sequence(cuda_bsgs_context *ctx,
                                                const cuda_bsgs_affine_point *seed,
                                                uint64_t first_user_index,
                                                uint32_t work_count,
                                                cuda_bsgs_center_hit *hits,
                                                uint32_t max_hits,
                                                uint32_t *hit_count,
                                                cuda_bsgs_affine_point *next_seed) {
  if (ctx == NULL || seed == NULL || hit_count == NULL || work_count == 0 ||
      work_count > CUDA_BSGS_CENTER_SEQUENCE_MAX || (max_hits > 0 && hits == NULL)) {
    set_error(ctx, "invalid center sequence arguments", cudaSuccess);
    return 0;
  }
  if (ctx->d_bloom1 == NULL || ctx->bloom_shard_count != CUDA_BSGS_SHARDS) {
    set_error(ctx, "Bloom1 must be uploaded with 256 shards before center sequence", cudaSuccess);
    return 0;
  }
  if (!ctx->search_points_uploaded || ctx->d_gsn == NULL || ctx->d_2gsn == NULL ||
      ctx->d_center_offsets == NULL) {
    set_error(ctx, "search points must be uploaded before center sequence", cudaSuccess);
    return 0;
  }
  if (!ensure_center_buffers(ctx, work_count, max_hits)) {
    return 0;
  }

  if (!check_cuda(ctx, "cudaMemset center sequence hit count",
                  cudaMemset(ctx->d_center_hit_count, 0, sizeof(uint32_t)))) {
    return 0;
  }

  generate_center_sequence_kernel<<<(int)work_count, 1>>>(*seed,
                                                          first_user_index,
                                                          ctx->d_center_offsets,
                                                          ctx->d_center_works,
                                                          ctx->d_center_next_seed,
                                                          work_count);
  if (!check_cuda(ctx, "generate_center_sequence_kernel", cudaGetLastError())) {
    return 0;
  }

  const int block = 512;
  const int grid = (int)work_count;
  const size_t shared_bytes = 3 * CUDA_BSGS_CENTER_SCAN * sizeof(cuda_bsgs_u256);
  cudaFuncSetAttribute(probe_center_batches_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       (int)shared_bytes);
  probe_center_batches_kernel<<<grid, block, shared_bytes>>>(ctx->d_bloom1,
                                                             ctx->d_gsn,
                                                             ctx->d_2gsn,
                                                             ctx->d_center_works,
                                                             work_count,
                                                             ctx->d_center_hits,
                                                             max_hits,
                                                             ctx->d_center_hit_count);
  if (!check_cuda(ctx, "probe_center_sequence kernel", cudaGetLastError()) ||
      !check_cuda(ctx, "center sequence synchronize", cudaDeviceSynchronize()) ||
      !check_cuda(ctx, "copy center sequence hit count",
                  cudaMemcpy(hit_count, ctx->d_center_hit_count, sizeof(uint32_t),
                             cudaMemcpyDeviceToHost))) {
    return 0;
  }

  const uint32_t copy_hits = (*hit_count < max_hits) ? *hit_count : max_hits;
  if (copy_hits > 0 &&
      !check_cuda(ctx, "copy center sequence hits",
                  cudaMemcpy(hits, ctx->d_center_hits,
                             (size_t)copy_hits * sizeof(cuda_bsgs_center_hit),
                             cudaMemcpyDeviceToHost))) {
    return 0;
  }

  if (next_seed != NULL &&
      !check_cuda(ctx, "copy center sequence next seed",
                  cudaMemcpy(next_seed, ctx->d_center_next_seed,
                             sizeof(cuda_bsgs_affine_point), cudaMemcpyDeviceToHost))) {
    return 0;
  }

  return 1;
}

extern "C" int cuda_bsgs_probe_center_sequence_workspace(cuda_bsgs_context *ctx,
                                                          cuda_bsgs_center_workspace *workspace,
                                                          const cuda_bsgs_affine_point *seed,
                                                          uint64_t first_user_index,
                                                          uint32_t work_count,
                                                          cuda_bsgs_center_hit *hits,
                                                          uint32_t max_hits,
                                                          uint32_t *hit_count,
                                                          cuda_bsgs_affine_point *next_seed) {
  if (ctx == NULL || workspace == NULL || seed == NULL || hit_count == NULL ||
      work_count == 0 || work_count > CUDA_BSGS_CENTER_SEQUENCE_MAX ||
      (max_hits > 0 && hits == NULL)) {
    set_error(ctx, "invalid center workspace sequence arguments", cudaSuccess);
    return 0;
  }
  if (ctx->d_bloom1 == NULL || ctx->bloom_shard_count != CUDA_BSGS_SHARDS) {
    set_error(ctx, "Bloom1 must be uploaded with 256 shards before center workspace sequence", cudaSuccess);
    return 0;
  }
  if (!ctx->search_points_uploaded || ctx->d_gsn == NULL || ctx->d_2gsn == NULL ||
      ctx->d_center_offsets == NULL) {
    set_error(ctx, "search points must be uploaded before center workspace sequence", cudaSuccess);
    return 0;
  }
  if (!ensure_center_workspace_buffers(ctx, workspace, work_count, max_hits)) {
    return 0;
  }

  if (!check_cuda(ctx, "cudaMemsetAsync center workspace hit count",
                  cudaMemsetAsync(workspace->d_hit_count, 0, sizeof(uint32_t),
                                  workspace->stream))) {
    return 0;
  }

  generate_center_sequence_kernel<<<(int)work_count, 1, 0, workspace->stream>>>(
      *seed, first_user_index, ctx->d_center_offsets, workspace->d_works,
      workspace->d_next_seed, work_count);
  if (!check_cuda(ctx, "generate_center_sequence_workspace_kernel", cudaGetLastError())) {
    return 0;
  }

  const int block = 512;
  const int grid = (int)work_count;
  const size_t shared_bytes = 3 * CUDA_BSGS_CENTER_SCAN * sizeof(cuda_bsgs_u256);
  cudaFuncSetAttribute(probe_center_batches_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       (int)shared_bytes);
  probe_center_batches_kernel<<<grid, block, shared_bytes, workspace->stream>>>(
      ctx->d_bloom1, ctx->d_gsn, ctx->d_2gsn, workspace->d_works,
      work_count, workspace->d_hits, max_hits, workspace->d_hit_count);
  if (!check_cuda(ctx, "probe_center_workspace_sequence kernel", cudaGetLastError()) ||
      !check_cuda(ctx, "copy center workspace sequence hit count",
                  cudaMemcpyAsync(hit_count, workspace->d_hit_count, sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost, workspace->stream))) {
    return 0;
  }
  if (next_seed != NULL &&
      !check_cuda(ctx, "copy center workspace sequence next seed",
                  cudaMemcpyAsync(next_seed, workspace->d_next_seed,
                                  sizeof(cuda_bsgs_affine_point),
                                  cudaMemcpyDeviceToHost, workspace->stream))) {
    return 0;
  }
  if (!check_cuda(ctx, "center workspace sequence synchronize",
                  cudaStreamSynchronize(workspace->stream))) {
    return 0;
  }

  const uint32_t copy_hits = (*hit_count < max_hits) ? *hit_count : max_hits;
  if (copy_hits > 0 &&
      (!check_cuda(ctx, "copy center workspace sequence hits",
                   cudaMemcpyAsync(hits, workspace->d_hits,
                                   (size_t)copy_hits * sizeof(cuda_bsgs_center_hit),
                                   cudaMemcpyDeviceToHost, workspace->stream)) ||
       !check_cuda(ctx, "center workspace hits synchronize",
                   cudaStreamSynchronize(workspace->stream)))) {
    return 0;
  }

  return 1;
}

extern "C" int cuda_bsgs_filter_center_hits(cuda_bsgs_context *ctx,
                                             const cuda_bsgs_scalar *base_key,
                                             uint32_t k_index,
                                             const cuda_bsgs_center_hit *hits_in,
                                             uint32_t hit_count_in,
                                             cuda_bsgs_filter_hit *hits_out,
                                             uint32_t max_hits_out,
                                             uint32_t *hit_count_out) {
  if (ctx == NULL || base_key == NULL || hits_in == NULL || hit_count_out == NULL ||
      hit_count_in == 0 || (max_hits_out > 0 && hits_out == NULL)) {
    set_error(ctx, "invalid filter center hit arguments", cudaSuccess);
    return 0;
  }
  if (ctx->d_bloom2 == NULL || ctx->d_bloom3 == NULL ||
      ctx->table_values_soa == NULL || ctx->table_indices == NULL || ctx->table_count == 0) {
    set_error(ctx, "native bloom/table data must be uploaded before filter", cudaSuccess);
    return 0;
  }
  if (ctx->d_original_points == NULL || ctx->d_amp2 == NULL || ctx->d_amp3 == NULL ||
      ctx->original_point_count == 0) {
    set_error(ctx, "filter points must be uploaded before filter", cudaSuccess);
    return 0;
  }
  if (k_index >= ctx->original_point_count) {
    set_error(ctx, "filter k index out of range", cudaSuccess);
    return 0;
  }
  if (!ensure_filter_buffers(ctx, hit_count_in, max_hits_out)) {
    return 0;
  }

  const size_t input_bytes = (size_t)hit_count_in * sizeof(cuda_bsgs_center_hit);
  if (!check_cuda(ctx, "cudaMemcpy filter input hits",
                  cudaMemcpy(ctx->d_filter_input_hits, hits_in, input_bytes,
                             cudaMemcpyHostToDevice)) ||
      !check_cuda(ctx, "cudaMemset filter hit count",
                  cudaMemset(ctx->d_filter_hit_count, 0, sizeof(uint32_t)))) {
    return 0;
  }

  const int block = 128;
  const int grid = (hit_count_in + block - 1) / block;
  filter_center_hits_kernel<<<grid, block>>>(ctx->d_bloom2,
                                             ctx->d_bloom3,
                                             ctx->table_values_soa,
                                             ctx->table_indices,
                                             ctx->table_count,
                                             ctx->d_original_points,
                                             ctx->original_point_count,
                                             ctx->d_amp2,
                                             ctx->d_amp3,
                                             *base_key,
                                             ctx->h_m_double,
                                             ctx->h_m2_double,
                                             k_index,
                                             ctx->d_filter_input_hits,
                                             hit_count_in,
                                             ctx->d_filter_hits,
                                             max_hits_out,
                                             ctx->d_filter_hit_count);
  if (!check_cuda(ctx, "filter_center_hits_kernel", cudaGetLastError()) ||
      !check_cuda(ctx, "filter center hits synchronize", cudaDeviceSynchronize()) ||
      !check_cuda(ctx, "copy filter hit count",
                  cudaMemcpy(hit_count_out, ctx->d_filter_hit_count, sizeof(uint32_t),
                             cudaMemcpyDeviceToHost))) {
    return 0;
  }

  const uint32_t copy_hits = (*hit_count_out < max_hits_out) ? *hit_count_out : max_hits_out;
  if (copy_hits > 0 &&
      !check_cuda(ctx, "copy filter hits",
                  cudaMemcpy(hits_out, ctx->d_filter_hits,
                             (size_t)copy_hits * sizeof(cuda_bsgs_filter_hit),
                             cudaMemcpyDeviceToHost))) {
    return 0;
  }
  return 1;
}

extern "C" const char *cuda_bsgs_last_error(cuda_bsgs_context *ctx) {
  if (ctx == NULL) {
    return "null cuda_bsgs_context";
  }
  return ctx->last_error;
}
