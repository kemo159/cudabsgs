#ifndef KEYHUNT_CUDA_BSGS_H
#define KEYHUNT_CUDA_BSGS_H

#include <stddef.h>
#include <stdint.h>
#include "../bloom/bloom.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cuda_bsgs_context cuda_bsgs_context;
typedef struct cuda_bsgs_probe_workspace cuda_bsgs_probe_workspace;
typedef struct cuda_bsgs_center_workspace cuda_bsgs_center_workspace;

typedef struct cuda_bsgs_xvalue {
  uint8_t value[6];
  uint64_t index;
} cuda_bsgs_xvalue;

typedef struct cuda_bsgs_hit {
  uint32_t xpoint_index;
  uint64_t table_index;
  uint8_t stage;
  uint8_t reserved[7];
} cuda_bsgs_hit;

typedef struct cuda_bsgs_u256 {
  uint64_t v[4];
} cuda_bsgs_u256;

typedef struct cuda_bsgs_scalar {
  uint64_t v[4];
} cuda_bsgs_scalar;

typedef struct cuda_bsgs_affine_point {
  cuda_bsgs_u256 x;
  cuda_bsgs_u256 y;
  uint32_t infinity;
  uint32_t reserved;
} cuda_bsgs_affine_point;

typedef struct cuda_bsgs_center_work {
  cuda_bsgs_affine_point startP;
  uint64_t user_index;
} cuda_bsgs_center_work;

typedef struct cuda_bsgs_center_hit {
  uint32_t work_index;
  uint32_t point_index;
  uint64_t table_index;
  uint8_t x[32];
  uint8_t shard;
  uint8_t stage;
  uint8_t reserved[7];
} cuda_bsgs_center_hit;

typedef struct cuda_bsgs_filter_hit {
  uint32_t work_index;
  uint32_t point_index;
  uint32_t second_index;
  uint32_t third_index;
  uint64_t table_index;
  uint8_t stage;
  uint8_t reserved[7];
} cuda_bsgs_filter_hit;

int cuda_bsgs_available(void);
int cuda_bsgs_create(cuda_bsgs_context **ctx);
void cuda_bsgs_destroy(cuda_bsgs_context *ctx);
int cuda_bsgs_host_alloc(void **ptr, size_t bytes);
void cuda_bsgs_host_free(void *ptr);

int cuda_bsgs_upload_blooms(cuda_bsgs_context *ctx,
                            const struct bloom *bloom1,
                            const struct bloom *bloom2,
                            const struct bloom *bloom3,
                            size_t shard_count);

int cuda_bsgs_upload_table(cuda_bsgs_context *ctx,
                           const cuda_bsgs_xvalue *table,
                           uint64_t table_count);

int cuda_bsgs_load_native_cache(cuda_bsgs_context *ctx,
                                const char *filename);

int cuda_bsgs_save_native_cache(cuda_bsgs_context *ctx,
                                const char *filename);

int cuda_bsgs_probe_xpoints(cuda_bsgs_context *ctx,
                            const uint8_t *xpoints32,
                            uint32_t xpoint_count,
                            cuda_bsgs_hit *hits,
                            uint32_t max_hits,
                            uint32_t *hit_count);

int cuda_bsgs_probe_xpoints_bloom1(cuda_bsgs_context *ctx,
                                   const uint8_t *xpoints32,
                                   uint32_t xpoint_count,
                                   cuda_bsgs_hit *hits,
                                   uint32_t max_hits,
                                   uint32_t *hit_count);

int cuda_bsgs_probe_workspace_create(cuda_bsgs_probe_workspace **workspace);
void cuda_bsgs_probe_workspace_destroy(cuda_bsgs_probe_workspace *workspace);
int cuda_bsgs_probe_xpoints_bloom1_workspace(cuda_bsgs_context *ctx,
                                             cuda_bsgs_probe_workspace *workspace,
                                             const uint8_t *xpoints32,
                                             uint32_t xpoint_count,
                                             cuda_bsgs_hit *hits,
                                             uint32_t max_hits,
                                             uint32_t *hit_count);
int cuda_bsgs_probe_xpoints_bloom1_workspace_submit(cuda_bsgs_context *ctx,
                                                    cuda_bsgs_probe_workspace *workspace,
                                                    const uint8_t *xpoints32,
                                                    uint32_t xpoint_count,
                                                    uint32_t max_hits);
int cuda_bsgs_probe_xpoints_bloom1_workspace_finish(cuda_bsgs_context *ctx,
                                                    cuda_bsgs_probe_workspace *workspace,
                                                    cuda_bsgs_hit *hits,
                                                    uint32_t max_hits,
                                                    uint32_t *hit_count);

int cuda_bsgs_upload_search_points(cuda_bsgs_context *ctx,
                                   const cuda_bsgs_affine_point *GSn512,
                                   const cuda_bsgs_affine_point *_2GSn);

int cuda_bsgs_upload_filter_points(cuda_bsgs_context *ctx,
                                   const cuda_bsgs_affine_point *original_points,
                                   uint32_t original_point_count,
                                   const cuda_bsgs_affine_point *amp2_32,
                                   const cuda_bsgs_affine_point *amp3_32);

int cuda_bsgs_upload_filter_scalars(cuda_bsgs_context *ctx,
                                    const cuda_bsgs_scalar *m_double,
                                    const cuda_bsgs_scalar *m2_double,
                                    const cuda_bsgs_scalar *m3,
                                    const cuda_bsgs_scalar *m3_double,
                                    const cuda_bsgs_scalar *order);

int cuda_bsgs_probe_center_batches(cuda_bsgs_context *ctx,
                                   const cuda_bsgs_center_work *works,
                                   uint32_t work_count,
                                   cuda_bsgs_center_hit *hits,
                                   uint32_t max_hits,
                                   uint32_t *hit_count);

int cuda_bsgs_filter_center_hits(cuda_bsgs_context *ctx,
                                 const cuda_bsgs_scalar *base_key,
                                 uint32_t k_index,
                                 const cuda_bsgs_center_hit *hits_in,
                                 uint32_t hit_count_in,
                                 cuda_bsgs_filter_hit *hits_out,
                                 uint32_t max_hits_out,
                                 uint32_t *hit_count_out);

int cuda_bsgs_probe_center_sequence(cuda_bsgs_context *ctx,
                                    const cuda_bsgs_affine_point *seed,
                                    uint64_t first_user_index,
                                    uint32_t work_count,
                                    cuda_bsgs_center_hit *hits,
                                    uint32_t max_hits,
                                    uint32_t *hit_count,
                                    cuda_bsgs_affine_point *next_seed);
int cuda_bsgs_center_workspace_create(cuda_bsgs_center_workspace **workspace);
void cuda_bsgs_center_workspace_destroy(cuda_bsgs_center_workspace *workspace);
int cuda_bsgs_probe_center_sequence_workspace(cuda_bsgs_context *ctx,
                                              cuda_bsgs_center_workspace *workspace,
                                              const cuda_bsgs_affine_point *seed,
                                              uint64_t first_user_index,
                                              uint32_t work_count,
                                              cuda_bsgs_center_hit *hits,
                                              uint32_t max_hits,
                                              uint32_t *hit_count,
                                              cuda_bsgs_affine_point *next_seed);

const char *cuda_bsgs_last_error(cuda_bsgs_context *ctx);

#ifdef __cplusplus
}
#endif

#endif
