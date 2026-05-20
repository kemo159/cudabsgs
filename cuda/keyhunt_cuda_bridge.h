#ifndef KEYHUNT_CUDA_BRIDGE_H
#define KEYHUNT_CUDA_BRIDGE_H

#include <stdint.h>
#include "../bloom/bloom.h"
#include "bsgs_cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

int keyhunt_cuda_bsgs_prepare(const struct bloom *bloom1,
                              const struct bloom *bloom2,
                              const struct bloom *bloom3,
                              const void *bPtable,
                              uint64_t table_count);

int keyhunt_cuda_bsgs_upload_search_points(const cuda_bsgs_affine_point *GSn512,
                                           const cuda_bsgs_affine_point *_2GSn);

int keyhunt_cuda_bsgs_upload_filter_points(const cuda_bsgs_affine_point *original_points,
                                           uint32_t original_point_count,
                                           const cuda_bsgs_affine_point *amp2_32,
                                           const cuda_bsgs_affine_point *amp3_32);

int keyhunt_cuda_bsgs_upload_filter_scalars(const cuda_bsgs_scalar *m_double,
                                            const cuda_bsgs_scalar *m2_double,
                                            const cuda_bsgs_scalar *m3,
                                            const cuda_bsgs_scalar *m3_double,
                                            const cuda_bsgs_scalar *order);

int keyhunt_cuda_bsgs_probe_centers(const cuda_bsgs_center_work *works,
                                    uint32_t work_count,
                                    cuda_bsgs_center_hit *hits,
                                    uint32_t max_hits,
                                    uint32_t *hit_count);

int keyhunt_cuda_bsgs_probe_xpoints_bloom1(const uint8_t *xpoints32,
                                           uint32_t xpoint_count,
                                           cuda_bsgs_hit *hits,
                                           uint32_t max_hits,
                                           uint32_t *hit_count);

int keyhunt_cuda_bsgs_probe_workspace_create(cuda_bsgs_probe_workspace **workspace);
void keyhunt_cuda_bsgs_probe_workspace_destroy(cuda_bsgs_probe_workspace *workspace);
int keyhunt_cuda_bsgs_probe_xpoints_bloom1_workspace(cuda_bsgs_probe_workspace *workspace,
                                                     const uint8_t *xpoints32,
                                                     uint32_t xpoint_count,
                                                     cuda_bsgs_hit *hits,
                                                     uint32_t max_hits,
                                                     uint32_t *hit_count);
int keyhunt_cuda_bsgs_probe_xpoints_bloom1_submit(cuda_bsgs_probe_workspace *workspace,
                                                  const uint8_t *xpoints32,
                                                  uint32_t xpoint_count,
                                                  uint32_t max_hits);
int keyhunt_cuda_bsgs_probe_xpoints_bloom1_finish(cuda_bsgs_probe_workspace *workspace,
                                                  cuda_bsgs_hit *hits,
                                                  uint32_t max_hits,
                                                  uint32_t *hit_count);

int keyhunt_cuda_bsgs_filter_center_hits(const cuda_bsgs_scalar *base_key,
                                         uint32_t k_index,
                                         const cuda_bsgs_center_hit *hits_in,
                                         uint32_t hit_count_in,
                                         cuda_bsgs_filter_hit *hits_out,
                                         uint32_t max_hits_out,
                                         uint32_t *hit_count_out);

int keyhunt_cuda_bsgs_probe_center_sequence(const cuda_bsgs_affine_point *seed,
                                            uint64_t first_user_index,
                                            uint32_t work_count,
                                            cuda_bsgs_center_hit *hits,
                                            uint32_t max_hits,
                                            uint32_t *hit_count,
                                            cuda_bsgs_affine_point *next_seed);
int keyhunt_cuda_bsgs_center_workspace_create(cuda_bsgs_center_workspace **workspace);
void keyhunt_cuda_bsgs_center_workspace_destroy(cuda_bsgs_center_workspace *workspace);
int keyhunt_cuda_bsgs_probe_center_sequence_workspace(cuda_bsgs_center_workspace *workspace,
                                                      const cuda_bsgs_affine_point *seed,
                                                      uint64_t first_user_index,
                                                      uint32_t work_count,
                                                      cuda_bsgs_center_hit *hits,
                                                      uint32_t max_hits,
                                                      uint32_t *hit_count,
                                                      cuda_bsgs_affine_point *next_seed);

void keyhunt_cuda_bsgs_release(void);
const char *keyhunt_cuda_last_error(void);
int keyhunt_cuda_host_alloc(void **ptr, size_t bytes);
void keyhunt_cuda_host_free(void *ptr);

#ifdef __cplusplus
}
#endif

#endif
