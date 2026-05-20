#include "keyhunt_cuda_bridge.h"

#include <mutex>
#include <stdio.h>
#include <string.h>

static cuda_bsgs_context *g_cuda_ctx = nullptr;
static char g_cuda_error[256];
static std::mutex g_cuda_mutex;

static void set_bridge_error(const char *msg) {
  snprintf(g_cuda_error, sizeof(g_cuda_error), "%s", msg);
}

extern "C" int keyhunt_cuda_host_alloc(void **ptr, size_t bytes) {
  if (!cuda_bsgs_host_alloc(ptr, bytes)) {
    set_bridge_error("CUDA pinned host allocation failed");
    return 0;
  }
  return 1;
}

extern "C" void keyhunt_cuda_host_free(void *ptr) {
  cuda_bsgs_host_free(ptr);
}

static void make_native_cache_name(char *dst,
                                   size_t dst_size,
                                   const struct bloom *bloom1,
                                   const struct bloom *bloom2,
                                   const struct bloom *bloom3,
                                   uint64_t table_count) {
  snprintf(dst, dst_size,
           "keyhunt_bsgs_cuda_%llu_%llu_%llu_%llu.gcache",
           (unsigned long long)(bloom1 != nullptr ? bloom1[0].bytes : 0ULL),
           (unsigned long long)(bloom2 != nullptr ? bloom2[0].bytes : 0ULL),
           (unsigned long long)(bloom3 != nullptr ? bloom3[0].bytes : 0ULL),
           (unsigned long long)table_count);
}

extern "C" int keyhunt_cuda_bsgs_prepare(const struct bloom *bloom1,
                                          const struct bloom *bloom2,
                                          const struct bloom *bloom3,
                                          const void *bPtable,
                                          uint64_t table_count) {
  std::lock_guard<std::mutex> lock(g_cuda_mutex);
  g_cuda_error[0] = 0;

  if (!cuda_bsgs_available()) {
    set_bridge_error("CUDA device not available");
    return 0;
  }

  if (g_cuda_ctx != nullptr) {
    cuda_bsgs_destroy(g_cuda_ctx);
    g_cuda_ctx = nullptr;
  }

  if (!cuda_bsgs_create(&g_cuda_ctx)) {
    set_bridge_error("cuda_bsgs_create failed");
    return 0;
  }

  char native_cache[256];
  make_native_cache_name(native_cache, sizeof(native_cache), bloom1, bloom2, bloom3, table_count);
  printf("[+] CUDA native BSGS cache %s ", native_cache);
  fflush(stdout);
  if (cuda_bsgs_load_native_cache(g_cuda_ctx, native_cache)) {
    printf("loaded\n");
    return 1;
  }
  printf("not found, building\n");

  if (!cuda_bsgs_upload_blooms(g_cuda_ctx, bloom1, bloom2, bloom3, 256)) {
    snprintf(g_cuda_error, sizeof(g_cuda_error), "%s", cuda_bsgs_last_error(g_cuda_ctx));
    return 0;
  }

  if (!cuda_bsgs_upload_table(g_cuda_ctx,
                              static_cast<const cuda_bsgs_xvalue *>(bPtable),
                              table_count)) {
    snprintf(g_cuda_error, sizeof(g_cuda_error), "%s", cuda_bsgs_last_error(g_cuda_ctx));
    return 0;
  }

  if (!cuda_bsgs_save_native_cache(g_cuda_ctx, native_cache)) {
    snprintf(g_cuda_error, sizeof(g_cuda_error), "%s", cuda_bsgs_last_error(g_cuda_ctx));
    return 0;
  }
  printf("[+] CUDA native BSGS cache written to %s\n", native_cache);

  return 1;
}

extern "C" int keyhunt_cuda_bsgs_upload_search_points(const cuda_bsgs_affine_point *GSn512,
                                                       const cuda_bsgs_affine_point *_2GSn) {
  std::lock_guard<std::mutex> lock(g_cuda_mutex);
  g_cuda_error[0] = 0;

  if (g_cuda_ctx == nullptr) {
    set_bridge_error("CUDA BSGS context is not prepared");
    return 0;
  }

  if (!cuda_bsgs_upload_search_points(g_cuda_ctx, GSn512, _2GSn)) {
    snprintf(g_cuda_error, sizeof(g_cuda_error), "%s", cuda_bsgs_last_error(g_cuda_ctx));
    return 0;
  }

  return 1;
}

extern "C" int keyhunt_cuda_bsgs_upload_filter_points(const cuda_bsgs_affine_point *original_points,
                                                       uint32_t original_point_count,
                                                       const cuda_bsgs_affine_point *amp2_32,
                                                       const cuda_bsgs_affine_point *amp3_32) {
  std::lock_guard<std::mutex> lock(g_cuda_mutex);
  g_cuda_error[0] = 0;

  if (g_cuda_ctx == nullptr) {
    set_bridge_error("CUDA BSGS context is not prepared");
    return 0;
  }

  if (!cuda_bsgs_upload_filter_points(g_cuda_ctx, original_points, original_point_count,
                                      amp2_32, amp3_32)) {
    snprintf(g_cuda_error, sizeof(g_cuda_error), "%s", cuda_bsgs_last_error(g_cuda_ctx));
    return 0;
  }

  return 1;
}

extern "C" int keyhunt_cuda_bsgs_upload_filter_scalars(const cuda_bsgs_scalar *m_double,
                                                        const cuda_bsgs_scalar *m2_double,
                                                        const cuda_bsgs_scalar *m3,
                                                        const cuda_bsgs_scalar *m3_double,
                                                        const cuda_bsgs_scalar *order) {
  std::lock_guard<std::mutex> lock(g_cuda_mutex);
  g_cuda_error[0] = 0;

  if (g_cuda_ctx == nullptr) {
    set_bridge_error("CUDA BSGS context is not prepared");
    return 0;
  }

  if (!cuda_bsgs_upload_filter_scalars(g_cuda_ctx, m_double, m2_double, m3, m3_double, order)) {
    snprintf(g_cuda_error, sizeof(g_cuda_error), "%s", cuda_bsgs_last_error(g_cuda_ctx));
    return 0;
  }

  return 1;
}

extern "C" int keyhunt_cuda_bsgs_probe_centers(const cuda_bsgs_center_work *works,
                                                uint32_t work_count,
                                                cuda_bsgs_center_hit *hits,
                                                uint32_t max_hits,
                                                uint32_t *hit_count) {
  std::lock_guard<std::mutex> lock(g_cuda_mutex);
  g_cuda_error[0] = 0;

  if (g_cuda_ctx == nullptr) {
    set_bridge_error("CUDA BSGS context is not prepared");
    return 0;
  }

  if (!cuda_bsgs_probe_center_batches(g_cuda_ctx, works, work_count, hits, max_hits, hit_count)) {
    snprintf(g_cuda_error, sizeof(g_cuda_error), "%s", cuda_bsgs_last_error(g_cuda_ctx));
    return 0;
  }

  return 1;
}

extern "C" int keyhunt_cuda_bsgs_probe_xpoints_bloom1(const uint8_t *xpoints32,
                                                       uint32_t xpoint_count,
                                                       cuda_bsgs_hit *hits,
                                                       uint32_t max_hits,
                                                       uint32_t *hit_count) {
  std::lock_guard<std::mutex> lock(g_cuda_mutex);
  g_cuda_error[0] = 0;

  if (g_cuda_ctx == nullptr) {
    set_bridge_error("CUDA BSGS context is not prepared");
    return 0;
  }

  if (!cuda_bsgs_probe_xpoints_bloom1(g_cuda_ctx, xpoints32, xpoint_count,
                                      hits, max_hits, hit_count)) {
    snprintf(g_cuda_error, sizeof(g_cuda_error), "%s", cuda_bsgs_last_error(g_cuda_ctx));
    return 0;
  }

  return 1;
}

extern "C" int keyhunt_cuda_bsgs_probe_workspace_create(cuda_bsgs_probe_workspace **workspace) {
  std::lock_guard<std::mutex> lock(g_cuda_mutex);
  g_cuda_error[0] = 0;

  if (g_cuda_ctx == nullptr) {
    set_bridge_error("CUDA BSGS context is not prepared");
    return 0;
  }
  if (!cuda_bsgs_probe_workspace_create(workspace)) {
    set_bridge_error("CUDA probe workspace create failed");
    return 0;
  }
  return 1;
}

extern "C" void keyhunt_cuda_bsgs_probe_workspace_destroy(cuda_bsgs_probe_workspace *workspace) {
  cuda_bsgs_probe_workspace_destroy(workspace);
}

extern "C" int keyhunt_cuda_bsgs_probe_xpoints_bloom1_workspace(cuda_bsgs_probe_workspace *workspace,
                                                                 const uint8_t *xpoints32,
                                                                 uint32_t xpoint_count,
                                                                 cuda_bsgs_hit *hits,
                                                                 uint32_t max_hits,
                                                                 uint32_t *hit_count) {
  cuda_bsgs_context *ctx = g_cuda_ctx;
  if (ctx == nullptr) {
    set_bridge_error("CUDA BSGS context is not prepared");
    return 0;
  }
  if (!cuda_bsgs_probe_xpoints_bloom1_workspace(ctx, workspace, xpoints32,
                                                xpoint_count, hits, max_hits,
                                                hit_count)) {
    snprintf(g_cuda_error, sizeof(g_cuda_error), "%s", cuda_bsgs_last_error(ctx));
    return 0;
  }
  return 1;
}

extern "C" int keyhunt_cuda_bsgs_probe_xpoints_bloom1_submit(cuda_bsgs_probe_workspace *workspace,
                                                              const uint8_t *xpoints32,
                                                              uint32_t xpoint_count,
                                                              uint32_t max_hits) {
  cuda_bsgs_context *ctx = g_cuda_ctx;
  if (ctx == nullptr) {
    set_bridge_error("CUDA BSGS context is not prepared");
    return 0;
  }
  if (!cuda_bsgs_probe_xpoints_bloom1_workspace_submit(ctx, workspace, xpoints32,
                                                       xpoint_count, max_hits)) {
    snprintf(g_cuda_error, sizeof(g_cuda_error), "%s", cuda_bsgs_last_error(ctx));
    return 0;
  }
  return 1;
}

extern "C" int keyhunt_cuda_bsgs_probe_xpoints_bloom1_finish(cuda_bsgs_probe_workspace *workspace,
                                                              cuda_bsgs_hit *hits,
                                                              uint32_t max_hits,
                                                              uint32_t *hit_count) {
  cuda_bsgs_context *ctx = g_cuda_ctx;
  if (ctx == nullptr) {
    set_bridge_error("CUDA BSGS context is not prepared");
    return 0;
  }
  if (!cuda_bsgs_probe_xpoints_bloom1_workspace_finish(ctx, workspace, hits,
                                                       max_hits, hit_count)) {
    snprintf(g_cuda_error, sizeof(g_cuda_error), "%s", cuda_bsgs_last_error(ctx));
    return 0;
  }
  return 1;
}

extern "C" int keyhunt_cuda_bsgs_filter_center_hits(const cuda_bsgs_scalar *base_key,
                                                     uint32_t k_index,
                                                     const cuda_bsgs_center_hit *hits_in,
                                                     uint32_t hit_count_in,
                                                     cuda_bsgs_filter_hit *hits_out,
                                                     uint32_t max_hits_out,
                                                     uint32_t *hit_count_out) {
  std::lock_guard<std::mutex> lock(g_cuda_mutex);
  g_cuda_error[0] = 0;

  if (g_cuda_ctx == nullptr) {
    set_bridge_error("CUDA BSGS context is not prepared");
    return 0;
  }

  if (!cuda_bsgs_filter_center_hits(g_cuda_ctx, base_key, k_index, hits_in, hit_count_in,
                                    hits_out, max_hits_out, hit_count_out)) {
    snprintf(g_cuda_error, sizeof(g_cuda_error), "%s", cuda_bsgs_last_error(g_cuda_ctx));
    return 0;
  }

  return 1;
}

extern "C" int keyhunt_cuda_bsgs_probe_center_sequence(const cuda_bsgs_affine_point *seed,
                                                        uint64_t first_user_index,
                                                        uint32_t work_count,
                                                        cuda_bsgs_center_hit *hits,
                                                        uint32_t max_hits,
                                                        uint32_t *hit_count,
                                                        cuda_bsgs_affine_point *next_seed) {
  std::lock_guard<std::mutex> lock(g_cuda_mutex);
  g_cuda_error[0] = 0;

  if (g_cuda_ctx == nullptr) {
    set_bridge_error("CUDA BSGS context is not prepared");
    return 0;
  }

  if (!cuda_bsgs_probe_center_sequence(g_cuda_ctx, seed, first_user_index,
                                       work_count, hits, max_hits, hit_count, next_seed)) {
    snprintf(g_cuda_error, sizeof(g_cuda_error), "%s", cuda_bsgs_last_error(g_cuda_ctx));
    return 0;
  }

  return 1;
}

extern "C" int keyhunt_cuda_bsgs_center_workspace_create(cuda_bsgs_center_workspace **workspace) {
  std::lock_guard<std::mutex> lock(g_cuda_mutex);
  g_cuda_error[0] = 0;

  if (g_cuda_ctx == nullptr) {
    set_bridge_error("CUDA BSGS context is not prepared");
    return 0;
  }
  if (!cuda_bsgs_center_workspace_create(workspace)) {
    set_bridge_error("CUDA center workspace create failed");
    return 0;
  }
  return 1;
}

extern "C" void keyhunt_cuda_bsgs_center_workspace_destroy(cuda_bsgs_center_workspace *workspace) {
  cuda_bsgs_center_workspace_destroy(workspace);
}

extern "C" int keyhunt_cuda_bsgs_probe_center_sequence_workspace(cuda_bsgs_center_workspace *workspace,
                                                                  const cuda_bsgs_affine_point *seed,
                                                                  uint64_t first_user_index,
                                                                  uint32_t work_count,
                                                                  cuda_bsgs_center_hit *hits,
                                                                  uint32_t max_hits,
                                                                  uint32_t *hit_count,
                                                                  cuda_bsgs_affine_point *next_seed) {
  cuda_bsgs_context *ctx = g_cuda_ctx;
  if (ctx == nullptr) {
    set_bridge_error("CUDA BSGS context is not prepared");
    return 0;
  }
  if (!cuda_bsgs_probe_center_sequence_workspace(ctx, workspace, seed, first_user_index,
                                                 work_count, hits, max_hits, hit_count,
                                                 next_seed)) {
    snprintf(g_cuda_error, sizeof(g_cuda_error), "%s", cuda_bsgs_last_error(ctx));
    return 0;
  }
  return 1;
}

extern "C" void keyhunt_cuda_bsgs_release(void) {
  std::lock_guard<std::mutex> lock(g_cuda_mutex);
  if (g_cuda_ctx != nullptr) {
    cuda_bsgs_destroy(g_cuda_ctx);
    g_cuda_ctx = nullptr;
  }
}

extern "C" const char *keyhunt_cuda_last_error(void) {
  return g_cuda_error;
}
