#include "bsgs_cuda.h"
#include "../secp256k1/SECP256K1.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

static void free_blooms(bloom *blooms) {
  if (blooms == nullptr) {
    return;
  }
  for (int i = 0; i < 256; i++) {
    bloom_free(&blooms[i]);
  }
  free(blooms);
}

static void point_to_cuda(const Point &src, cuda_bsgs_affine_point *dst) {
  for (int i = 0; i < 4; i++) {
    dst->x.v[i] = src.x.bits64[i];
    dst->y.v[i] = src.y.bits64[i];
  }
  dst->infinity = const_cast<Point &>(src).isZero() ? 1U : 0U;
  dst->reserved = 0;
}

static void point_x_bytes(Point &p, unsigned char out[32]) {
  p.x.Get32Bytes(out);
}

static int run_center_selftest(cuda_bsgs_context *ctx, bloom *b1, bloom *b2, bloom *b3) {
  Secp256K1 secp;
  secp.Init();

  cuda_bsgs_affine_point gsn_cuda[512];
  Point gsn_cpu[512];
  Point step = secp.G;
  for (int i = 0; i < 512; i++) {
    if (i == 0) {
      gsn_cpu[i] = step;
    } else if (i == 1) {
      gsn_cpu[i] = secp.DoubleDirect(step);
    } else {
      gsn_cpu[i] = secp.AddDirect(gsn_cpu[i - 1], step);
    }
    point_to_cuda(gsn_cpu[i], &gsn_cuda[i]);
  }

  Int scalar;
  scalar.SetInt32(2000);
  Point start = secp.ComputePublicKey(&scalar);

  cuda_bsgs_affine_point two_gsn;
  point_to_cuda(step, &two_gsn);
  if (!cuda_bsgs_upload_search_points(ctx, gsn_cuda, &two_gsn)) {
    std::fprintf(stderr, "upload search points failed: %s\n", cuda_bsgs_last_error(ctx));
    return 0;
  }

  const uint32_t selected[] = {0, 1, 127, 511, 512, 513, 900, 1023};
  unsigned char expected[sizeof(selected) / sizeof(selected[0])][32];
  for (size_t i = 0; i < sizeof(selected) / sizeof(selected[0]); i++) {
    const uint32_t point_index = selected[i];
    Point expected_point;
    if (point_index < 512) {
      const int gsn_index = 511 - static_cast<int>(point_index);
      Point neg = secp.Negation(gsn_cpu[gsn_index]);
      expected_point = secp.AddDirect(start, neg);
    } else if (point_index == 512) {
      expected_point = start;
    } else {
      const int gsn_index = static_cast<int>(point_index) - 513;
      expected_point = secp.AddDirect(start, gsn_cpu[gsn_index]);
    }
    point_x_bytes(expected_point, expected[i]);
    bloom_add(&b1[expected[i][0]], expected[i], 32);
  }

  if (!cuda_bsgs_upload_blooms(ctx, b1, b2, b3, 256)) {
    std::fprintf(stderr, "upload center blooms failed: %s\n", cuda_bsgs_last_error(ctx));
    return 0;
  }

  cuda_bsgs_center_work work;
  std::memset(&work, 0, sizeof(work));
  point_to_cuda(start, &work.startP);
  cuda_bsgs_center_hit hits[32];
  uint32_t hit_count = 0;
  if (!cuda_bsgs_probe_center_batches(ctx, &work, 1, hits, 32, &hit_count)) {
    std::fprintf(stderr, "center probe failed: %s\n", cuda_bsgs_last_error(ctx));
    return 0;
  }

  for (size_t i = 0; i < sizeof(selected) / sizeof(selected[0]); i++) {
    int found = 0;
    for (uint32_t h = 0; h < hit_count && h < 32; h++) {
      if (hits[h].work_index == 0 &&
          hits[h].point_index == selected[i] &&
          std::memcmp(hits[h].x, expected[i], 32) == 0) {
        found = 1;
        break;
      }
    }
    if (!found) {
      std::fprintf(stderr, "missing center hit for point index %u (hit_count=%u)\n",
                   selected[i], hit_count);
      return 0;
    }
  }

  return 1;
}

static int run_center_sequence_selftest(cuda_bsgs_context *ctx, bloom *b1, bloom *b2, bloom *b3) {
  Secp256K1 secp;
  secp.Init();

  cuda_bsgs_affine_point gsn_cuda[512];
  Point gsn_cpu[512];
  Point step = secp.G;
  for (int i = 0; i < 512; i++) {
    if (i == 0) {
      gsn_cpu[i] = step;
    } else if (i == 1) {
      gsn_cpu[i] = secp.DoubleDirect(step);
    } else {
      gsn_cpu[i] = secp.AddDirect(gsn_cpu[i - 1], step);
    }
    point_to_cuda(gsn_cpu[i], &gsn_cuda[i]);
  }

  Point stride = secp.DoubleDirect(gsn_cpu[511]);
  cuda_bsgs_affine_point stride_cuda;
  point_to_cuda(stride, &stride_cuda);
  if (!cuda_bsgs_upload_search_points(ctx, gsn_cuda, &stride_cuda)) {
    std::fprintf(stderr, "upload sequence search points failed: %s\n", cuda_bsgs_last_error(ctx));
    return 0;
  }

  Int scalar;
  scalar.SetInt32(5000);
  Point seed = secp.ComputePublicKey(&scalar);
  unsigned char seed_x[32];
  point_x_bytes(seed, seed_x);
  bloom_add(&b1[seed_x[0]], seed_x, 32);

  const uint32_t work_index = 7;
  const uint32_t point_index = 600;
  Point center = seed;
  for (uint32_t i = 0; i < work_index; i++) {
    center = secp.AddDirect(center, stride);
  }
  unsigned char center_x[32];
  point_x_bytes(center, center_x);
  bloom_add(&b1[center_x[0]], center_x, 32);
  Point expected_point = secp.AddDirect(center, gsn_cpu[point_index - 513]);
  unsigned char expected[32];
  point_x_bytes(expected_point, expected);
  bloom_add(&b1[expected[0]], expected, 32);
  if (!cuda_bsgs_upload_blooms(ctx, b1, b2, b3, 256)) {
    std::fprintf(stderr, "upload sequence blooms failed: %s\n", cuda_bsgs_last_error(ctx));
    return 0;
  }

  cuda_bsgs_affine_point seed_cuda;
  cuda_bsgs_affine_point next_seed_cuda;
  point_to_cuda(seed, &seed_cuda);
  cuda_bsgs_center_hit hits[32];
  uint32_t hit_count = 0;
  if (!cuda_bsgs_probe_center_sequence(ctx, &seed_cuda, 100, 16, hits, 32,
                                       &hit_count, &next_seed_cuda)) {
    std::fprintf(stderr, "center sequence probe failed: %s\n", cuda_bsgs_last_error(ctx));
    return 0;
  }

  int found = 0;
  int found_seed = 0;
  int found_center = 0;
  for (uint32_t h = 0; h < hit_count && h < 32; h++) {
    if (hits[h].work_index == 0 &&
        hits[h].point_index == 512 &&
        std::memcmp(hits[h].x, seed_x, 32) == 0) {
      found_seed = 1;
    }
    if (hits[h].work_index == work_index &&
        hits[h].point_index == 512 &&
        std::memcmp(hits[h].x, center_x, 32) == 0) {
      found_center = 1;
    }
    if (hits[h].work_index == work_index &&
        hits[h].point_index == point_index &&
        std::memcmp(hits[h].x, expected, 32) == 0) {
      found = 1;
    }
  }
  if (!found_seed) {
    std::fprintf(stderr, "missing center sequence seed hit (hit_count=%u)\n", hit_count);
    return 0;
  }
  if (!found_center) {
    std::fprintf(stderr, "missing center sequence center hit (hit_count=%u)\n", hit_count);
    return 0;
  }
  if (!found) {
    std::fprintf(stderr, "missing center sequence hit (hit_count=%u)\n", hit_count);
    return 0;
  }

  Point expected_next = seed;
  for (int i = 0; i < 16; i++) {
    expected_next = secp.AddDirect(expected_next, stride);
  }
  cuda_bsgs_affine_point expected_next_cuda;
  point_to_cuda(expected_next, &expected_next_cuda);
  if (std::memcmp(&next_seed_cuda.x, &expected_next_cuda.x, sizeof(next_seed_cuda.x)) != 0 ||
      std::memcmp(&next_seed_cuda.y, &expected_next_cuda.y, sizeof(next_seed_cuda.y)) != 0 ||
      next_seed_cuda.infinity != expected_next_cuda.infinity) {
    std::fprintf(stderr, "bad sequence next seed\n");
    return 0;
  }

  return 1;
}

int main() {
  if (!cuda_bsgs_available()) {
    std::fprintf(stderr, "CUDA device not available\n");
    return 1;
  }

  bloom *b1 = static_cast<bloom *>(std::calloc(256, sizeof(bloom)));
  bloom *b2 = static_cast<bloom *>(std::calloc(256, sizeof(bloom)));
  bloom *b3 = static_cast<bloom *>(std::calloc(256, sizeof(bloom)));
  if (b1 == nullptr || b2 == nullptr || b3 == nullptr) {
    std::fprintf(stderr, "calloc failed\n");
    free_blooms(b1);
    free_blooms(b2);
    free_blooms(b3);
    return 1;
  }

  for (int i = 0; i < 256; i++) {
    if (bloom_init2(&b1[i], 1000, 0.000001) ||
        bloom_init2(&b2[i], 1000, 0.000001) ||
        bloom_init2(&b3[i], 1000, 0.000001)) {
      std::fprintf(stderr, "bloom_init2 failed\n");
      free_blooms(b1);
      free_blooms(b2);
      free_blooms(b3);
      return 1;
    }
  }

  unsigned char xpoints[2][32];
  std::memset(xpoints, 0, sizeof(xpoints));
  for (int i = 0; i < 32; i++) {
    xpoints[0][i] = static_cast<unsigned char>(i + 7);
    xpoints[1][i] = static_cast<unsigned char>(0xa0 + i);
  }

  const int shard = xpoints[0][0];
  bloom_add(&b1[shard], xpoints[0], 32);
  bloom_add(&b3[shard], xpoints[0], 32);

  cuda_bsgs_xvalue table[1];
  std::memset(table, 0, sizeof(table));
  std::memcpy(table[0].value, xpoints[0] + 16, sizeof(table[0].value));
  table[0].index = 12345;

  cuda_bsgs_context *ctx = nullptr;
  if (!cuda_bsgs_create(&ctx)) {
    std::fprintf(stderr, "cuda_bsgs_create failed\n");
    free_blooms(b1);
    free_blooms(b2);
    free_blooms(b3);
    return 1;
  }

  cuda_bsgs_hit hits[4];
  uint32_t hit_count = 0;
  int ok = cuda_bsgs_upload_blooms(ctx, b1, b2, b3, 256) &&
           cuda_bsgs_upload_table(ctx, table, 1) &&
           cuda_bsgs_probe_xpoints(ctx, &xpoints[0][0], 2, hits, 4, &hit_count);
  if (!ok) {
    std::fprintf(stderr, "CUDA BSGS selftest failed: %s\n", cuda_bsgs_last_error(ctx));
    cuda_bsgs_destroy(ctx);
    free_blooms(b1);
    free_blooms(b2);
    free_blooms(b3);
    return 1;
  }

  if (hit_count != 1 || hits[0].xpoint_index != 0 || hits[0].table_index != 12345 || hits[0].stage != 3) {
    std::fprintf(stderr,
                 "unexpected hit result: count=%u xpoint=%u index=%llu stage=%u\n",
                 hit_count,
                 hits[0].xpoint_index,
                 static_cast<unsigned long long>(hits[0].table_index),
                 hits[0].stage);
    cuda_bsgs_destroy(ctx);
    free_blooms(b1);
    free_blooms(b2);
    free_blooms(b3);
    return 1;
  }

  if (!run_center_selftest(ctx, b1, b2, b3) ||
      !run_center_sequence_selftest(ctx, b1, b2, b3)) {
    std::fprintf(stderr, "CUDA BSGS center selftest failed: %s\n", cuda_bsgs_last_error(ctx));
    cuda_bsgs_destroy(ctx);
    free_blooms(b1);
    free_blooms(b2);
    free_blooms(b3);
    return 1;
  }

  std::printf("CUDA BSGS selftest OK\n");
  cuda_bsgs_destroy(ctx);
  free_blooms(b1);
  free_blooms(b2);
  free_blooms(b3);
  return 0;
}
