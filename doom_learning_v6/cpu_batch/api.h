#ifndef DOOMFLY_CPU_BATCH_API_H
#define DOOMFLY_CPU_BATCH_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
#define DF_CPU_BATCH_ABI_VERSION 1u

typedef void *df_cpu_batch_handle;

typedef struct {
  int32_t neurons;
  int64_t edges;
  int32_t delay_slots;
  int32_t plastic_edges;
  float dt_ms;
  float eligibility_tau_ms;
  float adaptation_jump_mv;
  float adaptation_tau_ms;
  const int64_t *out_ptr;
  const int32_t *out_post;
  const float *base_weight;
  const int16_t *plastic_slot;
  const uint8_t *kc_mask;
  const uint8_t *modulation_mask;
  const float *rest;
} df_cpu_batch_graph;

typedef struct {
  int64_t *cursor;
  float *plastic_weight;
  float *v;
  float *g;
  int16_t *refractory;
  float *drive;
  float *previous_drive;
  int32_t *queue;
  int32_t *queue_count;
  int32_t *counts;
  int32_t *active;
  uint8_t *active_flag;
  int32_t *nactive;
  int64_t *last;
  double *eligibility;
  int64_t *eligibility_last;
  float *modulation;
  int64_t *modulation_last;
  float *adaptation;
} df_cpu_batch_lane;

typedef struct {
  double native_wall_seconds;
  int32_t lanes_advanced;
  int32_t workers;
  int32_t steps;
  uint64_t generation;
  int32_t pool_threads;
} df_cpu_batch_timing;

uint32_t df_cpu_batch_abi_version(void);
int df_cpu_batch_create(const df_cpu_batch_graph *graph,
    const df_cpu_batch_lane *lanes, int32_t lane_count, int32_t workers,
    df_cpu_batch_handle *handle);
int df_cpu_batch_advance(df_cpu_batch_handle handle, int32_t steps,
    df_cpu_batch_timing *timing);
const char *df_cpu_batch_error(df_cpu_batch_handle handle);
void df_cpu_batch_destroy(df_cpu_batch_handle handle);

#ifdef __cplusplus
}
#endif

#endif
