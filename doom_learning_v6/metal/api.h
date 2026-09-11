#ifndef DOOMFLY_METAL_API_H
#define DOOMFLY_METAL_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DF_METAL_ABI_VERSION 1u

typedef void *df_metal_handle;

typedef struct {
  uint32_t abi_version;
  uint32_t supported;
  uint32_t has_unified_memory;
  uint32_t supports_apple7;
  uint64_t recommended_working_set;
  uint64_t registry_id;
  char device_name[256];
} df_metal_device_info;

typedef struct {
  int32_t neurons;
  int64_t edges;
  int32_t delay_slots;
  float dt_ms;
  float adaptation_jump_mv;
  float adaptation_tau_ms;
  const int64_t *out_ptr;
  const int32_t *out_post;
  const int64_t *in_ptr;
  const int32_t *in_pre;
  const int32_t *in_edge;
  const uint8_t *kc_mask;
  const uint8_t *modulation_mask;
} df_metal_graph;

typedef struct {
  int64_t cursor;
  float *weight;
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
  float *modulation;
  int64_t *modulation_last;
  float *rest;
  float *adaptation;
} df_metal_state;

int df_metal_probe(df_metal_device_info *info);
int df_metal_create(const df_metal_graph *graph,const char *metallib_path,
    df_metal_handle *handle);
int df_metal_upload_state(df_metal_handle handle,const df_metal_state *state);
int df_metal_download_state(df_metal_handle handle,df_metal_state *state);
int df_metal_update_weights(df_metal_handle handle,int32_t count,
    const int64_t *edge_ids,const float *values);
const char *df_metal_last_error(void);
void df_metal_destroy(df_metal_handle handle);

#ifdef __cplusplus
}
#endif

#endif
