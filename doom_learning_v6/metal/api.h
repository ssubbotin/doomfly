#ifndef DOOMFLY_METAL_API_H
#define DOOMFLY_METAL_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DF_METAL_ABI_VERSION 7u

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

typedef struct {
  int64_t tick;
  int32_t neuron;
  int32_t reserved;
} df_metal_kc_event;

typedef struct {
  double native_total_seconds;
  double gpu_seconds;
  double counts_clear_seconds;
  double encode_seconds;
  double commit_call_seconds;
  double wait_call_seconds;
  double native_event_copy_seconds;
  uint64_t native_event_copy_bytes;
  uint32_t encoder_count;
  uint32_t dispatch_count;
  uint32_t mark_grid_threads;
  uint32_t gather_grid_threads;
  uint32_t indirect_dispatch_count;
  uint32_t edge_bitmap_words;
} df_metal_timing;

int df_metal_probe(df_metal_device_info *info);
int df_metal_create(const df_metal_graph *graph,const char *metallib_path,
    df_metal_handle *handle);
int df_metal_create_batch(const df_metal_graph *graph,const char *metallib_path,
    int32_t lanes,df_metal_handle *handle);
int df_metal_upload_lane_state(df_metal_handle handle,int32_t lane,const df_metal_state *state);
int df_metal_download_lane_state(df_metal_handle handle,int32_t lane,df_metal_state *state);
int df_metal_upload_lane_drive(df_metal_handle handle,int32_t lane,const float *drive);
int df_metal_download_lane_observation(df_metal_handle handle,int32_t lane,
    int32_t *counts,int64_t *cursor);
int df_metal_update_lane_weights(df_metal_handle handle,int32_t lane,int32_t count,
    const int64_t *edge_ids,const float *values);
int df_metal_apply_lane_eligibility(df_metal_handle handle,int32_t lane,
    double *eligibility,int64_t *eligibility_last,double tau_ms);
int df_metal_batch_memory_bytes(df_metal_handle handle,uint64_t *shared,uint64_t *mutable_bytes);
int df_metal_upload_state(df_metal_handle handle,const df_metal_state *state);
int df_metal_download_state(df_metal_handle handle,df_metal_state *state);
int df_metal_upload_drive(df_metal_handle handle,const float *drive);
int df_metal_download_observation(df_metal_handle handle,int32_t *counts,int64_t *cursor);
int df_metal_update_weights(df_metal_handle handle,int32_t count,
    const int64_t *edge_ids,const float *values);
int df_metal_advance(df_metal_handle handle,int32_t steps,
    df_metal_kc_event *events,int32_t event_capacity,int32_t *event_count,
    df_metal_timing *timing);
int df_metal_apply_eligibility(df_metal_handle handle,double *eligibility,
    int64_t *eligibility_last,double tau_ms);
int df_metal_set_diagnostics(df_metal_handle handle,int32_t enabled);
const char *df_metal_last_error(void);
void df_metal_destroy(df_metal_handle handle);

#ifdef __cplusplus
}
#endif

#endif
