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

int df_metal_probe(df_metal_device_info *info);
const char *df_metal_last_error(void);
void df_metal_destroy(df_metal_handle handle);

#ifdef __cplusplus
}
#endif

#endif
