#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "api.h"

#include <cstring>
#include <string>

namespace {
thread_local std::string last_error;

int fail(const char *message) {
  last_error=message;
  return 1;
}
}

extern "C" const char *df_metal_last_error(void) { return last_error.c_str(); }

extern "C" int df_metal_probe(df_metal_device_info *info) {
  if(info==nullptr)return fail("Device information pointer is null");
  std::memset(info,0,sizeof(*info));
  info->abi_version=DF_METAL_ABI_VERSION;
  @autoreleasepool {
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    if(device==nil)return fail("Metal device is unavailable");
    info->has_unified_memory=device.hasUnifiedMemory?1u:0u;
    if(@available(macOS 11.0,*))
      info->supports_apple7=[device supportsFamily:MTLGPUFamilyApple7]?1u:0u;
    info->recommended_working_set=device.recommendedMaxWorkingSetSize;
    info->registry_id=device.registryID;
    const char *name=device.name.UTF8String;
    if(name!=nullptr)std::strncpy(info->device_name,name,sizeof(info->device_name)-1);
    info->supported=info->has_unified_memory&&info->supports_apple7;
    if(!info->supported)return fail("Metal requires unified memory and Apple GPU family 7");
  }
  last_error.clear();
  return 0;
}

extern "C" void df_metal_destroy(df_metal_handle) {}
