#include "api.h"

extern "C" uint32_t df_cpu_batch_abi_version(void) {
  return DF_CPU_BATCH_ABI_VERSION;
}
