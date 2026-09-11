#include <metal_stdlib>
using namespace metal;

kernel void df_noop(uint index [[thread_position_in_grid]]) {
  (void)index;
}
