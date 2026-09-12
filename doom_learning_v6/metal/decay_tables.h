#ifndef DOOMFLY_METAL_DECAY_TABLES_H
#define DOOMFLY_METAL_DECAY_TABLES_H

#include <array>
#include <cmath>

namespace doomfly::metal {
inline std::array<float,3072> make_decay_tables(float dt,float adaptation_tau) {
  std::array<float,3072> result;
  for(int i=0;i<1024;i++) {
    result[i]=std::exp(-dt*i/20.f);
    result[1024+i]=std::exp(-dt*i/5.f);
    result[2048+i]=std::exp(-dt*i/adaptation_tau);
  }
  return result;
}
}

#endif
