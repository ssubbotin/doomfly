#include <metal_stdlib>
using namespace metal;

struct Params {
  uint neurons;
  uint words;
  uint edge_words;
  uint clear_threadgroup_width;
  uint slot;
  uint future;
  long clock;
  float dt;
  float adaptation_jump;
  float adaptation_tau;
  int refractory_ticks;
  uint event_capacity;
  uint capture_all_spikes;
  uint lanes;
  uint weight_lane;
  ulong edge_stride;
  ulong ring_stride;
  ulong edge_bitmap_stride;
  uint window_ticks;
  uint window_capacity;
};

struct KCEvent { long tick; int neuron; int reserved; };

kernel void df_update_weights(device float *weight [[buffer(0)]],
    device const long *edge_ids [[buffer(1)]],device const float *values [[buffer(2)]],
    constant Params &p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
  if(i<p.neurons)weight[p.weight_lane*p.edge_stride+edge_ids[i]]=values[i];
}

inline bool ring_contains(device atomic_uint *ring,uint words,uint slot,uint neuron) {
  uint word=atomic_load_explicit(&ring[slot*words+(neuron>>5)],memory_order_relaxed);
  return (word&(1u<<(neuron&31)))!=0;
}

inline void evolve(uint i,long now,float current,device float *v,device float *g,
    device short *refractory,device long *last,device const float *rest,
    device float *adaptation,device const float *decay,float dt,float adaptation_tau) {
  long d=now-last[i];if(d<=0)return;
  int frozen=refractory[i]>0?refractory[i]-1:0;
  int skip=min((int)d,frozen);
  if(skip>0&&adaptation[i]>0.0f)
    adaptation[i]*=skip<1024?decay[2048+skip]:exp(-dt*skip/adaptation_tau);
  refractory[i]=d>=refractory[i]?0:refractory[i]-(short)d;d-=skip;
  if(d>0){
    float a=d<1024?decay[d]:exp(-dt*d/20.0f);
    float b=d<1024?decay[1024+d]:exp(-dt*d/5.0f);
    float voltage=fma(v[i]-rest[i],a,rest[i]);
    voltage=fma(current,1.0f-a,voltage);
    v[i]=voltage+g[i]*(a-b)/3.0f;
    g[i]*=b;
    if(adaptation[i]>0.0f){
      float c=d<1024?decay[2048+d]:exp(-dt*d/adaptation_tau);
      float correction=(-adaptation[i]*adaptation_tau)/(adaptation_tau-20.0f);
      v[i]=fma(correction,c-a,v[i]);
      adaptation[i]*=c;
    }
  }
  last[i]=now;
}

inline void integrate_neuron(uint i,long now,uint future,uint lane,
    device float *v,device float *g,device short *refractory,device const float *drive,
    device uchar *active,device long *last,device const float *rest,device float *adaptation,
    device atomic_uint *ring,device int *counts,device const uchar *kc_mask,
    device KCEvent *events,device atomic_uint *event_count,device const float *decay,
    constant Params &p) {
  if(active[i]!=0){
    evolve(i,now,drive[i],v,g,refractory,last,rest,adaptation,decay,p.dt,p.adaptation_tau);
    if(refractory[i]==0&&v[i]>-45.0f){
      atomic_fetch_or_explicit(&ring[future*p.words+(i>>5)],1u<<(i&31),memory_order_relaxed);
      counts[i]++;
      if(kc_mask[i]||p.capture_all_spikes){
        uint position=atomic_fetch_add_explicit(event_count,1u,memory_order_relaxed);
        if(position<p.event_capacity)events[position]=KCEvent{now,(int)i,(int)lane};
      }
      if(kc_mask[i])adaptation[i]+=p.adaptation_jump;
    }
    float gap=-45.0f-rest[i];
    bool can_fire=v[i]>-45.0f||drive[i]>gap||drive[i]+g[i]>gap;
    if(!can_fire)active[i]=0;
  }
}

inline void gather_neuron(uint target,long now,device const long *in_ptr,
    device const int *in_pre,device const int *in_edge,device const float *weight,
    device atomic_uint *active_edge_bits,device atomic_uint *touched,
    device float *v,device float *g,device short *refractory,device const float *drive,
    device uchar *active,device long *last,device float *modulation,
    device long *modulation_last,device const uchar *modulation_mask,
    device const float *rest,device float *adaptation,device const float *decay,
    constant Params &p) {
  if(atomic_load_explicit(&touched[target],memory_order_relaxed)!=0&&
      atomic_exchange_explicit(&touched[target],0u,memory_order_relaxed)!=0){
    evolve(target,now,drive[target],v,g,refractory,last,rest,adaptation,decay,p.dt,p.adaptation_tau);
    float conductance=0.0f,modulatory=0.0f;bool has_fast=false,has_modulatory=false;
    long start=in_ptr[target],end=in_ptr[target+1];
    if(start<end){
      uint first=(uint)start>>5,last_word=(uint)(end-1)>>5;
      for(uint word=first;word<=last_word;word++){
        uint low=word==first?((uint)start&31):0;
        uint high=word==last_word?(((uint)(end-1)&31)+1):32;
        uint mask=0xffffffffu;
        if(low>0)mask&=0xffffffffu<<low;
        if(high<32)mask&=(1u<<high)-1;
        uint bits=0u;
        if((atomic_load_explicit(&active_edge_bits[word],memory_order_relaxed)&mask)!=0)
          bits=atomic_fetch_and_explicit(&active_edge_bits[word],~mask,
            memory_order_relaxed)&mask;
        while(bits!=0){
          uint bit=ctz(bits);uint position=(word<<5)+bit;
          int pre=in_pre[position];float value=weight[in_edge[position]];
          if(modulation_mask[pre]){modulatory+=fabs(value)/0.275f;has_modulatory=true;}
          else if(refractory[target]==0){conductance+=value;has_fast=true;}
          bits&=bits-1;
        }
      }
    }
    if(has_modulatory){
      long delta=now-modulation_last[target];
      modulation[target]*=exp(-p.dt*delta/100.0f);
      modulation[target]+=modulatory;modulation_last[target]=now;
    }
    if(has_fast){g[target]+=conductance;active[target]=1;}
  }
}

kernel void df_drive_change(device float *v [[buffer(0)]],device float *g [[buffer(1)]],
    device short *refractory [[buffer(2)]],device const float *drive [[buffer(3)]],
    device float *previous [[buffer(4)]],device uchar *active [[buffer(5)]],
    device long *last [[buffer(6)]],device const float *rest [[buffer(7)]],
    device float *adaptation [[buffer(8)]],device const float *decay [[buffer(9)]],
    constant Params &p [[buffer(10)]],
    uint2 position [[thread_position_in_grid]]) {
  uint i=position.x,lane=position.y;
  if(i>=p.neurons||lane>=p.lanes)return;
  ulong offset=ulong(lane)*p.neurons;
  v+=offset;g+=offset;refractory+=offset;drive+=offset;previous+=offset;
  active+=offset;last+=offset;adaptation+=offset;
  if(drive[i]==previous[i])return;
  evolve(i,p.clock-1,previous[i],v,g,refractory,last,rest,adaptation,decay,p.dt,p.adaptation_tau);
  previous[i]=drive[i];active[i]=1;
}

kernel void df_integrate_mark(device float *v [[buffer(0)]],device float *g [[buffer(1)]],
    device short *refractory [[buffer(2)]],device const float *drive [[buffer(3)]],
    device uchar *active [[buffer(4)]],device long *last [[buffer(5)]],
    device const float *rest [[buffer(6)]],device float *adaptation [[buffer(7)]],
    device atomic_uint *ring [[buffer(8)]],device int *counts [[buffer(9)]],
    device const uchar *kc_mask [[buffer(10)]],device KCEvent *events [[buffer(11)]],
    device atomic_uint *event_count [[buffer(12)]],device const long *out_ptr [[buffer(13)]],
    device const int *out_post [[buffer(14)]],device const int *edge_to_incoming [[buffer(15)]],
    device atomic_uint *touched [[buffer(16)]],device atomic_uint *active_edge_bits [[buffer(17)]],
    device const float *decay [[buffer(18)]],constant Params &p [[buffer(19)]],
    uint2 position [[thread_position_in_grid]]) {
  uint i=position.x,lane=position.y;
  if(i>=p.neurons||lane>=p.lanes)return;
  ulong offset=ulong(lane)*p.neurons;
  v+=offset;g+=offset;refractory+=offset;drive+=offset;active+=offset;
  last+=offset;adaptation+=offset;counts+=offset;touched+=offset;
  ring+=ulong(lane)*p.ring_stride;active_edge_bits+=ulong(lane)*p.edge_bitmap_stride;
  integrate_neuron(i,p.clock,p.future,lane,v,g,refractory,drive,active,last,rest,
    adaptation,ring,counts,kc_mask,events,event_count,decay,p);
  if(!ring_contains(ring,p.words,p.slot,i))return;
  for(long edge=out_ptr[i];edge<out_ptr[i+1];edge++){
    atomic_store_explicit(&touched[out_post[edge]],1u,memory_order_relaxed);
    uint position=(uint)edge_to_incoming[edge];
    atomic_fetch_or_explicit(&active_edge_bits[position>>5],
      1u<<(position&31),memory_order_relaxed);
  }
}

kernel void df_gather_finalize(device const long *in_ptr [[buffer(0)]],
    device const int *in_pre [[buffer(1)]],device const int *in_edge [[buffer(2)]],
    device const float *weight [[buffer(3)]],device atomic_uint *active_edge_bits [[buffer(4)]],
    device atomic_uint *touched [[buffer(5)]],device float *v [[buffer(6)]],
    device float *g [[buffer(7)]],device short *refractory [[buffer(8)]],
    device const float *drive [[buffer(9)]],device uchar *active [[buffer(10)]],
    device long *last [[buffer(11)]],device float *modulation [[buffer(12)]],
    device long *modulation_last [[buffer(13)]],device const uchar *modulation_mask [[buffer(14)]],
    device const float *rest [[buffer(15)]],device float *adaptation [[buffer(16)]],
    device atomic_uint *ring [[buffer(17)]],device const float *decay [[buffer(18)]],
    constant Params &p [[buffer(19)]],
    uint2 position [[thread_position_in_grid]]) {
  uint target=position.x,lane=position.y;
  if(target>=p.neurons||lane>=p.lanes)return;
  ulong offset=ulong(lane)*p.neurons;
  v+=offset;g+=offset;refractory+=offset;drive+=offset;active+=offset;last+=offset;
  modulation+=offset;modulation_last+=offset;adaptation+=offset;touched+=offset;
  weight+=ulong(lane)*p.edge_stride;
  active_edge_bits+=ulong(lane)*p.edge_bitmap_stride;ring+=ulong(lane)*p.ring_stride;
  gather_neuron(target,p.clock,in_ptr,in_pre,in_edge,weight,active_edge_bits,touched,
    v,g,refractory,drive,active,last,modulation,modulation_last,modulation_mask,
    rest,adaptation,decay,p);
  if(target<p.words)
    atomic_store_explicit(&ring[p.slot*p.words+target],0u,memory_order_relaxed);
  if(ring_contains(ring,p.words,p.future,target)){
    v[target]=rest[target];g[target]=0.0f;refractory[target]=(short)p.refractory_ticks;
  }
}

// A window is strictly shorter than the delay. Its input memberships are
// already fixed, and preparation owns each consumed physical ring word once.
kernel void df_window_prepare(device atomic_uint *ring [[buffer(0)]],
    device const long *out_ptr [[buffer(1)]],device const int *out_post [[buffer(2)]],
    device const int *edge_to_incoming [[buffer(3)]],
    device atomic_uint *time_touched [[buffer(4)]],device atomic_uint *time_bits [[buffer(5)]],
    constant Params &p [[buffer(6)]],uint2 position [[thread_position_in_grid]]) {
  uint lane=position.y;
  if(position.x>=p.words*p.window_ticks||lane>=p.lanes)return;
  uint tick_offset=position.x/p.words,word=position.x%p.words;
  uint slot=uint((p.clock+tick_offset)%19);
  ring+=ulong(lane)*p.ring_stride;
  ulong plane=ulong(lane)*p.window_capacity+tick_offset;
  time_touched+=plane*p.neurons;time_bits+=plane*p.edge_words;
  uint members=atomic_exchange_explicit(&ring[slot*p.words+word],0u,memory_order_relaxed);
  while(members!=0u){
    uint neuron=(word<<5)+ctz(members);
    if(neuron<p.neurons){
      for(long edge=out_ptr[neuron];edge<out_ptr[neuron+1];++edge){
        atomic_store_explicit(&time_touched[out_post[edge]],1u,memory_order_relaxed);
        uint incoming=uint(edge_to_incoming[edge]);
        atomic_fetch_or_explicit(&time_bits[incoming>>5],1u<<(incoming&31),memory_order_relaxed);
      }
    }
    members&=members-1u;
  }
}

// Each worker owns one neuron's chronological state. Incoming positions retain
// the reference ascending order; future membership stays atomic across workers.
kernel void df_window_neurons(device float *v [[buffer(0)]],device float *g [[buffer(1)]],
    device short *refractory [[buffer(2)]],device const float *drive [[buffer(3)]],
    device uchar *active [[buffer(4)]],device long *last [[buffer(5)]],
    device const float *rest [[buffer(6)]],device float *adaptation [[buffer(7)]],
    device atomic_uint *ring [[buffer(8)]],device int *counts [[buffer(9)]],
    device const uchar *kc_mask [[buffer(10)]],device KCEvent *events [[buffer(11)]],
    device atomic_uint *event_count [[buffer(12)]],device const long *in_ptr [[buffer(13)]],
    device const int *in_pre [[buffer(14)]],device const int *in_edge [[buffer(15)]],
    device const float *weight [[buffer(16)]],device atomic_uint *time_touched [[buffer(17)]],
    device atomic_uint *time_bits [[buffer(18)]],device float *modulation [[buffer(19)]],
    device long *modulation_last [[buffer(20)]],device const uchar *modulation_mask [[buffer(21)]],
    device const float *decay [[buffer(22)]],constant Params &p [[buffer(23)]],
    uint2 position [[thread_position_in_grid]]) {
  uint target=position.x,lane=position.y;
  if(target>=p.neurons||lane>=p.lanes)return;
  ulong lane_offset=ulong(lane)*p.neurons;
  v+=lane_offset;g+=lane_offset;refractory+=lane_offset;drive+=lane_offset;
  active+=lane_offset;last+=lane_offset;adaptation+=lane_offset;counts+=lane_offset;
  modulation+=lane_offset;modulation_last+=lane_offset;
  ring+=ulong(lane)*p.ring_stride;weight+=ulong(lane)*p.edge_stride;
  time_touched+=ulong(lane)*p.window_capacity*p.neurons;
  time_bits+=ulong(lane)*p.window_capacity*p.edge_words;
  for(uint offset=0;offset<p.window_ticks;++offset){
    long now=p.clock+offset;
    uint future=uint((now+18)%19);
    integrate_neuron(target,now,future,lane,v,g,refractory,drive,active,last,rest,
      adaptation,ring,counts,kc_mask,events,event_count,decay,p);
    device atomic_uint *touched=time_touched+ulong(offset)*p.neurons;
    device atomic_uint *active_edge_bits=time_bits+ulong(offset)*p.edge_words;
    gather_neuron(target,now,in_ptr,in_pre,in_edge,weight,active_edge_bits,touched,
      v,g,refractory,drive,active,last,modulation,modulation_last,modulation_mask,
      rest,adaptation,decay,p);
    if(ring_contains(ring,p.words,future,target)){
      v[target]=rest[target];g[target]=0.0f;refractory[target]=(short)p.refractory_ticks;
    }
  }
}

kernel void df_materialize(device float *v [[buffer(0)]],device float *g [[buffer(1)]],
    device short *refractory [[buffer(2)]],device const float *drive [[buffer(3)]],
    device long *last [[buffer(4)]],device const float *rest [[buffer(5)]],
    device float *adaptation [[buffer(6)]],device const float *decay [[buffer(7)]],
    constant Params &p [[buffer(8)]],
    uint2 position [[thread_position_in_grid]]) {
  uint i=position.x,lane=position.y;
  if(i>=p.neurons||lane>=p.lanes)return;
  ulong offset=ulong(lane)*p.neurons;
  v+=offset;g+=offset;refractory+=offset;drive+=offset;last+=offset;adaptation+=offset;
  evolve(i,p.clock,drive[i],v,g,refractory,last,rest,adaptation,decay,p.dt,p.adaptation_tau);
}
