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
};

struct KCEvent { long tick; int neuron; int reserved; };

inline bool ring_contains(device atomic_uint *ring,uint words,uint slot,uint neuron) {
  uint word=atomic_load_explicit(&ring[slot*words+(neuron>>5)],memory_order_relaxed);
  return (word&(1u<<(neuron&31)))!=0;
}

inline void evolve(uint i,long now,float current,device float *v,device float *g,
    device short *refractory,device long *last,device const float *rest,
    device float *adaptation,float dt,float adaptation_tau) {
  long d=now-last[i];if(d<=0)return;
  int frozen=refractory[i]>0?refractory[i]-1:0;
  int skip=min((int)d,frozen);
  if(skip>0&&adaptation[i]>0.0f)adaptation[i]*=exp(-dt*skip/adaptation_tau);
  refractory[i]=d>=refractory[i]?0:refractory[i]-(short)d;d-=skip;
  if(d>0){
    float a=exp(-dt*d/20.0f),b=exp(-dt*d/5.0f);
    v[i]=rest[i]+(v[i]-rest[i])*a+current*(1.0f-a)+g[i]*(a-b)/3.0f;
    g[i]*=b;
    if(adaptation[i]>0.0f){
      float c=exp(-dt*d/adaptation_tau);
      v[i]-=adaptation[i]*adaptation_tau/(adaptation_tau-20.0f)*(c-a);
      adaptation[i]*=c;
    }
  }
  last[i]=now;
}

kernel void df_drive_change(device float *v [[buffer(0)]],device float *g [[buffer(1)]],
    device short *refractory [[buffer(2)]],device const float *drive [[buffer(3)]],
    device float *previous [[buffer(4)]],device uchar *active [[buffer(5)]],
    device long *last [[buffer(6)]],device const float *rest [[buffer(7)]],
    device float *adaptation [[buffer(8)]],constant Params &p [[buffer(9)]],
    uint i [[thread_position_in_grid]]) {
  if(i>=p.neurons||drive[i]==previous[i])return;
  evolve(i,p.clock-1,previous[i],v,g,refractory,last,rest,adaptation,p.dt,p.adaptation_tau);
  previous[i]=drive[i];active[i]=1;
}

kernel void df_integrate(device float *v [[buffer(0)]],device float *g [[buffer(1)]],
    device short *refractory [[buffer(2)]],device const float *drive [[buffer(3)]],
    device uchar *active [[buffer(4)]],device long *last [[buffer(5)]],
    device const float *rest [[buffer(6)]],device float *adaptation [[buffer(7)]],
    device atomic_uint *ring [[buffer(8)]],device int *counts [[buffer(9)]],
    device const uchar *kc_mask [[buffer(10)]],device KCEvent *events [[buffer(11)]],
    device atomic_uint *event_count [[buffer(12)]],constant Params &p [[buffer(13)]],
    uint i [[thread_position_in_grid]]) {
  if(i>=p.neurons||active[i]==0)return;
  evolve(i,p.clock,drive[i],v,g,refractory,last,rest,adaptation,p.dt,p.adaptation_tau);
  if(refractory[i]==0&&v[i]>-45.0f){
    atomic_fetch_or_explicit(&ring[p.future*p.words+(i>>5)],1u<<(i&31),memory_order_relaxed);
    counts[i]++;
    if(kc_mask[i]||p.capture_all_spikes){
      uint position=atomic_fetch_add_explicit(event_count,1u,memory_order_relaxed);
      if(position<p.event_capacity)events[position]=KCEvent{p.clock,(int)i,0};
    }
    if(kc_mask[i]){
      adaptation[i]+=p.adaptation_jump;
    }
  }
  float gap=-45.0f-rest[i];
  bool can_fire=v[i]>-45.0f||drive[i]>gap||drive[i]+g[i]>gap;
  if(!can_fire)active[i]=0;
}

kernel void df_mark_targets(device const long *out_ptr [[buffer(0)]],
    device const int *out_post [[buffer(1)]],device const int *edge_to_incoming [[buffer(2)]],
    device atomic_uint *ring [[buffer(3)]],device atomic_uint *touched [[buffer(4)]],
    device atomic_uint *active_edge_bits [[buffer(5)]],device uint *active_edge_words [[buffer(6)]],
    device atomic_uint *active_word_count [[buffer(7)]],device atomic_uint *clear_dispatch [[buffer(8)]],
    constant Params &p [[buffer(9)]],
    uint pre [[thread_position_in_grid]]) {
  if(pre>=p.neurons||!ring_contains(ring,p.words,p.slot,pre))return;
  for(long edge=out_ptr[pre];edge<out_ptr[pre+1];edge++){
    atomic_store_explicit(&touched[out_post[edge]],1u,memory_order_relaxed);
    uint position=(uint)edge_to_incoming[edge];
    uint word=position>>5;
    uint previous=atomic_fetch_or_explicit(&active_edge_bits[word],
      1u<<(position&31),memory_order_relaxed);
    if(previous==0){
      uint queue_position=atomic_fetch_add_explicit(active_word_count,1u,memory_order_relaxed);
      active_edge_words[queue_position]=word;
      uint groups=queue_position/p.clear_threadgroup_width+1;
      atomic_fetch_max_explicit(&clear_dispatch[0],groups,memory_order_relaxed);
    }
  }
}

kernel void df_gather_targets(device const long *in_ptr [[buffer(0)]],
    device const int *in_pre [[buffer(1)]],device const int *in_edge [[buffer(2)]],
    device const float *weight [[buffer(3)]],device atomic_uint *active_edge_bits [[buffer(4)]],
    device atomic_uint *touched [[buffer(5)]],device float *v [[buffer(6)]],
    device float *g [[buffer(7)]],device short *refractory [[buffer(8)]],
    device const float *drive [[buffer(9)]],device uchar *active [[buffer(10)]],
    device long *last [[buffer(11)]],device float *modulation [[buffer(12)]],
    device long *modulation_last [[buffer(13)]],device const uchar *modulation_mask [[buffer(14)]],
    device const float *rest [[buffer(15)]],device float *adaptation [[buffer(16)]],
    constant Params &p [[buffer(17)]],uint target [[thread_position_in_grid]]) {
  if(target>=p.neurons||atomic_exchange_explicit(&touched[target],0u,memory_order_relaxed)==0)return;
  evolve(target,p.clock,drive[target],v,g,refractory,last,rest,adaptation,p.dt,p.adaptation_tau);
  float conductance=0.0f,modulatory=0.0f;bool has_fast=false,has_modulatory=false;
  long start=in_ptr[target],end=in_ptr[target+1];
  if(start<end){
    uint first=(uint)start>>5,last_word=(uint)(end-1)>>5;
    for(uint word=first;word<=last_word;word++){
      uint bits=atomic_load_explicit(&active_edge_bits[word],memory_order_relaxed);
      uint low=word==first?((uint)start&31):0;
      uint high=word==last_word?(((uint)(end-1)&31)+1):32;
      if(low>0)bits&=0xffffffffu<<low;
      if(high<32)bits&=(1u<<high)-1;
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
    long delta=p.clock-modulation_last[target];
    modulation[target]*=exp(-p.dt*delta/100.0f);
    modulation[target]+=modulatory;modulation_last[target]=p.clock;
  }
  if(has_fast){g[target]+=conductance;active[target]=1;}
}

kernel void df_clear_active_edge_words(device atomic_uint *active_edge_bits [[buffer(0)]],
    device const uint *active_edge_words [[buffer(1)]],
    device atomic_uint *active_word_count [[buffer(2)]],constant Params &p [[buffer(3)]],
    uint position [[thread_position_in_grid]]) {
  uint count=atomic_load_explicit(active_word_count,memory_order_relaxed);
  if(position<count){
    uint word=active_edge_words[position];
    if(word<p.edge_words)atomic_store_explicit(&active_edge_bits[word],0u,memory_order_relaxed);
  }
}

kernel void df_finalize_tick(device atomic_uint *ring [[buffer(0)]],
    device float *v [[buffer(1)]],device float *g [[buffer(2)]],
    device short *refractory [[buffer(3)]],device const float *rest [[buffer(4)]],
    device atomic_uint *active_word_count [[buffer(5)]],
    device atomic_uint *clear_dispatch [[buffer(6)]],constant Params &p [[buffer(7)]],
    uint i [[thread_position_in_grid]]) {
  if(i<p.words)atomic_store_explicit(&ring[p.slot*p.words+i],0u,memory_order_relaxed);
  if(i<p.neurons&&ring_contains(ring,p.words,p.future,i)){
    v[i]=rest[i];g[i]=0.0f;refractory[i]=(short)p.refractory_ticks;
  }
  if(i==0){
    atomic_store_explicit(active_word_count,0u,memory_order_relaxed);
    atomic_store_explicit(&clear_dispatch[0],1u,memory_order_relaxed);
  }
}

kernel void df_materialize(device float *v [[buffer(0)]],device float *g [[buffer(1)]],
    device short *refractory [[buffer(2)]],device const float *drive [[buffer(3)]],
    device long *last [[buffer(4)]],device const float *rest [[buffer(5)]],
    device float *adaptation [[buffer(6)]],constant Params &p [[buffer(7)]],
    uint i [[thread_position_in_grid]]) {
  if(i<p.neurons)evolve(i,p.clock,drive[i],v,g,refractory,last,rest,adaptation,p.dt,p.adaptation_tau);
}
