#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "api.h"
#include "decay_tables.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace {
thread_local std::string last_error;

struct Backend {
  __strong id<MTLDevice> device;
  __strong id<MTLCommandQueue> command_queue;
  __strong id<MTLLibrary> library;
  int32_t neurons=0;
  int64_t edges=0;
  int32_t slots=0;
  int32_t words=0;
  int32_t lanes=1;
  int32_t window_ticks=0;
  std::vector<int64_t> cursors;
  std::vector<uint8_t> uploaded;
  bool rest_initialized=false;
  float dt=0.1f;
  float adaptation_jump=8.0f;
  float adaptation_tau=200.0f;
  __strong id<MTLBuffer> out_ptr,out_post,in_ptr,in_pre,in_edge,kc_mask,modulation_mask;
  __strong id<MTLBuffer> edge_to_incoming,active_edge_bits;
  __strong id<MTLBuffer> window_touched,window_edge_bits;
  __strong id<MTLBuffer> weight,v,g,refractory,drive,previous_drive,counts,active_flag,last;
  __strong id<MTLBuffer> modulation,modulation_last,rest,adaptation,ring,touched;
  __strong id<MTLBuffer> kc_events,event_count;
  __strong id<MTLBuffer> decay_tables;
  __strong id<MTLBuffer> weight_update_ids,weight_update_values;
  __strong id<MTLComputePipelineState> drive_pipeline,integrate_mark_pipeline;
  __strong id<MTLComputePipelineState> gather_finalize_pipeline;
  __strong id<MTLComputePipelineState> window_prepare_pipeline,window_neuron_pipeline;
  __strong id<MTLComputePipelineState> materialize_pipeline,weight_update_pipeline;
  int32_t edge_words=0;
  uint32_t event_capacity=0;
  uint32_t last_event_count=0;
  uint32_t weight_update_capacity=0;
  bool capture_all_spikes=false;
  bool poisoned=false;
};

struct KernelParams {
  uint32_t neurons;
  uint32_t words;
  uint32_t edge_words;
  uint32_t clear_threadgroup_width;
  uint32_t slot;
  uint32_t future;
  int64_t clock;
  float dt;
  float adaptation_jump;
  float adaptation_tau;
  int32_t refractory_ticks;
  uint32_t event_capacity;
  uint32_t capture_all_spikes;
  uint32_t lanes;
  uint32_t weight_lane;
  uint64_t edge_stride;
  uint64_t ring_stride;
  uint64_t edge_bitmap_stride;
  uint32_t window_ticks;
  uint32_t window_capacity;
};
static_assert(sizeof(KernelParams)==96,"Metal Params layout must match kernels.metal");
static_assert(offsetof(KernelParams,edge_stride)==64,"Metal lane strides must be aligned");
static_assert(offsetof(KernelParams,window_ticks)==88,"Metal window width must match");
static_assert(offsetof(KernelParams,window_capacity)==92,"Metal window capacity must match");

bool valid_lane(const Backend *b,int32_t lane) { return b&&lane>=0&&lane<b->lanes; }

void poison(Backend *b) {
  b->poisoned=true;b->last_event_count=0;
  std::fill(b->uploaded.begin(),b->uploaded.end(),0);
}

bool product(std::initializer_list<size_t> factors,size_t &result) {
  result=1;
  for(size_t factor:factors){
    if(factor!=0&&result>std::numeric_limits<size_t>::max()/factor)return false;
    result*=factor;
  }
  return true;
}

uint64_t shared_bytes(const Backend *b) {
  uint64_t bytes=0;
  for(id<MTLBuffer> buffer:{b->out_ptr,b->out_post,b->in_ptr,b->in_pre,b->in_edge,
      b->edge_to_incoming,b->kc_mask,b->modulation_mask,b->rest,b->decay_tables})
    if(buffer!=nil)bytes+=buffer.length;
  return bytes;
}

uint64_t mutable_bytes(const Backend *b) {
  uint64_t bytes=0;
  for(id<MTLBuffer> buffer:{b->weight,b->v,b->g,b->refractory,b->drive,b->previous_drive,
      b->counts,b->active_flag,b->last,b->modulation,b->modulation_last,b->adaptation,
      b->ring,b->touched,b->active_edge_bits,b->kc_events,b->event_count,
      b->weight_update_ids,b->weight_update_values,b->window_touched,b->window_edge_bits})
    if(buffer!=nil)bytes+=buffer.length;
  return bytes;
}

bool replacement_fits(const Backend *b,std::initializer_list<size_t> lengths) {
  // Include old buffers until replacement succeeds, preserving failure atomicity.
  uint64_t total=shared_bytes(b)+mutable_bytes(b);
  for(size_t length:lengths){
    const size_t allocated=std::max<size_t>(length,1);
    if(allocated>b->device.maxBufferLength||
        total>std::numeric_limits<uint64_t>::max()-allocated)return false;
    total+=allocated;
  }
  return total<=b->device.recommendedMaxWorkingSetSize;
}

void *lane_contents(id<MTLBuffer> buffer,int32_t lane,size_t stride) {
  return static_cast<uint8_t *>(buffer.contents)+size_t(lane)*stride;
}

Backend *cast(df_metal_handle handle) { return static_cast<Backend *>(handle); }

int fail(const char *message) {
  last_error=message==nullptr?"Unknown Metal backend error":message;
  return 1;
}

int fail(NSString *message) { return fail(message.UTF8String); }

template<class T> bool present(T *pointer) { return pointer!=nullptr; }

id<MTLBuffer> make_buffer(id<MTLDevice> device,const void *bytes,size_t length) {
  const size_t allocated=std::max<size_t>(length,1);
  id<MTLBuffer> buffer=[device newBufferWithLength:allocated options:MTLResourceStorageModeShared];
  if(buffer!=nil&&bytes!=nullptr&&length>0)std::memcpy(buffer.contents,bytes,length);
  return buffer;
}

id<MTLComputePipelineState> make_pipeline(Backend *backend,NSString *name,NSError **error) {
  id<MTLFunction> function=[backend->library newFunctionWithName:name];
  if(function==nil){
    if(error!=nullptr)*error=[NSError errorWithDomain:@"org.doomfly.metal" code:1
      userInfo:@{NSLocalizedDescriptionKey:[NSString stringWithFormat:@"Missing Metal kernel %@",name]}];
    return nil;
  }
  return [backend->device newComputePipelineStateWithFunction:function error:error];
}

void encode(id<MTLComputeCommandEncoder> encoder,id<MTLComputePipelineState> pipeline,
    std::initializer_list<id<MTLBuffer>> buffers,const KernelParams &params,NSUInteger threads,
    uint32_t &dispatch_count) {
  [encoder setComputePipelineState:pipeline];NSUInteger index=0;
  for(id<MTLBuffer> buffer:buffers)[encoder setBuffer:buffer offset:0 atIndex:index++];
  [encoder setBytes:&params length:sizeof(params) atIndex:index];
  NSUInteger width=std::min<NSUInteger>(256,pipeline.maxTotalThreadsPerThreadgroup);
  [encoder dispatchThreads:MTLSizeMake(threads,params.lanes,1)
    threadsPerThreadgroup:MTLSizeMake(width,1,1)];
  dispatch_count++;
}

bool state_pointers_present(const Backend *backend,const df_metal_state *s) {
  return backend!=nullptr&&s!=nullptr&&(backend->edges==0||present(s->weight))&&
    present(s->v)&&present(s->g)&&
    present(s->refractory)&&present(s->drive)&&present(s->previous_drive)&&
    present(s->queue)&&present(s->queue_count)&&present(s->counts)&&
    present(s->active)&&present(s->active_flag)&&present(s->nactive)&&
    present(s->last)&&present(s->modulation)&&present(s->modulation_last)&&
    present(s->rest)&&present(s->adaptation);
}

bool set_device_info(id<MTLDevice> device,df_metal_device_info *info) {
  std::memset(info,0,sizeof(*info));
  info->abi_version=DF_METAL_ABI_VERSION;
  if(device==nil)return false;
  info->has_unified_memory=device.hasUnifiedMemory?1u:0u;
  if(@available(macOS 11.0,*))
    info->supports_apple7=[device supportsFamily:MTLGPUFamilyApple7]?1u:0u;
  info->recommended_working_set=device.recommendedMaxWorkingSetSize;
  info->registry_id=device.registryID;
  const char *name=device.name.UTF8String;
  if(name!=nullptr)std::strncpy(info->device_name,name,sizeof(info->device_name)-1);
  info->supported=info->has_unified_memory&&info->supports_apple7;
  return info->supported;
}
}

extern "C" const char *df_metal_last_error(void) { return last_error.c_str(); }

extern "C" int df_metal_probe(df_metal_device_info *info) {
  if(info==nullptr)return fail("Device information pointer is null");
  @autoreleasepool {
    if(!set_device_info(MTLCreateSystemDefaultDevice(),info))
      return fail("Metal requires unified memory and Apple GPU family 7");
  }
  last_error.clear();
  return 0;
}

extern "C" int df_metal_create(const df_metal_graph *graph,const char *metallib_path,
    df_metal_handle *handle) {
  return df_metal_create_batch(graph,metallib_path,1,handle);
}

extern "C" int df_metal_create_batch(const df_metal_graph *graph,const char *metallib_path,
    int32_t lanes,df_metal_handle *handle) {
  return df_metal_create_batch_windowed(graph,metallib_path,lanes,0,handle);
}

extern "C" int df_metal_create_batch_windowed(const df_metal_graph *graph,const char *metallib_path,
    int32_t lanes,int32_t window_ticks,df_metal_handle *handle) {
  if(handle!=nullptr)*handle=nullptr;
  if(window_ticks<0||window_ticks>18)return fail("Metal window ticks must be in 0..18");
  try {
  if(graph==nullptr||handle==nullptr||metallib_path==nullptr)
    return fail("Metal create argument is null");
  *handle=nullptr;
  if(lanes<1||graph->neurons<1||graph->edges<0||graph->edges>std::numeric_limits<int32_t>::max()||
      graph->delay_slots<1||graph->dt_ms!=0.1f)
    return fail("Invalid Metal graph dimensions");
  if(graph->delay_slots!=19)return fail("Metal delay slot count must be 19");
  if(!std::isfinite(graph->adaptation_jump_mv)||!std::isfinite(graph->adaptation_tau_ms)||
      graph->adaptation_tau_ms<=0)return fail("Invalid Metal adaptation configuration");
  if(!present(graph->out_ptr)||!present(graph->in_ptr)||!present(graph->kc_mask)||
      !present(graph->modulation_mask)||(graph->edges>0&&(!present(graph->out_post)||
      !present(graph->in_pre)||!present(graph->in_edge))))return fail("Metal graph pointer is null");
  // Bounds and allocation products precede traversal of user-provided arrays.
  size_t neuron_cells=0,edge_cells=0,ring_words=0,event_bytes=0;
  size_t window_touched_bytes=0,window_edge_bytes=0;
  const size_t n=size_t(graph->neurons),e=size_t(graph->edges);
  const size_t words=(n+31)/32,edge_words=(e+31)/32;
  if(!product({n,size_t(lanes)},neuron_cells)||
      !product({e,size_t(lanes)},edge_cells)||
      !product({size_t(graph->delay_slots),words,size_t(lanes)},ring_words)||
      !product({neuron_cells,5,sizeof(df_metal_kc_event)},event_bytes)||
      !product({size_t(window_ticks),size_t(lanes),n,sizeof(uint32_t)},window_touched_bytes)||
      !product({size_t(window_ticks),size_t(lanes),edge_words,sizeof(uint32_t)},window_edge_bytes)||
      neuron_cells>std::numeric_limits<uint32_t>::max()/100u)
    return fail("Metal batch dimensions exceed checked grid limits");
  @autoreleasepool {
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    df_metal_device_info info;
    if(!set_device_info(device,&info))
      return fail("Metal requires unified memory and Apple GPU family 7");
    uint64_t total=0;
    const size_t lengths[]={(n+1)*8,e*4,(n+1)*8,e*4,e*4,e*4,n,n,n*4,3072*4,
      edge_cells*4,neuron_cells*4,neuron_cells*4,neuron_cells*2,neuron_cells*4,
      neuron_cells*4,neuron_cells*4,neuron_cells,neuron_cells*8,neuron_cells*4,
      neuron_cells*8,neuron_cells*4,ring_words*4,neuron_cells*4,
      edge_words*size_t(lanes)*4,event_bytes,4};
    for(size_t length:lengths){
      size_t allocated=std::max<size_t>(length,1);
      if(allocated>device.maxBufferLength||total>std::numeric_limits<uint64_t>::max()-allocated)
        return fail("Metal batch allocation exceeds device buffer limit");
      total+=allocated;
    }
    if(window_ticks>0)for(size_t length:{window_touched_bytes,window_edge_bytes}){
      size_t allocated=std::max<size_t>(length,1);
      if(allocated>device.maxBufferLength||total>std::numeric_limits<uint64_t>::max()-allocated)
        return fail("Metal window allocation exceeds device buffer limit");
      total+=allocated;
    }
    if(total>device.recommendedMaxWorkingSetSize)
      return fail("Metal batch allocation exceeds recommended working set");
  }
  if(graph->out_ptr[0]!=0||graph->out_ptr[graph->neurons]!=graph->edges||
      graph->in_ptr[0]!=0||graph->in_ptr[graph->neurons]!=graph->edges)
    return fail("Metal graph CSR bounds mismatch");
  std::vector<int32_t> edge_to_incoming(size_t(graph->edges),-1);
  for(size_t neuron=0;neuron<n;neuron++){
    if(graph->out_ptr[neuron]<0||graph->out_ptr[neuron]>graph->out_ptr[neuron+1]||
        graph->out_ptr[neuron+1]>graph->edges||graph->in_ptr[neuron]<0||
        graph->in_ptr[neuron]>graph->in_ptr[neuron+1]||graph->in_ptr[neuron+1]>graph->edges)
      return fail("Metal graph CSR ordering is invalid");
  }
  for(int64_t position=0;position<graph->edges;position++){
    int32_t edge=graph->in_edge[position];
    if(edge<0||int64_t(edge)>=graph->edges||edge_to_incoming[edge]!=-1)
      return fail("Metal incoming edge permutation is invalid");
    edge_to_incoming[edge]=int32_t(position);
    if(graph->out_post[position]<0||graph->out_post[position]>=graph->neurons||
        graph->in_pre[position]<0||graph->in_pre[position]>=graph->neurons)
      return fail("Metal graph neuron index is invalid");
  }
  @autoreleasepool {
    auto owner=std::make_unique<Backend>();
    Backend *backend=owner.get();
    backend->window_ticks=window_ticks;
    backend->device=MTLCreateSystemDefaultDevice();
    df_metal_device_info info;
    if(!set_device_info(backend->device,&info)){
      return fail("Metal requires unified memory and Apple GPU family 7");
    }
    backend->command_queue=[backend->device newCommandQueue];
    NSError *error=nil;
    NSString *path=[NSString stringWithUTF8String:metallib_path];
    backend->library=[backend->device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&error];
    if(backend->command_queue==nil||backend->library==nil){
      std::string message=error==nil?"Metal command queue or library creation failed":error.localizedDescription.UTF8String;
      return fail(message.c_str());
    }
    backend->drive_pipeline=make_pipeline(backend,@"df_drive_change",&error);
    backend->integrate_mark_pipeline=make_pipeline(backend,@"df_integrate_mark",&error);
    backend->gather_finalize_pipeline=make_pipeline(backend,@"df_gather_finalize",&error);
    backend->materialize_pipeline=make_pipeline(backend,@"df_materialize",&error);
    backend->weight_update_pipeline=make_pipeline(backend,@"df_update_weights",&error);
    if(window_ticks>0){
      backend->window_prepare_pipeline=make_pipeline(backend,@"df_window_prepare",&error);
      backend->window_neuron_pipeline=make_pipeline(backend,@"df_window_neurons",&error);
    }
    if(backend->drive_pipeline==nil||backend->integrate_mark_pipeline==nil||
        backend->gather_finalize_pipeline==nil||
        backend->materialize_pipeline==nil||backend->weight_update_pipeline==nil||
        (window_ticks>0&&(backend->window_prepare_pipeline==nil||backend->window_neuron_pipeline==nil))){
      std::string message=error==nil?"Metal pipeline creation failed":error.localizedDescription.UTF8String;
      return fail(message.c_str());
    }
    const uint64_t word_count=(uint64_t(graph->neurons)+31u)/32u;
    if(word_count>uint64_t(std::numeric_limits<int32_t>::max())){
      return fail("Metal neuron bitmap is too large");
    }
    backend->neurons=graph->neurons;backend->edges=graph->edges;
    backend->lanes=lanes;backend->cursors.assign(lanes,0);backend->uploaded.assign(lanes,0);
    backend->slots=graph->delay_slots;backend->words=int32_t(word_count);
    backend->edge_words=int32_t((graph->edges+31)/32);
    backend->dt=graph->dt_ms;backend->adaptation_jump=graph->adaptation_jump_mv;
    backend->adaptation_tau=graph->adaptation_tau_ms;
    const auto decay_tables=doomfly::metal::make_decay_tables(backend->dt,backend->adaptation_tau);
    backend->decay_tables=make_buffer(backend->device,decay_tables.data(),
      decay_tables.size()*sizeof(float));
    backend->out_ptr=make_buffer(backend->device,graph->out_ptr,(n+1)*sizeof(int64_t));
    backend->out_post=make_buffer(backend->device,graph->out_post,e*sizeof(int32_t));
    backend->in_ptr=make_buffer(backend->device,graph->in_ptr,(n+1)*sizeof(int64_t));
    backend->in_pre=make_buffer(backend->device,graph->in_pre,e*sizeof(int32_t));
    backend->in_edge=make_buffer(backend->device,graph->in_edge,e*sizeof(int32_t));
    backend->edge_to_incoming=make_buffer(backend->device,edge_to_incoming.data(),
      e*sizeof(int32_t));
    backend->active_edge_bits=make_buffer(backend->device,nullptr,
      size_t(backend->edge_words)*size_t(lanes)*sizeof(uint32_t));
    backend->kc_mask=make_buffer(backend->device,graph->kc_mask,n);
    backend->modulation_mask=make_buffer(backend->device,graph->modulation_mask,n);
    backend->weight=make_buffer(backend->device,nullptr,edge_cells*sizeof(float));
    backend->v=make_buffer(backend->device,nullptr,neuron_cells*sizeof(float));
    backend->g=make_buffer(backend->device,nullptr,neuron_cells*sizeof(float));
    backend->refractory=make_buffer(backend->device,nullptr,neuron_cells*sizeof(int16_t));
    backend->drive=make_buffer(backend->device,nullptr,neuron_cells*sizeof(float));
    backend->previous_drive=make_buffer(backend->device,nullptr,neuron_cells*sizeof(float));
    backend->counts=make_buffer(backend->device,nullptr,neuron_cells*sizeof(int32_t));
    backend->active_flag=make_buffer(backend->device,nullptr,neuron_cells);
    backend->last=make_buffer(backend->device,nullptr,neuron_cells*sizeof(int64_t));
    backend->modulation=make_buffer(backend->device,nullptr,neuron_cells*sizeof(float));
    backend->modulation_last=make_buffer(backend->device,nullptr,neuron_cells*sizeof(int64_t));
    backend->rest=make_buffer(backend->device,nullptr,n*sizeof(float));
    backend->adaptation=make_buffer(backend->device,nullptr,neuron_cells*sizeof(float));
    backend->ring=make_buffer(backend->device,nullptr,ring_words*sizeof(uint32_t));
    backend->touched=make_buffer(backend->device,nullptr,neuron_cells*sizeof(uint32_t));
    if(window_ticks>0){
      backend->window_touched=make_buffer(backend->device,nullptr,window_touched_bytes);
      backend->window_edge_bits=make_buffer(backend->device,nullptr,window_edge_bytes);
      if(backend->window_touched==nil||backend->window_edge_bits==nil)
        return fail("Metal window buffer allocation failed");
      std::memset(backend->window_touched.contents,0,backend->window_touched.length);
      std::memset(backend->window_edge_bits.contents,0,backend->window_edge_bits.length);
    }
    uint32_t kc_count=0;
    for(size_t i=0;i<n;i++)if(graph->kc_mask[i])kc_count++;
    backend->event_capacity=std::max<uint32_t>(1,kc_count*5u*uint32_t(lanes));
    backend->kc_events=make_buffer(backend->device,nullptr,
      size_t(backend->event_capacity)*sizeof(df_metal_kc_event));
    backend->event_count=make_buffer(backend->device,nullptr,sizeof(uint32_t));
    const id<MTLBuffer> required[]={backend->out_ptr,backend->out_post,backend->in_ptr,
      backend->in_pre,backend->in_edge,backend->kc_mask,backend->modulation_mask,
      backend->edge_to_incoming,backend->active_edge_bits,
      backend->weight,backend->v,backend->g,backend->refractory,backend->drive,
      backend->previous_drive,backend->counts,backend->active_flag,backend->last,
      backend->modulation,backend->modulation_last,backend->rest,backend->adaptation,
      backend->ring,backend->touched,backend->kc_events,backend->event_count,
      backend->decay_tables};
    for(id<MTLBuffer> buffer:required)if(buffer==nil){
      return fail("Metal shared buffer allocation failed");
    }
    std::memset(backend->ring.contents,0,ring_words*sizeof(uint32_t));
    std::memset(backend->touched.contents,0,neuron_cells*sizeof(uint32_t));
    std::memset(backend->active_edge_bits.contents,0,
      size_t(backend->edge_words)*size_t(lanes)*sizeof(uint32_t));
    *handle=owner.release();
  }
  last_error.clear();return 0;
  } catch(const std::exception &error) {
    return fail(error.what());
  }
}

extern "C" int df_metal_upload_state(df_metal_handle handle,const df_metal_state *s) {
  return df_metal_upload_lane_state(handle,0,s);
}

extern "C" int df_metal_upload_lane_state(df_metal_handle handle,int32_t lane,
    const df_metal_state *s) {
  try {
  Backend *b=cast(handle);
  if(b==nullptr||!state_pointers_present(b,s))return fail("Metal state pointer is null");
  if(!valid_lane(b,lane))return fail("Metal lane index out of bounds");
  const int32_t n=b->neurons,slots=b->slots,active_count=*s->nactive;
  if(s->cursor<0)return fail("Invalid Metal lane cursor");
  for(int64_t edge=0;edge<b->edges;edge++)
    if(!std::isfinite(s->weight[edge]))return fail("Nonfinite Metal state weight");
  // Sleeping histories may precede zero. Reserve 99 further ticks so evolve's
  // unchanged int(d) cast remains representable for a maximum 100-tick command.
  const int64_t history_limit=int64_t(std::numeric_limits<int32_t>::max())-99;
  for(int32_t neuron=0;neuron<n;neuron++){
    for(const float *field:{s->v,s->g,s->drive,s->previous_drive,s->modulation,s->rest,s->adaptation})
      if(!std::isfinite(field[neuron]))return fail("Nonfinite Metal neuron state");
    if(s->refractory[neuron]<0||s->counts[neuron]<0||s->active_flag[neuron]>1||
        s->last[neuron]<s->cursor-history_limit||
        s->last[neuron]>s->cursor||
        s->modulation_last[neuron]<0||s->modulation_last[neuron]>s->cursor)
      return fail("Invalid Metal discrete neuron state");
  }
  if(active_count<0||active_count>n)return fail("Invalid active neuron count");
  std::vector<uint8_t> active_seen(n,0);
  for(int32_t k=0;k<active_count;k++){
    int32_t id=s->active[k];
    if(id<0||id>=n||active_seen[id])return fail("Invalid active neuron list");
    active_seen[id]=1;
  }
  for(int32_t i=0;i<n;i++)if((s->active_flag[i]!=0)!=bool(active_seen[i]))
    return fail("Active neuron list and flags disagree");
  std::vector<uint32_t> ring(size_t(slots)*b->words,0);
  for(int32_t slot=0;slot<slots;slot++){
    int32_t count=s->queue_count[slot];
    if(count<0||count>n)return fail("Invalid delayed neuron count");
    for(int32_t q=0;q<count;q++){
      int32_t id=s->queue[size_t(slot)*n+q];
      if(id<0||id>=n)return fail("Delayed neuron out of bounds");
      uint32_t mask=uint32_t(1)<<(id&31);uint32_t &word=ring[size_t(slot)*b->words+(id>>5)];
      if(word&mask)return fail("duplicate delayed neuron");
      word|=mask;
    }
  }
  const size_t nf=size_t(n)*sizeof(float),ni=size_t(n)*sizeof(int32_t),nl=size_t(n)*sizeof(int64_t);
  if(b->lanes>1&&b->rest_initialized&&std::memcmp(b->rest.contents,s->rest,nf)!=0)
    return fail("Metal batch lanes require identical resting configuration");
  if(b->edges>0)std::memcpy(lane_contents(b->weight,lane,size_t(b->edges)*sizeof(float)),
    s->weight,size_t(b->edges)*sizeof(float));
  std::memcpy(lane_contents(b->v,lane,nf),s->v,nf);
  std::memcpy(lane_contents(b->g,lane,nf),s->g,nf);
  std::memcpy(lane_contents(b->refractory,lane,size_t(n)*sizeof(int16_t)),s->refractory,size_t(n)*sizeof(int16_t));
  std::memcpy(lane_contents(b->drive,lane,nf),s->drive,nf);
  std::memcpy(lane_contents(b->previous_drive,lane,nf),s->previous_drive,nf);
  std::memcpy(lane_contents(b->counts,lane,ni),s->counts,ni);
  std::memcpy(lane_contents(b->active_flag,lane,size_t(n)),s->active_flag,n);
  std::memcpy(lane_contents(b->last,lane,nl),s->last,nl);
  std::memcpy(lane_contents(b->modulation,lane,nf),s->modulation,nf);
  std::memcpy(lane_contents(b->modulation_last,lane,nl),s->modulation_last,nl);
  std::memcpy(b->rest.contents,s->rest,nf);b->rest_initialized=true;
  std::memcpy(lane_contents(b->adaptation,lane,nf),s->adaptation,nf);
  std::memcpy(lane_contents(b->ring,lane,ring.size()*sizeof(uint32_t)),ring.data(),ring.size()*sizeof(uint32_t));
  std::memset(lane_contents(b->touched,lane,ni),0,ni);b->cursors[lane]=s->cursor;
  const size_t edge_bytes=size_t(b->edge_words)*sizeof(uint32_t);
  std::memset(lane_contents(b->active_edge_bits,lane,edge_bytes),0,edge_bytes);
  if(b->window_ticks>0){
    const size_t touched_plane_bytes=size_t(b->window_ticks)*ni;
    const size_t edge_plane_bytes=size_t(b->window_ticks)*edge_bytes;
    std::memset(lane_contents(b->window_touched,lane,touched_plane_bytes),0,touched_plane_bytes);
    std::memset(lane_contents(b->window_edge_bits,lane,edge_plane_bytes),0,edge_plane_bytes);
  }
  b->uploaded[lane]=1;b->last_event_count=0;
  if(std::all_of(b->uploaded.begin(),b->uploaded.end(),[](uint8_t value){return value!=0;}))
    b->poisoned=false;
  last_error.clear();return 0;
  } catch(const std::exception &error) {
    return fail(error.what());
  }
}

extern "C" int df_metal_download_state(df_metal_handle handle,df_metal_state *s) {
  return df_metal_download_lane_state(handle,0,s);
}

extern "C" int df_metal_download_lane_state(df_metal_handle handle,int32_t lane,
    df_metal_state *s) {
  Backend *b=cast(handle);
  if(b==nullptr||!state_pointers_present(b,s))return fail("Metal state pointer is null");
  if(!valid_lane(b,lane))return fail("Metal lane index out of bounds");
  if(!b->uploaded[lane])return fail("Metal lane state is not uploaded");
  const int32_t n=b->neurons,slots=b->slots;
  const size_t nf=size_t(n)*sizeof(float),ni=size_t(n)*sizeof(int32_t),nl=size_t(n)*sizeof(int64_t);
  if(b->edges>0)std::memcpy(s->weight,lane_contents(b->weight,lane,size_t(b->edges)*sizeof(float)),size_t(b->edges)*sizeof(float));
  std::memcpy(s->v,lane_contents(b->v,lane,nf),nf);std::memcpy(s->g,lane_contents(b->g,lane,nf),nf);
  std::memcpy(s->refractory,lane_contents(b->refractory,lane,size_t(n)*sizeof(int16_t)),size_t(n)*sizeof(int16_t));
  std::memcpy(s->drive,lane_contents(b->drive,lane,nf),nf);
  std::memcpy(s->previous_drive,lane_contents(b->previous_drive,lane,nf),nf);
  std::memcpy(s->counts,lane_contents(b->counts,lane,ni),ni);
  std::memcpy(s->active_flag,lane_contents(b->active_flag,lane,size_t(n)),n);
  std::memcpy(s->last,lane_contents(b->last,lane,nl),nl);
  std::memcpy(s->modulation,lane_contents(b->modulation,lane,nf),nf);
  std::memcpy(s->modulation_last,lane_contents(b->modulation_last,lane,nl),nl);
  std::memcpy(s->rest,b->rest.contents,nf);
  std::memcpy(s->adaptation,lane_contents(b->adaptation,lane,nf),nf);
  const uint32_t *ring=static_cast<const uint32_t *>(lane_contents(b->ring,lane,size_t(slots)*b->words*sizeof(uint32_t)));
  std::memset(s->queue,0,size_t(slots)*n*sizeof(int32_t));
  for(int32_t slot=0;slot<slots;slot++){
    int32_t count=0;
    for(int32_t id=0;id<n;id++)if(ring[size_t(slot)*b->words+(id>>5)]&(uint32_t(1)<<(id&31)))
      s->queue[size_t(slot)*n+count++]=id;
    s->queue_count[slot]=count;
  }
  std::memset(s->active,0,ni);int32_t count=0;
  const uint8_t *flags=static_cast<const uint8_t *>(lane_contents(b->active_flag,lane,size_t(n)));
  for(int32_t id=0;id<n;id++)if(flags[id])s->active[count++]=id;
  *s->nactive=count;s->cursor=b->cursors[lane];
  last_error.clear();return 0;
}

extern "C" int df_metal_upload_drive(df_metal_handle handle,const float *drive) {
  return df_metal_upload_lane_drive(handle,0,drive);
}

extern "C" int df_metal_upload_lane_drive(df_metal_handle handle,int32_t lane,const float *drive) {
  Backend *b=cast(handle);
  if(b==nullptr||drive==nullptr)return fail("Metal drive pointer is null");
  if(!valid_lane(b,lane))return fail("Metal lane index out of bounds");
  if(b->poisoned)return fail("Metal backend is poisoned");
  for(int32_t neuron=0;neuron<b->neurons;neuron++)
    if(!std::isfinite(drive[neuron]))return fail("Nonfinite Metal drive");
  std::memcpy(lane_contents(b->drive,lane,size_t(b->neurons)*sizeof(float)),drive,size_t(b->neurons)*sizeof(float));
  last_error.clear();return 0;
}

extern "C" int df_metal_download_observation(df_metal_handle handle,int32_t *counts,
    int64_t *cursor) {
  return df_metal_download_lane_observation(handle,0,counts,cursor);
}

extern "C" int df_metal_download_lane_observation(df_metal_handle handle,int32_t lane,
    int32_t *counts,int64_t *cursor) {
  Backend *b=cast(handle);
  if(b==nullptr||counts==nullptr||cursor==nullptr)
    return fail("Metal observation pointer is null");
  if(!valid_lane(b,lane))return fail("Metal lane index out of bounds");
  if(!b->uploaded[lane])return fail("Metal lane state is not uploaded");
  std::memcpy(counts,lane_contents(b->counts,lane,size_t(b->neurons)*sizeof(int32_t)),size_t(b->neurons)*sizeof(int32_t));
  *cursor=b->cursors[lane];
  last_error.clear();return 0;
}

extern "C" int df_metal_update_weights(df_metal_handle handle,int32_t count,
    const int64_t *edge_ids,const float *values) {
  return df_metal_update_lane_weights(handle,0,count,edge_ids,values);
}

extern "C" int df_metal_update_lane_weights(df_metal_handle handle,int32_t lane,int32_t count,
    const int64_t *edge_ids,const float *values) {
  Backend *b=cast(handle);
  if(b==nullptr||count<0||(count>0&&(!present(edge_ids)||!present(values))))
    return fail("Invalid plastic weight update");
  if(!valid_lane(b,lane))return fail("Metal lane index out of bounds");
  if(b->poisoned)return fail("Metal backend is poisoned");
  for(int32_t i=0;i<count;i++)
    if(edge_ids[i]<0||edge_ids[i]>=b->edges||!std::isfinite(values[i]))
      return fail("Plastic edge or value out of bounds");
  if(count==0){last_error.clear();return 0;}
  if(uint32_t(count)>b->weight_update_capacity){
    if(!replacement_fits(b,{size_t(count)*sizeof(int64_t),size_t(count)*sizeof(float)}))
      return fail("Metal plastic weight allocation exceeds device limits");
    id<MTLBuffer> ids=make_buffer(b->device,nullptr,size_t(count)*sizeof(int64_t));
    id<MTLBuffer> values_buffer=make_buffer(b->device,nullptr,size_t(count)*sizeof(float));
    if(ids==nil||values_buffer==nil)
      return fail("Metal plastic weight buffer allocation failed");
    b->weight_update_ids=ids;b->weight_update_values=values_buffer;
    b->weight_update_capacity=uint32_t(count);
  }
  std::memcpy(b->weight_update_ids.contents,edge_ids,size_t(count)*sizeof(int64_t));
  std::memcpy(b->weight_update_values.contents,values,size_t(count)*sizeof(float));
  @autoreleasepool {
    id<MTLCommandBuffer> command=[b->command_queue commandBuffer];
    if(command==nil){poison(b);return fail("Metal weight command buffer creation failed");}
    id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    if(encoder==nil){poison(b);return fail("Metal weight encoder creation failed");}
    KernelParams p{};p.neurons=uint32_t(count);p.lanes=1;
    p.weight_lane=uint32_t(lane);p.edge_stride=uint64_t(b->edges);uint32_t dispatch_count=0;
    encode(encoder,b->weight_update_pipeline,
      {b->weight,b->weight_update_ids,b->weight_update_values},p,count,dispatch_count);
    [encoder endEncoding];[command commit];[command waitUntilCompleted];
    if(command.status!=MTLCommandBufferStatusCompleted){
      poison(b);
      if(command.error==nil)return fail("Metal weight command failed");
      return fail(command.error.localizedDescription);
    }
  }
  last_error.clear();return 0;
}

extern "C" int df_metal_advance(df_metal_handle handle,int32_t steps,
    df_metal_kc_event *events,int32_t event_capacity,int32_t *event_count,
    df_metal_timing *timing) {
  Backend *b=cast(handle);
  if(b==nullptr||event_count==nullptr||timing==nullptr||steps<1||steps>100||
      event_capacity<0||(event_capacity>0&&events==nullptr))return fail("Invalid Metal advance arguments");
  if(b->poisoned)return fail("Metal backend is poisoned");
  if(!std::all_of(b->uploaded.begin(),b->uploaded.end(),[](uint8_t value){return value!=0;}))
    return fail("Metal advance requires every lane state to be uploaded");
  const int64_t cursor=b->cursors[0];
  for(int64_t lane_cursor:b->cursors)
    if(lane_cursor!=cursor)return fail("Metal batch lane cursors differ");
  const int32_t delay=std::lround(1.8f/b->dt);
  if(cursor>std::numeric_limits<int64_t>::max()-steps-delay)
    return fail("Metal batch cursor overflow");
  std::memset(timing,0,sizeof(*timing));
  timing->edge_bitmap_words=uint32_t(b->edge_words);
  const uint32_t kernel_capacity=std::min<uint32_t>(b->event_capacity,event_capacity);
  *static_cast<uint32_t *>(b->event_count.contents)=0;
  const auto started=std::chrono::steady_clock::now();
  const auto clear_started=std::chrono::steady_clock::now();
  std::memset(b->counts.contents,0,size_t(b->neurons)*size_t(b->lanes)*sizeof(int32_t));
  timing->counts_clear_seconds=std::chrono::duration<double>(
    std::chrono::steady_clock::now()-clear_started).count();
  @autoreleasepool {
    const auto encode_started=std::chrono::steady_clock::now();
    id<MTLCommandBuffer> command=[b->command_queue commandBuffer];
    if(command==nil){poison(b);return fail("Metal command buffer creation failed");}
    id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    if(encoder==nil){poison(b);return fail("Metal compute encoder creation failed");}
    uint32_t dispatch_count=0;
    KernelParams p{uint32_t(b->neurons),uint32_t(b->words),uint32_t(b->edge_words),
      0,0,0,cursor,b->dt,
      b->adaptation_jump,b->adaptation_tau,int32_t(std::lround(2.2f/b->dt)),kernel_capacity,
      b->capture_all_spikes?1u:0u,uint32_t(b->lanes),0,uint64_t(b->edges),
      uint64_t(b->slots)*uint64_t(b->words),uint64_t(b->edge_words)};
    encode(encoder,b->drive_pipeline,{b->v,b->g,b->refractory,b->drive,b->previous_drive,
      b->active_flag,b->last,b->rest,b->adaptation,b->decay_tables},p,b->neurons,dispatch_count);
    if(b->window_ticks>0){
      for(int32_t done=0;done<steps;){
        const uint32_t width=uint32_t(std::min(b->window_ticks,steps-done));
        p.clock=cursor+done;p.window_ticks=width;p.window_capacity=uint32_t(b->window_ticks);
        encode(encoder,b->window_prepare_pipeline,{b->ring,b->out_ptr,b->out_post,
          b->edge_to_incoming,b->window_touched,b->window_edge_bits},p,
          uint32_t(b->words)*width,dispatch_count);
        encode(encoder,b->window_neuron_pipeline,{b->v,b->g,b->refractory,b->drive,
          b->active_flag,b->last,b->rest,b->adaptation,b->ring,b->counts,b->kc_mask,
          b->kc_events,b->event_count,b->in_ptr,b->in_pre,b->in_edge,b->weight,
          b->window_touched,b->window_edge_bits,b->modulation,b->modulation_last,
          b->modulation_mask,b->decay_tables},p,b->neurons,dispatch_count);
        done+=int32_t(width);
      }
    }else for(int32_t step=0;step<steps;step++){
      p.clock=cursor+step;p.slot=uint32_t(p.clock%b->slots);
      p.future=uint32_t((p.clock+delay)%b->slots);
      encode(encoder,b->integrate_mark_pipeline,{b->v,b->g,b->refractory,b->drive,
        b->active_flag,b->last,b->rest,b->adaptation,b->ring,b->counts,b->kc_mask,
        b->kc_events,b->event_count,b->out_ptr,b->out_post,b->edge_to_incoming,
        b->touched,b->active_edge_bits,b->decay_tables},p,b->neurons,dispatch_count);
      encode(encoder,b->gather_finalize_pipeline,{b->in_ptr,b->in_pre,b->in_edge,b->weight,
        b->active_edge_bits,
        b->touched,b->v,b->g,b->refractory,b->drive,b->active_flag,b->last,b->modulation,
        b->modulation_last,b->modulation_mask,b->rest,b->adaptation,b->ring,b->decay_tables},
        p,b->neurons,dispatch_count);
    }
    p.clock=cursor+steps-1;
    encode(encoder,b->materialize_pipeline,{b->v,b->g,b->refractory,b->drive,b->last,
      b->rest,b->adaptation,b->decay_tables},p,b->neurons,dispatch_count);
    [encoder endEncoding];
    timing->encode_seconds=std::chrono::duration<double>(
      std::chrono::steady_clock::now()-encode_started).count();
    const auto commit_started=std::chrono::steady_clock::now();
    [command commit];
    timing->commit_call_seconds=std::chrono::duration<double>(
      std::chrono::steady_clock::now()-commit_started).count();
    const auto wait_started=std::chrono::steady_clock::now();
    [command waitUntilCompleted];
    timing->wait_call_seconds=std::chrono::duration<double>(
      std::chrono::steady_clock::now()-wait_started).count();
    if(command.status!=MTLCommandBufferStatusCompleted){
      poison(b);
      if(command.error==nil)return fail("Metal command failed");
      return fail(command.error.localizedDescription);
    }
    const auto event_copy_started=std::chrono::steady_clock::now();
    const uint32_t produced=*static_cast<uint32_t *>(b->event_count.contents);
    if(produced>kernel_capacity){poison(b);return fail("KC event capacity exceeded");}
    if(produced>0)std::memcpy(events,b->kc_events.contents,size_t(produced)*sizeof(df_metal_kc_event));
    timing->native_event_copy_seconds=std::chrono::duration<double>(
      std::chrono::steady_clock::now()-event_copy_started).count();
    timing->native_event_copy_bytes=uint64_t(produced)*sizeof(df_metal_kc_event);
    *event_count=int32_t(produced);b->last_event_count=produced;
    for(int64_t &lane_cursor:b->cursors)lane_cursor+=steps;
    timing->gpu_seconds=command.GPUEndTime>=command.GPUStartTime?
      command.GPUEndTime-command.GPUStartTime:0.0;
    timing->encoder_count=1;timing->dispatch_count=dispatch_count;
    timing->indirect_dispatch_count=0;
    timing->mark_grid_threads=uint32_t(steps)*uint32_t(b->window_ticks>0?b->words:b->neurons)*uint32_t(b->lanes);
    const uint32_t workers=b->window_ticks>0?uint32_t((steps+b->window_ticks-1)/b->window_ticks):uint32_t(steps);
    timing->gather_grid_threads=workers*uint32_t(b->neurons)*uint32_t(b->lanes);
  }
  timing->native_total_seconds=std::chrono::duration<double>(
    std::chrono::steady_clock::now()-started).count();
  last_error.clear();return 0;
}

extern "C" int df_metal_apply_eligibility(df_metal_handle handle,double *eligibility,
    int64_t *eligibility_last,double tau_ms) {
  return df_metal_apply_lane_eligibility(handle,0,eligibility,eligibility_last,tau_ms);
}

extern "C" int df_metal_apply_lane_eligibility(df_metal_handle handle,int32_t lane,
    double *eligibility,int64_t *eligibility_last,double tau_ms) {
  try {
  Backend *b=cast(handle);
  if(b==nullptr||eligibility==nullptr||eligibility_last==nullptr||
      !std::isfinite(tau_ms)||tau_ms<=0)return fail("Invalid eligibility update");
  if(!valid_lane(b,lane))return fail("Metal lane index out of bounds");
  if(b->poisoned)return fail("Metal backend is poisoned");
  const auto *stored=static_cast<const df_metal_kc_event *>(b->kc_events.contents);
  std::vector<df_metal_kc_event> events(stored,stored+b->last_event_count);
  std::sort(events.begin(),events.end(),[](const auto &left,const auto &right){
    return left.tick<right.tick||(left.tick==right.tick&&left.neuron<right.neuron);
  });
  for(const auto &event:events){
    if(event.reserved!=lane)continue;
    if(event.neuron<0||event.neuron>=b->neurons)return fail("KC event neuron out of bounds");
    const auto *kc_mask=static_cast<const uint8_t *>(b->kc_mask.contents);
    if(!kc_mask[event.neuron])continue;
    const int64_t delta=event.tick-eligibility_last[event.neuron];
    const float decay=std::exp(-b->dt*float(delta)/float(tau_ms));
    eligibility[event.neuron]*=decay;
    eligibility[event.neuron]+=1.0;eligibility_last[event.neuron]=event.tick;
  }
  last_error.clear();return 0;
  } catch(const std::exception &error) {
    return fail(error.what());
  }
}

extern "C" int df_metal_set_diagnostics(df_metal_handle handle,int32_t enabled) {
  Backend *b=cast(handle);
  if(b==nullptr||(enabled!=0&&enabled!=1))return fail("Invalid Metal diagnostic mode");
  if(b->poisoned)return fail("Metal backend is poisoned");
  if(enabled&&b->event_capacity<uint32_t(b->neurons)*5u*uint32_t(b->lanes)){
    const uint32_t capacity=uint32_t(b->neurons)*5u*uint32_t(b->lanes);
    if(!replacement_fits(b,{size_t(capacity)*sizeof(df_metal_kc_event)}))
      return fail("Metal diagnostic event allocation exceeds device limits");
    id<MTLBuffer> events=make_buffer(b->device,nullptr,size_t(capacity)*sizeof(df_metal_kc_event));
    if(events==nil)return fail("Metal diagnostic event allocation failed");
    b->kc_events=events;b->event_capacity=capacity;
  }
  b->capture_all_spikes=enabled!=0;last_error.clear();return 0;
}

extern "C" int df_metal_batch_memory_bytes(df_metal_handle handle,uint64_t *shared,
    uint64_t *mutable_storage) {
  Backend *b=cast(handle);
  if(b==nullptr||shared==nullptr||mutable_storage==nullptr)
    return fail("Metal memory accounting pointer is null");
  *shared=shared_bytes(b);*mutable_storage=mutable_bytes(b);
  last_error.clear();return 0;
}

extern "C" void df_metal_destroy(df_metal_handle handle) { delete cast(handle); }
