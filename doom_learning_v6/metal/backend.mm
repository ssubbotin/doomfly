#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "api.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <initializer_list>
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
  int64_t cursor=0;
  float dt=0.1f;
  float adaptation_jump=8.0f;
  float adaptation_tau=200.0f;
  __strong id<MTLBuffer> out_ptr,out_post,in_ptr,in_pre,in_edge,kc_mask,modulation_mask;
  __strong id<MTLBuffer> weight,v,g,refractory,drive,previous_drive,counts,active_flag,last;
  __strong id<MTLBuffer> modulation,modulation_last,rest,adaptation,ring,touched;
  __strong id<MTLBuffer> kc_events,event_count;
  __strong id<MTLComputePipelineState> drive_pipeline,integrate_pipeline,mark_pipeline;
  __strong id<MTLComputePipelineState> gather_pipeline,clear_pipeline,reset_pipeline,materialize_pipeline;
  uint32_t event_capacity=0;
  uint32_t last_event_count=0;
  bool capture_all_spikes=false;
  bool poisoned=false;
};

struct KernelParams {
  uint32_t neurons;
  uint32_t words;
  uint32_t slot;
  uint32_t future;
  int64_t clock;
  float dt;
  float adaptation_jump;
  float adaptation_tau;
  int32_t refractory_ticks;
  uint32_t event_capacity;
  uint32_t capture_all_spikes;
};

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

void encode(id<MTLCommandBuffer> command,id<MTLComputePipelineState> pipeline,
    std::initializer_list<id<MTLBuffer>> buffers,const KernelParams &params,NSUInteger threads) {
  id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];NSUInteger index=0;
  for(id<MTLBuffer> buffer:buffers)[encoder setBuffer:buffer offset:0 atIndex:index++];
  [encoder setBytes:&params length:sizeof(params) atIndex:index];
  NSUInteger width=std::min<NSUInteger>(256,pipeline.maxTotalThreadsPerThreadgroup);
  [encoder dispatchThreads:MTLSizeMake(threads,1,1) threadsPerThreadgroup:MTLSizeMake(width,1,1)];
  [encoder endEncoding];
}

bool state_pointers_present(const df_metal_state *s) {
  return s!=nullptr&&present(s->weight)&&present(s->v)&&present(s->g)&&
    present(s->refractory)&&present(s->drive)&&present(s->previous_drive)&&
    present(s->queue)&&present(s->queue_count)&&present(s->counts)&&
    present(s->active)&&present(s->active_flag)&&present(s->nactive)&&
    present(s->last)&&present(s->modulation)&&present(s->modulation_last)&&
    present(s->rest)&&present(s->adaptation);
}

Backend *cast(df_metal_handle handle) { return static_cast<Backend *>(handle); }

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
  if(graph==nullptr||handle==nullptr||metallib_path==nullptr)
    return fail("Metal create argument is null");
  *handle=nullptr;
  if(graph->neurons<1||graph->edges<0||graph->delay_slots<1||graph->dt_ms!=0.1f)
    return fail("Invalid Metal graph dimensions");
  if(!present(graph->out_ptr)||!present(graph->out_post)||!present(graph->in_ptr)||
      !present(graph->in_pre)||!present(graph->in_edge)||!present(graph->kc_mask)||
      !present(graph->modulation_mask))return fail("Metal graph pointer is null");
  if(graph->out_ptr[0]!=0||graph->out_ptr[graph->neurons]!=graph->edges||
      graph->in_ptr[0]!=0||graph->in_ptr[graph->neurons]!=graph->edges)
    return fail("Metal graph CSR bounds mismatch");
  @autoreleasepool {
    auto *backend=new Backend();
    backend->device=MTLCreateSystemDefaultDevice();
    df_metal_device_info info;
    if(!set_device_info(backend->device,&info)){
      delete backend;return fail("Metal requires unified memory and Apple GPU family 7");
    }
    backend->command_queue=[backend->device newCommandQueue];
    NSError *error=nil;
    NSString *path=[NSString stringWithUTF8String:metallib_path];
    backend->library=[backend->device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&error];
    if(backend->command_queue==nil||backend->library==nil){
      std::string message=error==nil?"Metal command queue or library creation failed":error.localizedDescription.UTF8String;
      delete backend;return fail(message.c_str());
    }
    backend->drive_pipeline=make_pipeline(backend,@"df_drive_change",&error);
    backend->integrate_pipeline=make_pipeline(backend,@"df_integrate",&error);
    backend->mark_pipeline=make_pipeline(backend,@"df_mark_targets",&error);
    backend->gather_pipeline=make_pipeline(backend,@"df_gather_targets",&error);
    backend->clear_pipeline=make_pipeline(backend,@"df_clear_slot",&error);
    backend->reset_pipeline=make_pipeline(backend,@"df_reset_future",&error);
    backend->materialize_pipeline=make_pipeline(backend,@"df_materialize",&error);
    if(backend->drive_pipeline==nil||backend->integrate_pipeline==nil||
        backend->mark_pipeline==nil||backend->gather_pipeline==nil||
        backend->clear_pipeline==nil||backend->reset_pipeline==nil||
        backend->materialize_pipeline==nil){
      std::string message=error==nil?"Metal pipeline creation failed":error.localizedDescription.UTF8String;
      delete backend;return fail(message.c_str());
    }
    backend->neurons=graph->neurons;backend->edges=graph->edges;
    backend->slots=graph->delay_slots;backend->words=(graph->neurons+31)/32;
    backend->dt=graph->dt_ms;backend->adaptation_jump=graph->adaptation_jump_mv;
    backend->adaptation_tau=graph->adaptation_tau_ms;
    const size_t n=graph->neurons,e=graph->edges;
    backend->out_ptr=make_buffer(backend->device,graph->out_ptr,(n+1)*sizeof(int64_t));
    backend->out_post=make_buffer(backend->device,graph->out_post,e*sizeof(int32_t));
    backend->in_ptr=make_buffer(backend->device,graph->in_ptr,(n+1)*sizeof(int64_t));
    backend->in_pre=make_buffer(backend->device,graph->in_pre,e*sizeof(int32_t));
    backend->in_edge=make_buffer(backend->device,graph->in_edge,e*sizeof(int32_t));
    backend->kc_mask=make_buffer(backend->device,graph->kc_mask,n);
    backend->modulation_mask=make_buffer(backend->device,graph->modulation_mask,n);
    backend->weight=make_buffer(backend->device,nullptr,e*sizeof(float));
    backend->v=make_buffer(backend->device,nullptr,n*sizeof(float));
    backend->g=make_buffer(backend->device,nullptr,n*sizeof(float));
    backend->refractory=make_buffer(backend->device,nullptr,n*sizeof(int16_t));
    backend->drive=make_buffer(backend->device,nullptr,n*sizeof(float));
    backend->previous_drive=make_buffer(backend->device,nullptr,n*sizeof(float));
    backend->counts=make_buffer(backend->device,nullptr,n*sizeof(int32_t));
    backend->active_flag=make_buffer(backend->device,nullptr,n);
    backend->last=make_buffer(backend->device,nullptr,n*sizeof(int64_t));
    backend->modulation=make_buffer(backend->device,nullptr,n*sizeof(float));
    backend->modulation_last=make_buffer(backend->device,nullptr,n*sizeof(int64_t));
    backend->rest=make_buffer(backend->device,nullptr,n*sizeof(float));
    backend->adaptation=make_buffer(backend->device,nullptr,n*sizeof(float));
    backend->ring=make_buffer(backend->device,nullptr,backend->slots*backend->words*sizeof(uint32_t));
    backend->touched=make_buffer(backend->device,nullptr,n*sizeof(uint32_t));
    uint32_t kc_count=0;
    for(size_t i=0;i<n;i++)if(graph->kc_mask[i])kc_count++;
    backend->event_capacity=std::max<uint32_t>(1,kc_count*5u);
    backend->kc_events=make_buffer(backend->device,nullptr,
      size_t(backend->event_capacity)*sizeof(df_metal_kc_event));
    backend->event_count=make_buffer(backend->device,nullptr,sizeof(uint32_t));
    const id<MTLBuffer> required[]={backend->out_ptr,backend->out_post,backend->in_ptr,
      backend->in_pre,backend->in_edge,backend->kc_mask,backend->modulation_mask,
      backend->weight,backend->v,backend->g,backend->refractory,backend->drive,
      backend->previous_drive,backend->counts,backend->active_flag,backend->last,
      backend->modulation,backend->modulation_last,backend->rest,backend->adaptation,
      backend->ring,backend->touched,backend->kc_events,backend->event_count};
    for(id<MTLBuffer> buffer:required)if(buffer==nil){
      delete backend;return fail("Metal shared buffer allocation failed");
    }
    std::memset(backend->ring.contents,0,backend->slots*backend->words*sizeof(uint32_t));
    std::memset(backend->touched.contents,0,n*sizeof(uint32_t));
    *handle=backend;
  }
  last_error.clear();return 0;
}

extern "C" int df_metal_upload_state(df_metal_handle handle,const df_metal_state *s) {
  Backend *b=cast(handle);
  if(b==nullptr||!state_pointers_present(s))return fail("Metal state pointer is null");
  const int32_t n=b->neurons,slots=b->slots,active_count=*s->nactive;
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
  std::memcpy(b->weight.contents,s->weight,size_t(b->edges)*sizeof(float));
  std::memcpy(b->v.contents,s->v,nf);std::memcpy(b->g.contents,s->g,nf);
  std::memcpy(b->refractory.contents,s->refractory,size_t(n)*sizeof(int16_t));
  std::memcpy(b->drive.contents,s->drive,nf);std::memcpy(b->previous_drive.contents,s->previous_drive,nf);
  std::memcpy(b->counts.contents,s->counts,ni);std::memcpy(b->active_flag.contents,s->active_flag,n);
  std::memcpy(b->last.contents,s->last,nl);std::memcpy(b->modulation.contents,s->modulation,nf);
  std::memcpy(b->modulation_last.contents,s->modulation_last,nl);std::memcpy(b->rest.contents,s->rest,nf);
  std::memcpy(b->adaptation.contents,s->adaptation,nf);
  std::memcpy(b->ring.contents,ring.data(),ring.size()*sizeof(uint32_t));
  std::memset(b->touched.contents,0,size_t(n)*sizeof(uint32_t));b->cursor=s->cursor;
  last_error.clear();return 0;
}

extern "C" int df_metal_download_state(df_metal_handle handle,df_metal_state *s) {
  Backend *b=cast(handle);
  if(b==nullptr||!state_pointers_present(s))return fail("Metal state pointer is null");
  const int32_t n=b->neurons,slots=b->slots;
  const size_t nf=size_t(n)*sizeof(float),ni=size_t(n)*sizeof(int32_t),nl=size_t(n)*sizeof(int64_t);
  std::memcpy(s->weight,b->weight.contents,size_t(b->edges)*sizeof(float));
  std::memcpy(s->v,b->v.contents,nf);std::memcpy(s->g,b->g.contents,nf);
  std::memcpy(s->refractory,b->refractory.contents,size_t(n)*sizeof(int16_t));
  std::memcpy(s->drive,b->drive.contents,nf);std::memcpy(s->previous_drive,b->previous_drive.contents,nf);
  std::memcpy(s->counts,b->counts.contents,ni);std::memcpy(s->active_flag,b->active_flag.contents,n);
  std::memcpy(s->last,b->last.contents,nl);std::memcpy(s->modulation,b->modulation.contents,nf);
  std::memcpy(s->modulation_last,b->modulation_last.contents,nl);std::memcpy(s->rest,b->rest.contents,nf);
  std::memcpy(s->adaptation,b->adaptation.contents,nf);
  const uint32_t *ring=static_cast<const uint32_t *>(b->ring.contents);
  std::memset(s->queue,0,size_t(slots)*n*sizeof(int32_t));
  for(int32_t slot=0;slot<slots;slot++){
    int32_t count=0;
    for(int32_t id=0;id<n;id++)if(ring[size_t(slot)*b->words+(id>>5)]&(uint32_t(1)<<(id&31)))
      s->queue[size_t(slot)*n+count++]=id;
    s->queue_count[slot]=count;
  }
  std::memset(s->active,0,ni);int32_t count=0;
  const uint8_t *flags=static_cast<const uint8_t *>(b->active_flag.contents);
  for(int32_t id=0;id<n;id++)if(flags[id])s->active[count++]=id;
  *s->nactive=count;s->cursor=b->cursor;
  last_error.clear();return 0;
}

extern "C" int df_metal_update_weights(df_metal_handle handle,int32_t count,
    const int64_t *edge_ids,const float *values) {
  Backend *b=cast(handle);
  if(b==nullptr||count<0||(count>0&&(!present(edge_ids)||!present(values))))
    return fail("Invalid plastic weight update");
  float *weight=static_cast<float *>(b->weight.contents);
  for(int32_t i=0;i<count;i++){
    if(edge_ids[i]<0||edge_ids[i]>=b->edges)return fail("Plastic edge out of bounds");
    weight[edge_ids[i]]=values[i];
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
  const uint32_t kernel_capacity=std::min<uint32_t>(b->event_capacity,event_capacity);
  *static_cast<uint32_t *>(b->event_count.contents)=0;
  const auto started=std::chrono::steady_clock::now();
  @autoreleasepool {
    id<MTLCommandBuffer> command=[b->command_queue commandBuffer];
    if(command==nil){b->poisoned=true;return fail("Metal command buffer creation failed");}
    KernelParams p{uint32_t(b->neurons),uint32_t(b->words),0,0,b->cursor,b->dt,
      b->adaptation_jump,b->adaptation_tau,int32_t(std::lround(2.2f/b->dt)),kernel_capacity,
      b->capture_all_spikes?1u:0u};
    encode(command,b->drive_pipeline,{b->v,b->g,b->refractory,b->drive,b->previous_drive,
      b->active_flag,b->last,b->rest,b->adaptation},p,b->neurons);
    const int32_t delay=std::lround(1.8f/b->dt);
    for(int32_t step=0;step<steps;step++){
      p.clock=b->cursor+step;p.slot=uint32_t(p.clock%b->slots);
      p.future=uint32_t((p.clock+delay)%b->slots);
      encode(command,b->integrate_pipeline,{b->v,b->g,b->refractory,b->drive,b->active_flag,
        b->last,b->rest,b->adaptation,b->ring,b->counts,b->kc_mask,b->kc_events,b->event_count},p,b->neurons);
      encode(command,b->mark_pipeline,{b->out_ptr,b->out_post,b->ring,b->touched},p,b->neurons);
      encode(command,b->gather_pipeline,{b->in_ptr,b->in_pre,b->in_edge,b->weight,b->ring,
        b->touched,b->v,b->g,b->refractory,b->drive,b->active_flag,b->last,b->modulation,
        b->modulation_last,b->modulation_mask,b->rest,b->adaptation},p,b->neurons);
      encode(command,b->clear_pipeline,{b->ring},p,b->words);
      encode(command,b->reset_pipeline,{b->ring,b->v,b->g,b->refractory,b->rest},p,b->neurons);
    }
    p.clock=b->cursor+steps-1;
    encode(command,b->materialize_pipeline,{b->v,b->g,b->refractory,b->drive,b->last,
      b->rest,b->adaptation},p,b->neurons);
    [command commit];[command waitUntilCompleted];
    if(command.status!=MTLCommandBufferStatusCompleted){
      b->poisoned=true;
      if(command.error==nil)return fail("Metal command failed");
      return fail(command.error.localizedDescription);
    }
    const uint32_t produced=*static_cast<uint32_t *>(b->event_count.contents);
    if(produced>kernel_capacity){b->poisoned=true;return fail("KC event capacity exceeded");}
    if(produced>0)std::memcpy(events,b->kc_events.contents,size_t(produced)*sizeof(df_metal_kc_event));
    *event_count=int32_t(produced);b->last_event_count=produced;b->cursor+=steps;
    timing->gpu_seconds=command.GPUEndTime>=command.GPUStartTime?
      command.GPUEndTime-command.GPUStartTime:0.0;
  }
  timing->host_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
  last_error.clear();return 0;
}

extern "C" int df_metal_apply_eligibility(df_metal_handle handle,double *eligibility,
    int64_t *eligibility_last,double tau_ms) {
  Backend *b=cast(handle);
  if(b==nullptr||eligibility==nullptr||eligibility_last==nullptr||
      !std::isfinite(tau_ms)||tau_ms<=0)return fail("Invalid eligibility update");
  const auto *stored=static_cast<const df_metal_kc_event *>(b->kc_events.contents);
  std::vector<df_metal_kc_event> events(stored,stored+b->last_event_count);
  std::sort(events.begin(),events.end(),[](const auto &left,const auto &right){
    return left.tick<right.tick||(left.tick==right.tick&&left.neuron<right.neuron);
  });
  for(const auto &event:events){
    if(event.neuron<0||event.neuron>=b->neurons)return fail("KC event neuron out of bounds");
    const auto *kc_mask=static_cast<const uint8_t *>(b->kc_mask.contents);
    if(!kc_mask[event.neuron])continue;
    const int64_t delta=event.tick-eligibility_last[event.neuron];
    const float decay=std::exp(-b->dt*float(delta)/float(tau_ms));
    eligibility[event.neuron]*=decay;
    eligibility[event.neuron]+=1.0;eligibility_last[event.neuron]=event.tick;
  }
  last_error.clear();return 0;
}

extern "C" int df_metal_set_diagnostics(df_metal_handle handle,int32_t enabled) {
  Backend *b=cast(handle);
  if(b==nullptr||(enabled!=0&&enabled!=1))return fail("Invalid Metal diagnostic mode");
  if(enabled&&b->event_capacity<uint32_t(b->neurons)*5u){
    const uint32_t capacity=uint32_t(b->neurons)*5u;
    id<MTLBuffer> events=make_buffer(b->device,nullptr,size_t(capacity)*sizeof(df_metal_kc_event));
    if(events==nil)return fail("Metal diagnostic event allocation failed");
    b->kc_events=events;b->event_capacity=capacity;
  }
  b->capture_all_spikes=enabled!=0;last_error.clear();return 0;
}

extern "C" void df_metal_destroy(df_metal_handle handle) { delete cast(handle); }
