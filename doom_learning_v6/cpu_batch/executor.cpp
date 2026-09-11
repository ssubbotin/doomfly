#include "api.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <new>
#include <string>
#include <vector>

namespace {

thread_local std::string last_error;

struct Executor {
  df_cpu_batch_graph graph;
  std::vector<df_cpu_batch_lane> lanes;
  int32_t workers;
  std::string error;
};

int fail(Executor *executor, const char *message) {
  if (executor) executor->error = message;
  last_error = message;
  return 1;
}

bool present(const void *pointer) { return pointer != nullptr; }

int validate(const df_cpu_batch_graph *graph, const df_cpu_batch_lane *lanes,
             int32_t lane_count, int32_t workers) {
  if (!graph || !lanes || lane_count < 1) return fail(nullptr, "Missing CPU batch graph or lanes");
  if (workers < 1 || workers > lane_count) return fail(nullptr, "Invalid CPU batch worker count");
  if (graph->neurons < 1 || graph->edges < 0 || graph->delay_slots < 2 ||
      graph->plastic_edges < 0 || graph->plastic_edges > 32767)
    return fail(nullptr, "Invalid CPU batch graph counts");
  if (graph->dt_ms != .1f || graph->delay_slots != std::lround(1.8f / graph->dt_ms) + 1 ||
      !std::isfinite(graph->eligibility_tau_ms) || graph->eligibility_tau_ms <= 0 ||
      !std::isfinite(graph->adaptation_jump_mv) || graph->adaptation_jump_mv < 0 ||
      !std::isfinite(graph->adaptation_tau_ms) || graph->adaptation_tau_ms <= 20)
    return fail(nullptr, "Invalid CPU batch dynamics");
  if (!present(graph->out_ptr) || !present(graph->out_post) ||
      !present(graph->base_weight) || !present(graph->plastic_slot) ||
      !present(graph->kc_mask) || !present(graph->modulation_mask) || !present(graph->rest))
    return fail(nullptr, "Missing CPU batch graph buffer");
  if (graph->out_ptr[0] != 0 || graph->out_ptr[graph->neurons] != graph->edges)
    return fail(nullptr, "Invalid CPU batch CSR boundary");
  for (int32_t i = 0; i < graph->neurons; ++i) {
    if (graph->out_ptr[i] > graph->out_ptr[i + 1] || !std::isfinite(graph->rest[i]))
      return fail(nullptr, "Invalid CPU batch graph array");
  }
  for (int64_t edge = 0; edge < graph->edges; ++edge) {
    const int16_t slot = graph->plastic_slot[edge];
    if (graph->out_post[edge] < 0 || graph->out_post[edge] >= graph->neurons ||
        !std::isfinite(graph->base_weight[edge]) || slot < -1 || slot >= graph->plastic_edges)
      return fail(nullptr, "Invalid CPU batch edge array");
  }
  for (int32_t lane_index = 0; lane_index < lane_count; ++lane_index) {
    const auto &lane = lanes[lane_index];
    if (!present(lane.cursor) || !present(lane.v) || !present(lane.g) ||
        !present(lane.refractory) || !present(lane.drive) ||
        !present(lane.previous_drive) || !present(lane.queue) ||
        !present(lane.queue_count) || !present(lane.counts) ||
        !present(lane.active) || !present(lane.active_flag) ||
        !present(lane.nactive) || !present(lane.last) ||
        !present(lane.eligibility) || !present(lane.eligibility_last) ||
        !present(lane.modulation) || !present(lane.modulation_last) ||
        !present(lane.adaptation) ||
        (graph->plastic_edges && !present(lane.plastic_weight)))
      return fail(nullptr, "Missing CPU batch lane buffer");
    if (*lane.nactive < 0 || *lane.nactive > graph->neurons)
      return fail(nullptr, "Invalid CPU batch active count");
  }
  return 0;
}

inline float edge_weight(const df_cpu_batch_graph &graph,
                         const df_cpu_batch_lane &lane, int64_t edge) {
  const int16_t slot = graph.plastic_slot[edge];
  return slot < 0 ? graph.base_weight[edge] : lane.plastic_weight[slot];
}

void advance_lane(const df_cpu_batch_graph &graph, df_cpu_batch_lane &lane, int steps) {
  const int n = graph.neurons;
  const float dt = graph.dt_ms;
  const int delay = std::lround(1.8f / dt);
  const int rfc = std::lround(2.2f / dt);
  const int slots = delay + 1;
  float av[1024], ag[1024], aa[1024];
  for (int i = 0; i < 1024; ++i) {
    av[i] = std::exp(-dt * i / 20.f);
    ag[i] = std::exp(-dt * i / 5.f);
    aa[i] = std::exp(-dt * i / graph.adaptation_tau_ms);
  }
  auto evolve = [&](int i, int64_t now, float current) {
    int64_t d = now - lane.last[i];
    if (d <= 0) return;
    const int frozen = lane.refractory[i] > 0 ? lane.refractory[i] - 1 : 0;
    const int skip = static_cast<int>(d < frozen ? d : frozen);
    if (skip > 0 && lane.adaptation[i] > 0)
      lane.adaptation[i] *= skip < 1024 ? aa[skip]
          : std::exp(-dt * skip / graph.adaptation_tau_ms);
    lane.refractory[i] = d >= lane.refractory[i] ? 0 : lane.refractory[i] - d;
    d -= skip;
    if (d > 0) {
      const float a = d < 1024 ? av[d] : std::exp(-dt * d / 20.f);
      const float b = d < 1024 ? ag[d] : std::exp(-dt * d / 5.f);
      lane.v[i] = graph.rest[i] + (lane.v[i] - graph.rest[i]) * a +
          current * (1.f - a) + lane.g[i] * (a - b) / 3.f;
      lane.g[i] *= b;
      if (lane.adaptation[i] > 0) {
        const float c = d < 1024 ? aa[d]
            : std::exp(-dt * d / graph.adaptation_tau_ms);
        lane.v[i] -= lane.adaptation[i] * graph.adaptation_tau_ms /
            (graph.adaptation_tau_ms - 20.f) * (c - a);
        lane.adaptation[i] *= c;
      }
    }
    lane.last[i] = now;
  };
  auto awaken = [&](int i) {
    if (!lane.active_flag[i]) {
      lane.active_flag[i] = 1;
      lane.active[(*lane.nactive)++] = i;
    }
  };
  for (int i = 0; i < n; ++i) {
    if (lane.drive[i] != lane.previous_drive[i]) {
      evolve(i, *lane.cursor - 1, lane.previous_drive[i]);
      lane.previous_drive[i] = lane.drive[i];
      awaken(i);
    }
  }
  for (int t = 0; t < steps; ++t, ++(*lane.cursor)) {
    const int slot = *lane.cursor % slots;
    const int future = (*lane.cursor + delay) % slots;
    int kept = 0;
    const int original = *lane.nactive;
    for (int k = 0; k < original; ++k) {
      const int i = lane.active[k];
      evolve(i, *lane.cursor, lane.drive[i]);
      if (lane.refractory[i] == 0 && lane.v[i] > -45.f) {
        lane.queue[future * n + lane.queue_count[future]++] = i;
        lane.counts[i]++;
        if (graph.kc_mask[i]) {
          lane.adaptation[i] += graph.adaptation_jump_mv;
          lane.eligibility[i] *= std::exp(-dt * (*lane.cursor - lane.eligibility_last[i]) /
              graph.eligibility_tau_ms);
          lane.eligibility[i] += 1.;
          lane.eligibility_last[i] = *lane.cursor;
        }
      }
      const float gap = -45.f - graph.rest[i];
      const bool can_fire = lane.v[i] > -45.f || lane.drive[i] > gap ||
          lane.drive[i] + lane.g[i] > gap;
      if (can_fire) lane.active[kept++] = i;
      else lane.active_flag[i] = 0;
    }
    *lane.nactive = kept;
    for (int q = 0; q < lane.queue_count[slot]; ++q) {
      const int i = lane.queue[slot * n + q];
      if (graph.modulation_mask[i]) {
        for (int64_t edge = graph.out_ptr[i]; edge < graph.out_ptr[i + 1]; ++edge) {
          const int j = graph.out_post[edge];
          lane.modulation[j] *= std::exp(-dt * (*lane.cursor - lane.modulation_last[j]) / 100.f);
          lane.modulation[j] += std::abs(edge_weight(graph, lane, edge)) / .275f;
          lane.modulation_last[j] = *lane.cursor;
        }
        continue;
      }
      for (int64_t edge = graph.out_ptr[i]; edge < graph.out_ptr[i + 1]; ++edge) {
        const int j = graph.out_post[edge];
        evolve(j, *lane.cursor, lane.drive[j]);
        if (lane.refractory[j] == 0) {
          lane.g[j] += edge_weight(graph, lane, edge);
          awaken(j);
        }
      }
    }
    lane.queue_count[slot] = 0;
    for (int q = 0; q < lane.queue_count[future]; ++q) {
      const int i = lane.queue[future * n + q];
      lane.v[i] = graph.rest[i];
      lane.g[i] = 0.f;
      lane.refractory[i] = rfc;
    }
  }
  for (int i = 0; i < n; ++i) evolve(i, *lane.cursor - 1, lane.drive[i]);
}

}  // namespace

extern "C" uint32_t df_cpu_batch_abi_version(void) {
  return DF_CPU_BATCH_ABI_VERSION;
}

extern "C" int df_cpu_batch_create(const df_cpu_batch_graph *graph,
    const df_cpu_batch_lane *lanes, int32_t lane_count, int32_t workers,
    df_cpu_batch_handle *handle) {
  if (!handle) return fail(nullptr, "Missing CPU batch output handle");
  *handle = nullptr;
  if (validate(graph, lanes, lane_count, workers)) return 1;
  try {
    auto *executor = new Executor{*graph,
        std::vector<df_cpu_batch_lane>(lanes, lanes + lane_count), workers, {}};
    *handle = executor;
    return 0;
  } catch (const std::exception &exception) {
    return fail(nullptr, exception.what());
  } catch (...) {
    return fail(nullptr, "Unknown CPU batch creation failure");
  }
}

extern "C" int df_cpu_batch_advance(df_cpu_batch_handle handle, int32_t steps,
    df_cpu_batch_timing *timing) {
  auto *executor = static_cast<Executor *>(handle);
  if (!executor) return fail(nullptr, "Missing CPU batch handle");
  if (steps < 1 || steps > 100) return fail(executor, "CPU batch steps must be 1 through 100");
  if (!timing) return fail(executor, "Missing CPU batch timing output");
  const auto started = std::chrono::steady_clock::now();
  try {
    for (auto &lane : executor->lanes) advance_lane(executor->graph, lane, steps);
  } catch (const std::exception &exception) {
    return fail(executor, exception.what());
  } catch (...) {
    return fail(executor, "Unknown CPU batch worker failure");
  }
  const auto completed = std::chrono::steady_clock::now();
  timing->native_wall_seconds = std::chrono::duration<double>(completed - started).count();
  timing->lanes_advanced = static_cast<int32_t>(executor->lanes.size());
  timing->workers = executor->workers;
  timing->steps = steps;
  return 0;
}

extern "C" const char *df_cpu_batch_error(df_cpu_batch_handle handle) {
  auto *executor = static_cast<Executor *>(handle);
  return executor ? executor->error.c_str() : last_error.c_str();
}

extern "C" void df_cpu_batch_destroy(df_cpu_batch_handle handle) {
  delete static_cast<Executor *>(handle);
}
