# Positive-Case Saved-Weight Diagnostic

## Scope

Follow up replica `42053`, held-out start `62053`, the only beneficial comparison
in the completed [54-episode cohort](2026-09-12-controlled-cpu-training.md).
Selection occurred after inspecting that cohort. These seven evaluations reuse
existing saved states and the same start: they are a retrospective mechanism
and reproducibility diagnostic, with no new training or independent replica.

The [exact invocation](2026-09-12-positive-case-invocation.md) was saved before
execution. Runtime source remains `2dd86b5a63d0a3ca88595ae1a1ecad4ad261c5fc`,
using the CPU reference and `adaptive-centered-v6`. Source, graph, configuration
and backend artifact guards matched the parent. All 166,700 MaleCNS v1.0 neurons,
25,582,938 released connections and 4,184 plastic slots remain retained.

## Conditions and Results

Each episode starts with fresh neural, sensory and controller state and two
seconds of black warmup. Evaluation freezes efficacy state and imposes zero
reinforcement. RGB uses the existing sensory mapping; neural propagation and
fixed DNp20/DNpe017 decoding select controls. Telemetry remains observer-only.

| Condition | Game tics | Survival, seconds |
|---|---:|---:|
| Saved plastic state, replay 1 | 320 | 9.143 |
| Saved plastic state, replay 2 | 320 | 9.143 |
| Saved frozen control | 128 | 3.657 |
| Saved timing-shuffled control | 128 | 3.657 |
| Saved plastic state, black input | 192 | 5.486 |
| Saved plastic state, five-second passive interval | 128 | 3.657 |
| Memory erased to original baseline | 128 | 3.657 |

All seven episodes ended in death, with no censoring. Both plastic replays
reproduced the parent's entire recorded trace. Frozen and shuffled replays
matched their original full traces, and erasure matched the original frozen
trace. All eight technical checks passed, including clock alignment, one
observation per game tic, frozen efficacy identity and zero imposed stimulus.

Black input reduced survival while leaving it above the RGB frozen-control
duration. This provides no complete visual-dependence demonstration: a matched
black-input frozen control was not included here. The five-second interval
uses the existing black-input dynamics with learning disabled and passive
efficacy dynamics enabled, before the usual episode reset. It changes efficacy
state without new reinforcement. Survival then returns to the control duration;
1,947 connections still differ from baseline. Retention failure therefore does
not mean all changed weights vanished. Erasure resets both memory state and
efficacies; it does not isolate their individual roles.

## Timing and Evidence

Episodes including warmups processed 52.400 brain seconds in 153.059 recorded
awake seconds (0.342x). Native propagation took 141.539 seconds (92.5%).
The diagnostic timer recorded 160.005 elapsed seconds after parent-artifact
preflight, including provenance/construction and the five-second passive
interval. The latter took 6.062 seconds on both clocks. These are serial CPU
measurements, separately from the shared-graph propagation benchmark.

The 27 untouched JSON evidence files are under
`outputs/doom-learning/cpu-positive-case-diagnostic-20260912/` (1,683,327 bytes).
Remote and local sorted manifests matched SHA-256
`6bdff5cc8eb5acefcb5d32aeaab3537aa96f590463083bf5338756c4310a2b2f`.
Manifest entries contain relative `path`, `size_bytes` and `sha256`; the list
uses sorted JSON keys and compact separators. Source snapshots, graph datasets,
binaries and generated game assets are excluded. Job-scoped sleep assertions
released on normal process exit; permanent power settings remain unchanged.

This confirms reproducible efficacy-state-dependent behavior in one selected
case and exposes failed short-term retention. It does not establish general
survival learning or repair the failed physiological/conditioning gates.
No public model, stream or launch status changed.
