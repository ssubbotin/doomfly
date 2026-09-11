# Metal Two-Dispatch Tick Pipeline Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the Metal tick pipeline from five dispatches to two while retaining the exact Metal event, count, state, and weight results.

**Architecture:** Combine integration with marking because integration writes the delayed future ring slot while marking reads the current slot. Combine target gathering, exact bitmap clearing, and tick finalization after the required dispatch boundary. Each target atomically claims only its incoming-position bits, preserving ascending incoming-edge accumulation and allowing words shared by adjacent targets.

**Tech Stack:** Python 3.11, NumPy, pytest, C++17, Objective-C++, Metal 2.4, `ctypes`, Apple M4 Pro.

**Evidence:** `outputs/doom-learning/metal-resident-dispatch-lower-bound-m4pro.json`

## Global Constraints

- Retain all 166,700 neurons and all 25,582,938 MaleCNS v1.0 connections.
- Preserve every connection, duplicate, self edge, weak edge, and modulatory edge.
- Preserve ascending incoming-edge summation within each target.
- Keep live RGB input, propagation, reinforcement, plasticity, and fixed decoding separate.
- Keep one dispatch boundary between marking and gathering.
- Use only conservative `macos-metal2.4` features supported by Apple GPU family 7.
- Preserve the passing 40 ms values and reproduce the retained failing 80 ms Metal values exactly.
- Write each behavioral test first and observe its expected failure.

### Task 1: Lock the Dispatch and Digest Contracts

**Files:**
- Modify: `tests/test_doom_metal_parity.py`
- Modify: `tests/test_doom_metal_state.py`

- [x] Add an expected-failure test requiring 202 dispatches per 10 ms bin and 808 per 40 ms trace.
- [x] Add shared-word cases where one 32-position incoming word spans several targets and contains mixed-sign weights.
- [x] Require bitwise equality with the current Metal result for counts, events, queue state, weights, and all integer state arrays.
- [x] Run the focused tests on M4 Pro and record the expected dispatch-count failure.

### Task 2: Fuse Integration and Marking

**Files:**
- Modify: `doom_learning_v6/metal/kernels.metal`
- Modify: `doom_learning_v6/metal/backend.mm`
- Modify: `tests/test_doom_metal_parity.py`

- [x] Add `df_integrate_mark`, using one thread per neuron.
- [x] Run the existing integration logic, then scan the current delayed-ring slot for the same neuron and mark its released outgoing edges.
- [x] Keep the 18-tick delay invariant explicit. The current slot and future slot must differ.
- [x] Replace the separate integrate and mark encodes. Require 402 dispatches per 10 ms bin at this checkpoint.
- [x] Run micrograph, shared-word, diagnostic-event, checkpoint, and repeated-run tests on M4 Pro.

### Task 3: Fuse Gathering, Clearing, and Finalization

**Files:**
- Modify: `doom_learning_v6/metal/kernels.metal`
- Modify: `doom_learning_v6/metal/backend.mm`
- Modify: `tests/test_doom_metal_parity.py`

- [x] Add `df_gather_finalize`, using one thread per target neuron.
- [x] For every incoming word, build the target's exact bit mask and use atomic `fetch_and` to claim and clear only that target's positions.
- [x] Accumulate claimed positions in ascending incoming order with the existing mixed-sign arithmetic.
- [x] After gathering, clear the current delayed-ring word, apply future-slot reset, and retain active-state semantics.
- [x] Remove the indirect clear dispatch and obsolete active-word queue from the hot path.
- [x] Require 202 dispatches per 10 ms bin and 808 per 40 ms trace.

### Task 4: Validate Exact Behavior

**Files:**
- Modify: `tests/test_doom_metal_checkpoint.py`
- Modify: `tests/test_doom_metal_validation.py`
- Create: fresh reports under `outputs/doom-learning/`

- [x] Run the complete local and M4 Pro test suites.
- [x] Run a fresh source-hashed 40 ms validation. Require state `59a2739ed22f68ba13811fe4d241e8a2a0a7e8f88a90ef94576385e69d0272e6`, counts `89ec6db8cc45677533caf82db7f288bb68285525adf799f7e2204339a1363605`, 21,326 events, and every retained bin digest.
- [x] Rerun the unchanged 80 ms control. Require Metal state `8fec98ef0d7636b7fd04e7a80a0349c658ef4b282adbbee2fd93452a6334a80e`, counts `f9c25bbd7ca544b10d08461cb91d5542d838ed8eb7ac0624034d1df961a2ca72`, 52,236 events, and every retained Metal bin digest.
- [x] Preserve any failed implementation and report before changing direction.

### Task 5: Measure and Decide

**Files:**
- Modify: `docs/doom-metal-backend.md`
- Modify: `outputs/doom-learning/metal-optimization-study-m4pro.json`
- Create: fresh benchmark report under `outputs/doom-learning/`

- [x] Run five contiguous Metal repetitions using the passing validation report.
- [x] Require 808 dispatches, median GPU time at most 56.90 ms per 40 ms, no hot-path full-state transfers, and all resident-state gates.
- [x] Keep the 2x CPU training gate closed unless Metal neural median reaches half the retained CPU reference, at most 17.27 ms.
- [x] Record raw samples, source and binary hashes, memory pressure, swap, and system load.
- [x] Update documentation and commit the implementation with its evidence.

## Completion Record

Implemented in `c847791a75dd8d6a284b955d5c5ecd8c76379e3f`. Local tests passed with 115 successes and 35 platform skips. M4 Pro tests passed with 147 successes and 3 skips. The exact 40 ms and retained 80 ms Metal digests were reproduced. Five contiguous samples recorded 808 dispatches, zero indirect dispatches, median GPU time `0.03010 s`, median Metal neural time `0.03328 s`, and `1.2019x` real-time throughput. The 2x CPU training gate remains closed.
