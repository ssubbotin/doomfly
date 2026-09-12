# Metal Host Decay Tables Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether resident canonical CPU decay coefficients improve Metal numerical parity without changing the neural model.

**Architecture:** Objective-C++ creates one 12 KiB packed host-generated float table at backend creation. All four Metal evolution kernels receive it; short intervals use identical coefficients and long intervals retain the existing fallback. Advance the kernel ABI epoch and invalidate build/validation identities.

**Tech Stack:** C++17, Objective-C++, Metal 2.4, ctypes, NumPy and pytest.

**Spec:** `docs/superpowers/specs/2026-09-13-metal-decay-tables-design.md`

## Global Constraints

- Minimum macOS 13.0, arm64, unified memory and Apple GPU family 7.
- Metal language `macos-metal2.4`; `-fno-fast-math` and `-ffp-contract=off`.
- Retain all 166,700 neurons and 25,582,938 connections in full-graph runs.
- Preserve the canonical CPU kernel, dt 0.1 ms, every neural tick, plasticity,
  sensory mapping, reinforcement and fixed neuron-to-button decoding.
- Preserve parity thresholds: Jaccard >= 0.995, spike fraction <= 0.005,
  rate correlation >= 0.999, within-one-tick fraction >= 0.999, weight rtol 1e-4,
  exact decoder/scientific decisions, structural integrity and repeated state.
- Preserve prior failed controls and strict-arithmetic state epochs in history
  and evidence; numerical tests do not establish learning or biological validity.
- Keep machine paths and origins in ignored configuration. No public deployment
  changes or launch green light are part of this experiment.

## File responsibilities

The implementation worker owns the portable table header, native/kernel wiring,
build and validation identity changes and their tests. The controller owns
target-machine execution, experiment invocations, portable evidence and public
experimental documentation. Neither modifies the running cohort's checkout.

### Task 1: Resident tables and checked identity

**Files:**
- Create: `doom_learning_v6/metal/decay_tables.h`
- Modify: `doom_learning_v6/metal/api.h`, `backend.mm`, `kernels.metal`, `build.py`, `validate.py`
- Create: `tests/test_doom_metal_decay.py`
- Modify: `tests/test_doom_metal_build.py`, `test_doom_metal_parity.py`, `test_doom_metal_validation.py` only where needed for the new identity/arithmetic epoch.

**Interfaces:**
- Consumes: existing `df_metal_graph.dt_ms` and `.adaptation_tau_ms`; no C structure changes.
- Produces: `doomfly::metal::make_decay_tables(float dt, float adaptation_tau) -> std::array<float, 3072>`; three contiguous 1,024-float regions in voltage/conductance/adaptation order.
- Produces: ABI epoch 4 and a resident table buffer passed immediately before `Params` to `df_drive_change`, `df_integrate_mark`, `df_gather_finalize`, `df_materialize`.
- Preserves: long-interval/modulation fallback and all other arithmetic, dispatch and transfer contracts.

- [ ] **Step 1: Write genuine native failing coefficient and cache tests.**

Compile a temporary C++17 probe using the production header. Compare all entries
bitwise against independent canonical expressions at dt 0.1 and adaptation tau
200 and 137. Reject a table missing from build or validation source identity.
For cache invalidation, copy all build sources into a temporary directory, build
with the existing fake Apple toolchain, alter only the header and require another
three compile/link calls. Do not mutate real repository sources in a test.

```cpp
auto table=doomfly::metal::make_decay_tables(dt, tau);
for(int i=0;i<1024;i++) {
  float expected[3]={std::exp(-dt*i/20.f),std::exp(-dt*i/5.f),std::exp(-dt*i/tau)};
  for(int region=0;region<3;region++)
    if(std::memcmp(&table[region*1024+i],&expected[region],sizeof(float)))return 1;
}
```

- [ ] **Step 2: Run focused tests and retain expected RED output.**

```sh
python -m pytest -q tests/test_doom_metal_decay.py tests/test_doom_metal_build.py
```

Expected missing-header/compiler failure for the new production interface and
missing-header identity/cache assertions. Distinguish environment failures from
behavioral RED evidence.

- [ ] **Step 3: Generate once and bind resident tables.**

```cpp
inline std::array<float,3072> make_decay_tables(float dt,float adaptation_tau) {
  std::array<float,3072> result;
  for(int i=0;i<1024;i++) {
    result[i]=std::exp(-dt*i/20.f);
    result[1024+i]=std::exp(-dt*i/5.f);
    result[2048+i]=std::exp(-dt*i/adaptation_tau);
  }
  return result;
}
```

Add its immutable buffer to backend ownership and required allocation checks.
Copy the generated array once. Append the buffer before `Params` in all four
Objective-C++ encoder calls and shift the corresponding Metal bindings.

```cpp
float a=d<1024?decay[d]:exp(-dt*d/20.0f);
float b=d<1024?decay[1024+d]:exp(-dt*d/5.0f);
float c=d<1024?decay[2048+d]:exp(-dt*d/adaptation_tau);
```

Apply the same adaptation lookup to refractory `skip`; retain every surrounding
expression. Set both ABI constants to 4. Include the header in build source
hashes and `source_identity()`. Preserve graph ctypes layout, compile flags and
no-upload hot-path contracts.

- [ ] **Step 4: Add arithmetic fixtures and verify GREEN.**

Use paired actual CPU/Metal zero-edge fixture brains with explicit empty circuit
and modulation masks. Seed host states before initial upload, evolve and
materialize; test intervals `[1,2,18,22,100,286,1023,1024]` for no-spike voltage,
conductance, adaptation and refractory states. At 1,024 assert finite state and
unchanged fallback formula, without claiming bitwise CPU agreement. Use exact
coefficient comparisons where the state expression isolates a coefficient;
retain established tolerances for composite arithmetic. Mark Apple-only tests
with the repository's normal platform guards. Run the full Linux suite once.

```sh
python -m pytest -q tests/test_doom_metal_decay.py tests/test_doom_metal_build.py tests/test_doom_metal_validation.py
python -m pytest -q
git diff --check
```

Do not guess a new micrograph golden digest or remove its assertion. Report a
hardware-pending epoch and await the controller's two independent measurements.

- [ ] **Step 5: Commit reviewed implementation with TDD report.**

```sh
git add doom_learning_v6/metal/decay_tables.h doom_learning_v6/metal/api.h doom_learning_v6/metal/backend.mm doom_learning_v6/metal/kernels.metal doom_learning_v6/metal/build.py doom_learning_v6/metal/validate.py tests/test_doom_metal_decay.py tests/test_doom_metal_build.py
git -c user.name='Sergey Subbotin' -c user.email='ssubbotin@gmail.com' commit -m 'Use resident host decay tables in Metal evolution'
```

Stage any additionally owned validation/parity tests individually. Complete the
SDD report with commands, actual RED/GREEN output, files and unresolved hardware
evidence. Controller dispatches independent task review; original worker handles
any fixes and measured golden-epoch update.

### Task 2: M4 Pro bounded experiment and evidence (controller-owned)

**Files:**
- Create: `docs/experiments/2026-09-13-metal-decay-tables-invocation.md`
- Create: `docs/experiments/2026-09-13-metal-decay-tables.md`
- Create: portable JSON under `outputs/doom-learning/metal-decay-tables-20260913/`
- Modify: `docs/doom-metal-backend.md` only to state measured results and limits.

**Interfaces:**
- Consumes: Task 1 commit, unchanged table/probe API, existing `validate.run(out)` and spike gates.
- Produces: exact-source M4 arithmetic results, two measured micrograph epochs, full-graph 40/80 ms results and explicit remaining limits.

- [ ] **Step 1: Save invocation before running.** Record source/build/model identity, fresh output requirement, fixed thresholds and absence of launch/learning claims. Use ignored configuration for target origins.
- [ ] **Step 2: Wait for the existing cohort to finish naturally.** Confirm `results.json`, no `failure.json`, all eight complete episodes and teacher-free controls. Do not compete with its full-graph workload.
- [ ] **Step 3: Create a separate owned target checkout.** Fetch the exact Task 1 commit and build using the existing target-local builder and bootstrap kernel. Run the focused decay tests and two independent instances of the unchanged micrograph trace. Record actual coefficient/state/count/event hashes.

```sh
python -m doom_learning_v6.metal.build --probe
python -m pytest -q tests/test_doom_metal_decay.py
```

- [ ] **Step 4: Return measured golden evidence to the original worker.** Require exact matching measurements before a scoped test update; preserve prior digest in evidence. Independently re-review that fix range.
- [ ] **Step 5: Run full-graph controls with unchanged gates.** Run `validate.run` first with its four-frame trace, then temporarily choose the retained eight-frame trace within the invocation process, restoring `TRACE` in `finally`.

```python
from doom_learning_v6.metal import validate
forty=validate.run(out/'parity-40ms')
previous=validate.TRACE
validate.TRACE=(('black',False),('blue',False),('green',False),('white',True),
                ('left_blue',False),('right_blue',True),('vertical',False),('horizontal',False))
try:eighty=validate.run(out/'parity-80ms')
finally:validate.TRACE=previous
```

- [ ] **Step 6: Preserve every measured pass/failure and publish evidence.** Compare coefficients and full-graph metrics to the retained strict epoch; distinguish arithmetic improvement from parity, training and biology. Run target suite and `git diff --check`, review the complete component diff on the most capable available reviewer, commit exact allowlisted documents/evidence and push the branch. Continue the next Metal hypothesis if the 80 ms gate remains failed; keep the goal active.
