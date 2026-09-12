# CPU-Compatible Metal Arithmetic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the reproduced voltage arithmetic mismatch without changing the CPU reference or modeled circuitry.

**Architecture:** Explicit Metal `fma` calls reproduce the actual M4 CPU voltage instruction association. Keep strict flags and incoming propagation fixed, advance checked arithmetic identity, then measure short and longer retained full-graph controls.

**Tech Stack:** C++17, Objective-C++, Metal 2.4, ctypes, NumPy and pytest.

**Spec:** `docs/superpowers/specs/2026-09-13-metal-arithmetic-design.md`

## Global Constraints

- Minimum macOS 13.0, arm64, unified memory and Apple GPU family 7.
- Metal language `macos-metal2.4`; `-fno-fast-math` and `-ffp-contract=off`.
- Preserve canonical CPU source/flags, dt 0.1 ms, neural ticks, calibration,
  every retained MaleCNS v1.0 connection and 4,184 identified plastic slots.
- Preserve sensory mapping, reinforcement, host plasticity and fixed decoding;
  targets/observer telemetry never choose actions.
- Preserve parity thresholds: Jaccard >= 0.995, spike fraction <= 0.005,
  rate correlation >= 0.999, within-one-tick fraction >= 0.999, weight rtol 1e-4,
  exact decoder/scientific decisions, structural integrity and repeated state.
- Preserve all previous failed controls and measured epochs. Numerical tests,
  changed weights and survival alone establish neither useful nor fly learning.
- No public deployment, launch green light, unrelated workload interruption,
  credentials/origins in tracked files or bundled game/research assets.

### Task 1: Voltage arithmetic and checked epoch

**Files:**
- Create: `tests/test_doom_metal_arithmetic.py`
- Modify: `doom_learning_v6/metal/kernels.metal`, `api.h`, `build.py`
- Modify if required by the epoch: `tests/test_doom_metal_build.py`, `tests/test_doom_metal_validation.py`
- Modify `tests/test_doom_metal_parity.py` only after two actual matching measurements.

**Interfaces:**
- Consumes: existing `evolve`, packed resident coefficients and unchanged Metal C structs.
- Produces: explicit voltage FMA operations and ABI epoch 5, no new dispatch/transfer contracts.
- Hardware RED/GREEN and micrograph measurements come from the controller.

- [ ] **Step 1: Write real paired arithmetic regressions first.**

Reuse `test_doom_metal_decay._edgeless_pair` (four neurons) in a macOS-only test.
Check one and 200 sequential 100-tick bins at tau 200 and 137; seed before device
initialization, reset counts per bin and materialize before comparisons.
Include a simple driven relaxation test that specifically reproduces the first
dark-bin mismatch, separately from the composite adaptation case. Use actual
native CPU as the reference. Always close both backends in `finally`.

Hardware-derived amendment: keep all original driven comparisons strict. Add
43-bin strict composite cases and continue the 200-bin composite comparisons
under the spec's explicitly scoped conductance-underflow rule. Every other
field stays bit-exact throughout 200 bins. Add portable negative checks proving
the exception rejects normal CPU values and nonzero Metal mismatches. The
original trial failures remain evidence; avoid broad tolerances or xfail masks.
This amendment supersedes only the snippet's unconditional 200-bin composite
`g` comparison. Full-graph gates and production arithmetic remain unchanged.

```python
cpu, metal = _edgeless_pair(tmp_path, tau)
for brain in [cpu, metal]:
    brain.drive[:] = [12., 9.87, 11.3125, 3.75]
    # Simple regression: default rest, zero g/adaptation.
    # Separate composite regression:
    # brain.rest[2:] = [-60., -60.]
    # brain.v[:] = [-51., -52., -59., -58.]
    # brain.g[:] = [0., .5, 1., 1.25]
    # brain.adaptation[:] = [0., .75, 4., .7]
for _ in range(bins):
    for brain in [cpu, metal]:
        brain.counts.fill(0)
        brain.backend.advance(100)
        brain.backend.materialize('arithmetic-regression')
    assert cpu.cursor == metal.cursor
    np.testing.assert_array_equal(cpu.counts, metal.counts)
    for name in ['v', 'g', 'adaptation']:
        np.testing.assert_array_equal(getattr(cpu, name).view(np.uint32),
                                      getattr(metal, name).view(np.uint32))
    np.testing.assert_array_equal(cpu.refractory, metal.refractory)
    np.testing.assert_array_equal(cpu.last, metal.last)
```

- [ ] **Step 2: Obtain genuine hardware RED before implementation.**

Commit the test-only checkpoint, report its hash/path to the controller and
pause implementation until the actual M4 focused run returns. The controller
pushes that checkpoint to the isolated target and runs:

```sh
OPENBLAS_NUM_THREADS=1 python -m pytest -q tests/test_doom_metal_arithmetic.py
```

Expected uint32 voltage mismatch in the driven fixture. A compiler/environment
error does not count as RED. Keep all output; resolve any unexpected composite
arithmetic mechanism separately, with the CPU and strict flags untouched.

- [ ] **Step 3: Apply only the explicit voltage operations and epoch.**

```cpp
float voltage = fma(v[i] - rest[i], a, rest[i]);
voltage = fma(current, 1.0f - a, voltage);
v[i] = voltage + g[i] * (a - b) / 3.0f;
// Existing adaptation branch, after choosing c:
float correction = (-adaptation[i] * adaptation_tau) / (adaptation_tau - 20.0f);
v[i] = fma(correction, c - a, v[i]);
```

Change `DF_METAL_ABI_VERSION` and builder `ABI_VERSION` to 5. Retain all flags,
table bindings, coefficient/fallback updates, existing input gather, plasticity
and state transfers. Source identity already includes the changed kernel/API.
Adjust only epoch-dependent tests; preserve meaningful stale-report/cache tests.

- [ ] **Step 4: Verify and hand off actual hardware GREEN.**

```sh
python -m pytest -q tests/test_doom_metal_arithmetic.py tests/test_doom_metal_decay.py tests/test_doom_metal_build.py tests/test_doom_metal_validation.py
python -m pytest -q
git diff --check
```

Use the provided LLVM 21 environment on Linux. Record Apple skips honestly.
Controller builds on M4 and reruns real arithmetic tests. Await its two actual
micrograph measurements before updating any golden digest. Preserve the
previous `6163f146b096db3ba7c086b2b1933977b380e3fec2cd08c6d7c7ce141b16d791`
epoch in comments/evidence. Independently review this task and any scoped fixes.

- [ ] **Step 5: Commit with complete TDD/self-review report.**

Stage only owned source/tests, use Sergey Subbotin <ssubbotin@gmail.com>, and
write actual RED/GREEN commands/output, files and unresolved concerns to the
task report. No SSH, source changes outside ownership or subagent dispatches.

### Task 2: M4 experiment and retained evidence (controller-owned)

**Files:**
- Create: `docs/experiments/2026-09-13-metal-arithmetic-invocation.md`
- Create: `docs/experiments/2026-09-13-metal-arithmetic.md`
- Create: portable JSON under `outputs/doom-learning/metal-arithmetic-20260913/`
- Update: `docs/doom-metal-backend.md` only for measured results and limits.

**Interfaces:**
- Consumes: reviewed Task 1 source/build, actual paired regressions and retained diagnostic protocol.
- Produces: exact-source arithmetic/40/80 ms/longer results, measured epoch and remaining hypothesis.

- [ ] **Step 1: Save exact invocation before execution.** Record source/build, strict flags, unchanged gates, fresh output and ignored target origins.
- [ ] **Step 2: Run actual M4 TDD controls.** Preserve test-only RED and updated GREEN; source remains fixed during jobs. Return unexpected findings to original worker.
- [ ] **Step 3: Repeat the existing micrograph twice.** Use the previous diagnostic's `paired_brains`, `run_trace`, capture and `_state_digest`; return actual matching evidence to the worker, then scoped review any golden update.
- [ ] **Step 4: Run unchanged full retained graph gates.**

```python
forty = validate.run(out / 'parity-40ms')
original = validate.TRACE
validate.TRACE = (('black', False), ('blue', False), ('green', False), ('white', True),
    ('left_blue', False), ('right_blue', True), ('vertical', False), ('horizontal', False))
try:
    eighty = validate.run(out / 'parity-80ms')
finally:
    validate.TRACE = original
```

- [ ] **Step 5: Re-run preserved long-horizon diagnostic.** Keep its 200 dark bins, same original first RGB, CPU initial checkpoint, teacher zero and fixed decoder. Only exact source/epoch/output identity changes. Compare first divergence, warmup metrics and decoded controls to the retained failure; diagnostic downloads invalidate throughput interpretation.
- [ ] **Step 6: Preserve failures and publish bounded evidence.** Actual target suite, artifact SHA/bytes, portable JSON checks, most-capable whole-component review, exact-file commit/push. If longer parity still fails, identify the next incoming/arithmetic hypothesis rather than weakening gates or stopping Metal work. Useful learning and public launch remain separate unestablished goals.
