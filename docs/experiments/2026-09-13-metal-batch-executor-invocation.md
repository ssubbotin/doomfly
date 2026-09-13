# Resident Metal Lane Execution Record

Saved before actual native execution. Numerical parent is the reviewed epoch-5
revision b24249ecfee12aa84b55ed803ffe47bd09c487ad. Batch ABI is a separate
execution identity; epoch-6 carry arithmetic stays outside this worktree.
Exact machine/data/checkpoint origins live only in ignored configuration.

## Healthy test-only RED, then native GREEN

Set DOOMFLY_RUNTIME_COMMIT to the exact committed test-only checkpoint.
Use a fresh owned target checkout and the existing single-threaded dependency
interpreter/bootstrap native kernel. Assert HEAD and source cleanliness before
execution. Capture original stdout/stderr and exit status, without replacing
failure output. Run:

```sh
OPENBLAS_NUM_THREADS=1 python -m doom_learning_v6.metal.build --probe
OPENBLAS_NUM_THREADS=1 python -m pytest -q tests/test_doom_metal_batch_native.py
```

The test-only source must build/probe successfully. Missing new lane symbols
are expected feature RED. Compiler, path, import dependency or graph setup
failures are prerequisites and must be resolved before production changes.
Keep test-only source fixed until the command ends naturally.

After root returns the genuine RED, the original worker adds only the planned
native lane implementation. Set the runtime commit to its exact checkpoint;
require no active job before switching the owned target. Run the same tests,
then unchanged serial controls and the complete actual M4 suite:

```sh
OPENBLAS_NUM_THREADS=1 python -m pytest -q \
  tests/test_doom_metal_batch_native.py tests/test_doom_metal_state.py \
  tests/test_doom_metal_parity.py tests/test_doom_metal_arithmetic.py \
  tests/test_doom_metal_decay.py tests/test_doom_metal_build.py \
  tests/test_doom_metal_validation.py
OPENBLAS_NUM_THREADS=1 python -m pytest -q
```

Repeat the existing serial micrograph twice and compare the unchanged literal
65524d57f34b5477f9bf61b7546bfb79248d65426b7fec6733f3d2315d53bd09.
No new golden or tolerance hides a changed lane-zero result. Native lane
tests compare every lane, delayed membership, spikes, state and eligibility,
including lane permutation and neighbor perturbations.

State-upload guards must retain valid pre-zero sleeping histories. The first
implementation's fourteen existing decay failures occurred before dispatch;
the correction accepts safe int32 elapsed histories while rejecting overflow.
Keep those failures, test-only missing-symbol RED and corrected GREEN separate.

CPU fixtures regenerate five owned runtime library manifests under
`outputs/doom-learning/` (original, `physiology-v2`, `physiology-v4`,
`physiology-v5`, `physiology-v6`). Require clean code/tests; permit only these
exact generated-manifest paths after verifying each against its own current
`kernel.cpp`, actual binary and unchanged -O3/-std=c++17/-shared/-fPIC flags.
Preserve them unstaged. Any other tracked delta stops the job. An overly narrow
single-manifest guard stopped the first Task2 launch before checkout or tests;
retain that prerequisite failure separately from subsequent test failures.

## Training-boundary and retained evidence

Run actual batch Python/RGB training tests and the complete suite on Task2's
exact committed source before its completion/review gate. After Task2 review,
save the separate retained/prefix/full-RGB
script before running it. Require fresh source-matching 40/80 ms reports and
unchanged scientific thresholds; record their horizon and batch-lane scope.

The actual Task2 focused command is:

```sh
OPENBLAS_NUM_THREADS=1 python -m pytest -q \
  tests/test_doom_metal_batch.py tests/test_doom_metal_batch_rgb.py \
  tests/test_doom_learning_v6.py tests/test_doom_imitation.py \
  tests/test_doom_metal_checkpoint.py tests/test_doom_metal_batch_native.py
```

At source08015d7101778e6aa3fc88103418d811db1481e5 it reports173passes on
the M4 Pro. The unchanged serial command above reports86passes/one platform
skip; the full suite reports508passes/three platform skips/50baseline warnings.
These test times are not complete training throughput.

Compare two-second dark and each original RGB lane against its serial epoch-5
reference, not just another batch lane. Known carry cancellation failures,
long CPU trajectory failures, original inputs and all negative/mixed learning
controls stay preserved. Diagnostics with full state/spike capture are
correctness evidence and do not establish training throughput.

Timing runs use complete matched original episodes, unchanged next-frame
teacher and plasticity, no padding/completed-lane stepping, separate warmup,
aggregate command timing counted once, actual memory and fresh repeated
samples. N=4 follows N=2 identity and working-set headroom. A measured lack
of gain leads to profiling/refinement, without a scientific speed gate,
1000x forecast, useful-learning assertion or public launch.
