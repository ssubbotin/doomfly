# Metal First-Divergence Diagnostic

## Question and Scope

Locate the numerical mismatch behind the preserved two-dispatch 80 ms parity
failure before changing the backend. This diagnostic uses the unchanged
`adaptive-centered-v6` sources from `2dd86b5` on the M4 Pro. It retains all
166,700 neurons, 25,582,938 released connections, and 4,184 plastic slots.
It provides numerical evidence only; the physiology and learning gates remain
unresolved. No training scheduler, kernel, decoder, or deployed model changed.

## Method

Construct identically calibrated CPU and Metal brains from the original baseline.
Capture every spike while presenting eight procedural RGB frames for 10 ms each:
black, blue, green, white, left blue, right blue, vertical, and horizontal.
White and right blue receive the existing +4 PPL101 stimulation. Keep learning
enabled, 0.1 ms integration, and the original 10 ms rate-rule updates.
Download Metal state after each original bin. This observation adds no
integration or upload and leaves its resident execution state intact.

Two executions reproduced the original per-bin counts and complete 80 ms spike
metrics. The recorded execution reports its own artifact hashes; CPU and native
Metal library hashes differ from the older build, while the metallib and all
tracked validation-source hashes match. Configuration signatures match between
backends. The original failure report is preserved separately.

## Observations

| Bin ends at | Frame | CPU spikes | Metal spikes | Events present in only one backend |
|---:|---|---:|---:|---:|
| 10 ms | black | 0 | 0 | 0 |
| 20 ms | blue | 10,973 | 10,973 | 40 |
| 30 ms | green | 4,829 | 4,829 | 0 |
| 40 ms | white | 5,524 | 5,524 | 16 |
| 50 ms | left blue | 11,180 | 11,195 | 251 |
| 60 ms | right blue | 6,911 | 6,903 | 526 |
| 70 ms | vertical | 6,472 | 6,481 | 735 |
| 80 ms | horizontal | 6,320 | 6,331 | 1,541 |

The first differing spike is at tick 178 (17.8 ms). Matching counts through
40 ms therefore do not imply matching spike times. At 80 ms, Jaccard is
0.9421871804 and the within-one-tick fraction is 0.9879393522. Decoder decisions
and plastic weights remain equal. These are the same failed timing metrics as
the earlier stress report, rather than evidence of a new failure.

Voltage already differs in 7,118 entries after the initial black bin, with a
maximum error of 0.000274658203125 mV. At that point both backends have zero
spikes, and conductance, modulation, adaptation, drive, previous drive,
refractory state, activity flags, and timestamps match. Arithmetic drift thus
precedes synaptic delivery and reinforcement. Drive matches in every later bin.

A separate synthetic one-neuron, zero-edge control isolates this path. With
rest/initial voltage -52 mV, constant drive 12, and zero conductance/adaptation,
one integration step differs by one float32 voltage ULP. After 100 steps, CPU
voltage is -47.278480529785156 and Metal is -47.27820587158203, exactly the
largest black-bin pair. Passive, conductance, and adaptation controls are also
preserved. This synthetic graph is a numerical control, not a cropped or
replacement connectome.

## Compiler Evidence and Remaining Questions

The actual compiler's `-###` dry run expands the current build into
`-ffast-math`, approximate functions, reassociation, reciprocal math,
`-ffp-contract=fast`, and fast float32 math functions. A dry run with
`-fno-fast-math -ffp-contract=off` selects precise float32 functions and disabled
fusion. Neither dry run compiles or changes a binary.

This identifies a contributing arithmetic path and a concrete next diagnostic.
It does not establish whether approximate decay, expression reassociation,
fusion, or their combination causes the measured ULP difference. Strict math
has not yet been tested in an executed build and is not an established fix.
The target gather also sums arriving weights from zero before adding to `g`;
CPU adds each weight directly to `g` in delayed-queue order. That separate
accumulation difference remains relevant after arrivals begin.

## Evidence and Reproduction

Portable measurements are under
`outputs/doom-learning/metal-first-divergence-20260912/`: `report.json`,
`edgeless-control.json`, and `compiler-math-modes.json`. Raw compiler paths,
datasets, graph archives, dependencies, generated binaries, and game assets
are excluded.

Reproduce the first-bin observation on the validated Apple Silicon environment:

```python
import numpy as np
from doom_learning_v6.calibration import calibrated_brain
from doom_learning_v2.vision import frame_for

cpu = calibrated_brain(backend='cpu')
metal = calibrated_brain(backend='metal')
try:
    for brain in (cpu, metal):
        brain.rgb_step(frame_for('black'), 10, learning=True)
        brain.backend.materialize('first-bin-diagnostic')
    print(np.count_nonzero(cpu.v != metal.v))
    print(np.max(np.abs(cpu.v.astype(float) - metal.v.astype(float))))
finally:
    cpu.backend.close()
    metal.backend.close()
```

Use `OPENBLAS_NUM_THREADS=1`. Expected: 7,118 differing entries and
0.000274658203125 maximum absolute error for the recorded default build.

```sh
xcrun -sdk macosx metal -### -std=macos-metal2.4 \
  -c doom_learning_v6/metal/kernels.metal -o /dev/null
```
