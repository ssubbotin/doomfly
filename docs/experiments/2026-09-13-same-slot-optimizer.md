# Same-slot Metal optimizer study: factual report

Status: complete private offline study. This report records numerical efficacy fitting on recorded RGB. It does not establish fly learning, a Doom skill, or launch readiness. The public claim flags remain `learning_demonstrated=false`, `doom_skill_demonstrated=false`, and `announcement_ready=false`.

## Question and model

The study asked whether two bounded numerical methods could improve a fixed full-connectome replay objective by changing only the existing positive KC to MBON11 efficacies. The retained MaleCNS v1.0 graph contains 166,700 modeled neurons and 25,582,938 released connections. All 4,184 eligible slots remained present. The selected model was `adaptive-centered-v6`, with `eta=0.001`; every other connection, sensory mapping, modeled dynamics, and decoder stayed fixed.

Recorded RGB frames drive the existing sensory path and neural propagation. Fourteen fixed readouts feed the fixed BCI decoder. The DNp20 readouts provide turn control. DNpe017 readouts provide forward and attack controls. Recorded action labels supplied the offline angular objective only. They did not select runtime controls or steer the neural replay. External reinforcement was zero during every dark warmup and RGB replay. Endogenous DAN activity could still be nonzero. The biological-proxy plasticity rule stayed unchanged and inactive during replay. The optimizer changed the efficacy vector between rollouts, with `memory_u=memory_w=f-1` and the corresponding eligible edge weights. It did not change the graph, decoder, controls, or neural rule.

The optimizer used four fixed generations with probe scales `[0.50, 0.35, 0.25, 0.20]`, step lengths `[0.20, 0.15, 0.10, 0.10]`, and `f=1+0.1*tanh(theta)`, bounded to `[0.9, 1.1]`. Each proposal was accepted only after a strictly lower equally weighted mean of the two complete training episodes. Probe results were never accepted directly. Training used episodes 2 and 3, with 948 and 897 complete frames. Held episodes were 4 and 5, with 1,036 and 990 frames. The held set was scored only after all seven vectors were frozen. Two independent fitting replicas were run for SPSA and Gaussian ES. Two predeclared random vectors were held-only controls, generated independently with seeds 200000301 and 200000302 and uniform fractions in `[0.9, 1.1]`. They had no training scores and no selection role.

Inputs were `Yunncheng/gamewam-vizdoom`, revision `6cc00d60462885c1d61dd480228af58c3b81b807`, scenario `defend_the_center`. Dataset and publisher metadata were inspected before probes. The split was therefore not fully blinded. Historical episodes 0 and 1 were preserved without rerunning the old matrix. The collector metadata was APPO `sample_factory_appo_rnn`, `tick_smooth_native`, frame skip 1, action repeat 4.

## Training decisions

The baseline training-bank loss was 2.6316897285 degrees. Each row gives accepted proposals over four generations. The proposal values are the complete two-episode training means used by the strict comparison.

| Method and replica | Generation 0 | Generation 1 | Generation 2 | Generation 3 | Accepted |
| --- | ---: | ---: | ---: | ---: | ---: |
| SPSA, seed 200000201 | 2.554758 (yes) | 2.570422 (no) | 2.657667 (no) | 2.539299 (yes) | 2/4 |
| SPSA, seed 200000202 | 2.499211 (yes) | 2.595315 (no) | 2.629821 (no) | 2.667857 (no) | 1/4 |
| Gaussian ES, seed 200000201 | 2.542016 (yes) | 2.572793 (no) | 2.554525 (no) | 2.553535 (no) | 1/4 |
| Gaussian ES, seed 200000202 | 2.608940 (yes) | 2.579151 (yes) | 2.595785 (no) | 2.608694 (no) | 2/4 |

Method-level acceptance totals were 3/8 for SPSA and 3/8 for Gaussian ES. Terminal training losses were 2.53929862697 and 2.49921088550 for SPSA, and 2.54201592196 and 2.57915125291 for Gaussian ES. These are fitting-bank results, with no claim that they transfer to gameplay.

## Held outcomes

The table reports every frozen role on both held episodes. `Delta` is the balanced angular MAE difference from the same-episode baseline. Negative values indicate a lower held loss. Forward is the mean decoded forward control diagnostic. Attack is the decoded attack fraction.

| Frozen role | Episode 4 MAE | Episode 4 delta | Episode 4 forward | Episode 4 attack | Episode 5 MAE | Episode 5 delta | Episode 5 forward | Episode 5 attack | Equal held mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 2.535240 | 0.000000 | 0.632722 | 0.039575 | 2.677517 | 0.000000 | 1.032066 | 0.059596 | 2.606379 |
| SPSA 200000201 | 2.513921 | -0.021319 | 0.938309 | 0.060811 | 2.660711 | -0.016807 | 0.820167 | 0.045455 | 2.587316 |
| SPSA 200000202 | 2.653508 | 0.118269 | 0.794872 | 0.050193 | 2.662629 | -0.014889 | 1.137460 | 0.069697 | 2.658069 |
| Gaussian ES 200000201 | 2.589399 | 0.054159 | 0.986049 | 0.062741 | 2.577536 | -0.099982 | 1.518428 | 0.092929 | 2.583467 |
| Gaussian ES 200000202 | 2.536484 | 0.001244 | 0.943526 | 0.057915 | 2.630543 | -0.046975 | 0.862596 | 0.050505 | 2.583513 |
| random 200000301 | 2.520490 | -0.014750 | 0.770396 | 0.049228 | 2.706790 | 0.029272 | 0.763662 | 0.044444 | 2.613640 |
| random 200000302 | 2.554603 | 0.019363 | 0.700706 | 0.046332 | 2.649868 | -0.027650 | 0.946397 | 0.054545 | 2.602235 |

Equal-episode method summaries were SPSA 2.6226921295 degrees, delta +0.0163135438; Gaussian ES 2.5834903439 degrees, delta -0.0228882417. The two SPSA replica means were 2.5873157210 and 2.6580685379, a range of 0.0707528169 degrees. The two Gaussian ES means were 2.5834672823 and 2.5835134056, a range of 0.0000461232 degrees. The random-control mean across both vectors was 2.6079376193 degrees, delta +0.0015590336. No held winner was selected. The held outputs were not used for updates.

The diagnostic forward and attack values vary alongside the fitted vectors, while the fixed decoder and recorded RGB path remain unchanged. They are diagnostics of this replay and do not demonstrate a playable controller. The angular mapping is a repository conversion; publisher angular calibration was not independently established.

## Accounting and validation

The run charged 120 of a maximum 128 resident-lane attempts: 8 baseline lanes, 64 probe lanes, 32 proposal lanes, and 16 held lanes including the two baseline fillers. There were 30 four-lane waves. All four resident lanes were advanced and charged, including padding and fillers. Every scheduled lane included a 2-second frozen dark warmup.

The controller audit found complete frame counts, control separation, all 124 checkpoints (120 terminal plus four initial) with all 24 fields and valid schema, exact baseline/filler all-24 comparisons, eligible-only weight differences, exact physical identity before and after at the measured source, and exact checksums for all 28 input files. The independent RNG and update-array audit reproduced 261 array pins. The seven held vectors were pinned before scoring. All four generations, both method replicas, both static controls, and all rejected proposals remain represented in private evidence.

The measured study source was commit `cec94940e370477bf21f0975725ecd2345bf9cba`. The immutable [protocol](2026-09-13-same-slot-optimizer-protocol.json) SHA256 was `fae8309b51cbc24b63b23fcfaf9c4a5e5f5d141a0ce060bee3d2913821fe9c03`. The complete retained graph and build/reference identities are recorded in the [compact JSON](../../outputs/doom-learning/same-slot-optimizer-m4pro/results.json). The centered-rate-rule source is [Nature DOI 10.1038/s41586-024-07819-w](https://doi.org/10.1038/s41586-024-07819-w). Its use here is an interpreted centered rate rule on existing individual edges. It is not an exact reproduction of the published nine-unit model. Trace constants, efficacy bounds, gain, and the 50 ms weight filter still require calibration.

The corrected logger was tested separately at source `49c138c8c617df920a301ceba7c4a12bcc6f4a69`. The native focused validation envelope passed 13 tests, skipped 0, warnings 0, exit 0, in 0.618689 seconds. Its printed pytest duration was 0.26 seconds. The ordinary full native validation envelope passed 1,486 tests, skipped 3, warnings 50, exit 0, in 32.464775 seconds. Its printed pytest duration was 31.99 seconds. The envelope clocks include pressure/test execution and the post-test physical pin audit. Both suites had zero full-connectome study attempts and provide numerical fixture validation only. Existing Brian2, Pyparsing, Cython, setuptools, and distutils deprecation warnings remain recorded debt.

Final caller-availability and exact protocol-type corrections were reviewed and tested at source `845fe29b5f58eea6dce4e194d2ef5169ea6b55f1`. Covering runner, replay, terminal-hash and control fixtures passed 172 tests on M4, with zero skips or warnings, exit 0. Printed pytest time was 14.40 seconds; the validation envelope was 14.751424 seconds with the same pressure/test/physical-audit scope. Before/after pins matched all 144 model/test source files and the unchanged graph, CPU, native and reference core. Free memory stayed at 86 percent, swap stayed at 0 MiB, and the GPU lease was released. These synthetic CPU and real four-lane Metal fixtures used zero full-connectome study attempts. The earlier 1,486-test full-suite proof belongs to source `49c138c`, rather than this final amended source.

The original terminal JSON native hashes are not authoritative. There are 1,680 known stale terminal fields because the original logger hashed host arrays before device materialization. The original JSON and checkpoints remain preserved. Corrected-source fixtures do not retroactively validate those original hashes.

## Timing and resources

These clocks have narrow scopes. Replay wall time summed to 5,197.240958 seconds. Batch advance elapsed time summed to 4,491.852077 seconds, with one shared four-lane batch copy per wave. The RGB loop summed to 5,018.671646 seconds. The allocation-to-release boundary was 5,256.891884 seconds and excludes preflight, final audit, and SSH. Lane brain time was 3,204.2292 seconds, with 240 seconds of warmup. These figures include replay setup, warmup, RGB work, and checkpoint writes where applicable. They are not GPU-only or end-to-end command times, and they provide no speed forecast or CPU comparison.

The run used 413,006,712 shared resident bytes and 737,265,700 mutable resident bytes. Every pressure boundary was accepted, minimum free memory was 86 percent, and maximum swap use was 0 MiB. AC-scoped `caffeinate -is` was active. Raw private study storage was 16,722,052,884 bytes in durable ignored storage. Native checkpoints and private origins are not bundled here.

Software was observed during the immutable run, after probes rather than in a pre-probe capture: Python 3.11.15, NumPy 1.24.4, PyArrow 20.0.0, ViZDoom 1.3.0, Numba 0.61.2, Pillow 11.3.0, arm64, macOS 26.6.2.

## Limits, controls, and conclusion

The evidence supports a small Gaussian ES mean improvement on these two held episodes and mixed SPSA replicas. It does not support a robust method conclusion. The study has two training episodes, two held episodes, two fitting replicas per method, and dependent frames and contiguous blocks. Those units do not support frame-independent uncertainty. Selection was not fully blinded because metadata and quality information were inspected before the probes.

Preserved negative evidence includes historical causal and sensitivity controls, the original source-clean preflight exit 1 before allocation and attempts, the failed acquisition and logger RED logs, the corrected read-only checkpoint-audit cleanup path, and stale original-source hashes. Numerical loss changes and changed efficacies alone do not establish biological learning. A live closed-loop Doom evaluation with independently frozen weights remains required for that claim. This report is private study evidence and is not a public launch announcement.
