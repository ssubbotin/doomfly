# First Complete Offline Reinforcement Cohort

## Scope and source

Executed on the M4 Pro with reviewed code
`fb57fd1c57960fe5297c3532cd2f939cc16d0685`, CPU reference, eta 0.001 and existing
calibrated v6 dynamics. Retained MaleCNS v1.0: 166,700 neurons, 25,582,938
connections and 4,184 existing KC→MBON11 plastic slots. Live service unchanged.

Original synchronized GameWAM APPO recordings supplied training episode
`defend_the_center/000000` (982 frames) and distinct evaluation episode `000001`
(988 frames), pinned revision `6cc00d60462885c1d61dd480228af58c3b81b807`.
These are agent demonstrations. Source files/license notice remain with the
locally downloaded data. No game video or commercial assets are bundled.
See the [source investigation](2026-09-12-online-doom-demonstrations.md) for
artifact hashes, action compatibility, collector assumptions and local yaw assay.

## Execution and controls

All eight complete phases finished, totaling 7,886 original RGB frames and
2,253,143 neural ticks, plus separate warmup and retention. No pacing, cropped
connections, label-selected actions or game telemetry in decoding. Fixed
DNp20/DNpe017 controls receive actual modeled neural counts. Turn labels are
scored after decoding; error-driven PPL101 current enters the next frame.

The [audit](../../outputs/doom-learning/offline-imitation-pilot-20260913/audit.json)
checks every cumulative tick boundary, shifted target, next-frame teacher current
and integrated dose. Raw controls, timestamps and RGB hashes match across arms.
All evaluations impose zero teacher current and freeze efficacies. Neuron IDs,
CSR endpoints, nonplastic weights and selected-slot identity remain unchanged.
Frozen training changes zero plastic weights. Shifted-label control uses 491
frames; its feedback dose differs and is explicitly measured.

## Failed useful-learning result

Held-out turning, with 624 left, 176 idle and 188 right target frames:

| Condition | MAE (degrees/tic) | Balanced direction recall |
|---|---:|---:|
| Plastic | 3.25482 | 0.31492 |
| Frozen | 3.13551 | 0.36595 |
| Shifted labels | 3.33886 | 0.32230 |
| Plastic after five-second dark retention | 3.29618 | 0.31417 |
| Erased memory | 3.13551 | 0.36595 |
| Constant left descriptive baseline | 2.16808 | 0.33333 |
| Constant idle descriptive baseline | 2.70966 | 0.33333 |

Plastic training changed 2,022 selected weights; shifted training changed 1,979.
Neither beat frozen turning MAE. Erased and frozen controls reproduce actions,
spike hashes and memory hashes exactly. Retention reduces the plastic score
further. Constant baselines use target statistics only for analysis and never
control the modeled brain. This pilot demonstrates execution and changed memory,
with a failed useful turn-learning advantage. It establishes no fly learning or
biological validation. The single training/evaluation pair is not replication.

## Timing and reproducibility limits

Each complete 28-second source episode took 117–124 wall seconds, including its
approximately 2.3–2.5-second warmup. Median neural-backend fraction of timed
post-warmup episode wall time is 97.47%. This points to neural execution as the
main measured cost for these runs. Episode walls sum to 968.55 seconds; the
five-second dark retention adds 19.91 seconds separately. These measurements
exclude cohort construction, checkpoint I/O and invariant hashing outside episodes.
They establish neither Metal throughput nor exclusive-host benchmarking.

Every arm used one M4 FFmpeg 9.0.1 decoder. Default Linux/M4 RGB conversion differs;
the preserved first-frame diagnostic localizes its tested difference to optimized
YUV→RGB conversion. Original source angle entries are padded, so source engine
turn settings remain unresolved. The unchanged DNpe017 decoding couples movement
and firing; this stationary shooting source supervises turning only.

Portable results, protocol, source provenance, invariant hashes and checksum
inventory are published with this report. Complete traces and full checkpoints
remain preserved locally and on the target, ignored to keep Git evidence bounded.
The [saved transfer protocol](2026-09-13-offline-imitation-transfer-invocation.md)
tests independently restored checkpoints in the real hazard arena with explicitly
zero reinforcement. Its behavior results are reported separately. Continued
Metal development targets the measured numerical discrepancy with fixed parity
gates. Learning and scientific launch flags remain false.
