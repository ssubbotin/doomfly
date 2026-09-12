# Teacher-Free Live Doom Transfer

Executed the [saved protocol](2026-09-13-offline-imitation-transfer-invocation.md)
after the complete offline cohort finished. Same M4 Pro, reviewed `fb57fd1`
code, full MaleCNS v1.0 graph, unchanged calibrated dynamics and fixed
DNp20/DNpe017 decoding. All twelve independently restored evaluation episodes
finished. Every evaluation froze efficacies and supplied an explicit all-zero
reinforcement schedule. Observer pose and health never selected actions.

## Paired seed results

Survival seconds in the original two-sector hazard arena (12-second cap):

| Seed | Plastic | Frozen | Shifted labels |
|---|---:|---:|---:|
| 61031 | 3.657, died | 3.657, died | 3.657, died |
| 61032 | 3.657, died | 10.971, died | 6.400, died |
| 61033 | 12.000, censored | 3.657, died | 6.400, died |

Plastic seed 61033 ended alive with health 60 and observer x=19.846, in the safe
sector at that boundary. Plastic seed 61032 died earlier than both controls.
This is a seed-dependent route change following changed weights. It establishes
no replicated learning advantage. The successful cap is right-censored survival,
not an escape claim or a full learned Doom policy. Offline held-out turning MAE
also favored frozen weights, as recorded in the [cohort report](2026-09-13-offline-imitation-pilot.md).

## Controls and limits

Preselected seed 61031 erasure, five-second dark retention and black-vision
controls all died at 3.657 seconds. Erasure reproduces frozen actions, spike
hashes, memory hashes and live RGB hashes exactly on that seed. Since the
plastic arm also dies there, these controls do not test erasure/retention of
the seed-61033 survival change. That responder needs a separately declared
follow-up plus independent training replicas before any benefit claim.

The [audit](../../outputs/doom-learning/offline-imitation-transfer-20260913/audit.json)
verifies every cumulative neural boundary, zero reinforcement, frozen weight
hashes and unchanged neuron IDs/CSR/nonplastic weights/selected slots. It records
observer endpoints and complete-trace hashes. Original complete JSON traces are
preserved locally and on the target, excluded from Git; portable results and
source provenance are published. No game imagery, generated WAD or IWAD is
bundled. Prescribed-action map validation is an environment check only.

This exploratory transfer cohort is one checkpoint per training arm and three
test seeds. It provides no independent training replication, biological
validation, successful learning certification or public launch green light.
The live service remains unchanged. Continued Metal work now has passing
bounded 40/80 ms numerical evidence; useful learning remains unresolved.
