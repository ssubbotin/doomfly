# Metal Training Prefix: Failed Longer Parity

## Identical inputs, different propagation

The saved [invocation](2026-09-13-metal-imitation-invocation.md) completed eight
35-frame phases on the M4 Pro at `a5d3145`, using the retained full graph and
4,184 identified plastic slots. Each phase includes two seconds of dark warmup
and one second of original GameWAM RGB. This diagnostic prefix is incomplete
relative to the source episodes. It establishes no useful learning.

The [CPU comparison](../../outputs/doom-learning/metal-imitation-prefix-20260913/comparison.json)
confirms identical model identity/configuration, decoded RGB hashes, raw controls,
targets and neural boundaries. Frozen held-out actions and whole-neuron spike
hashes differ on all 35 frames. The first decoded turn is +1.04365336 on CPU
and zero on Metal. Initial and erased frozen Metal runs reproduce one another
exactly. All graph/nonplastic invariants remain unchanged; every evaluation
uses zero teacher current and frozen efficacies.

Plastic and shifted Metal arms each change three slots, versus 1,687 and 1,701
in the corresponding CPU prefixes. Error feedback differs after the first
decoded frame because student propagation differs. This comparison establishes
neither equivalent training trajectories nor equivalent learned checkpoints.

## Measured cost and diagnostic follow-up

Metal phases take 7.98–8.18 wall seconds, versus 4.84–5.12 on CPU. Metal neural
execution excluding warmup takes 3.34–3.53 seconds per second of brain time;
its two-second warmup takes approximately 4.5 wall seconds. Passing cold 40/80 ms
parity and the separate 1.065x CPU short timing sample do not generalize to this
longer workload. Full Metal cohort remains unstarted while this discrepancy is
investigated; Metal implementation continues.

The separately [saved diagnostic](2026-09-13-metal-long-horizon-diagnostic-invocation.md)
reproduces the first-frame difference after 200 identical dark bins. The first
10 ms bin already differs in 7,118 voltages by at most 0.000003815 mV, with equal
conductances and spike counts. First spike mismatch is neuron 55450 at ticks
479 (CPU) and 478 (Metal). Two-second warmup Jaccard falls to 0.06476 and
within-one-tick agreement to 0.14503, with similar aggregate firing rates.
The unchanged parity thresholds are failed at this longer horizon.

The actual CPU binary contains fused multiply-add instructions. Artificial
controls also reproduce incoming conductance association and exclusively
modulatory settlement differences. Both implementations already settle fast
states at each bin boundary. The next experiment targets explicit CPU-compatible
voltage arithmetic, keeping incoming-event changes separate. Full prefix traces
remain locally/on target, ignored; portable protocol/results/diagnostic evidence
are retained. No public service or launch status changed.
