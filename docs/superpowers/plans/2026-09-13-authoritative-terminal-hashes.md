# Authoritative Terminal State Hash Correction

> REQUIRED workflow: superpowers:subagent-driven-development with TDD and independent review.

**Goal:** Make future terminal JSON state hashes describe materialized neural state.

**Evidence:** The private study at source `cec94940e370477bf21f0975725ecd2345bf9cba` logs terminal hashes before `brain.checkpoint()` materializes device state. A read-only first-wave check found mismatched `counts`, `v` and `queue_count` hashes, while checked memory/eligibility hashes agree. Preserve these original records and qualify their freshness. No action or loss calculation uses these terminal JSON hashes.

**Architecture:** Keep the existing backend materialization/ownership boundary. Synchronize terminal state before hashing. No simulation, installer, decoder, protocol, graph, learning-rule or kernel changes.

**Execution boundary:** The running M4 study remains at its pinned source and completes the fixed 120-attempt protocol without restart or adaptation. Implement locally while it runs. Deploy only after the real process exits and its original-source evidence is audited. A logging correction cannot retroactively validate stale hashes.

### Task 1: Materialize before terminal hashing

**Ownership:** `doom_learning_v6/causal_pilot.py` (terminal hash synchronization and its record-error preservation boundary only), new `tests/test_doom_learning_terminal_state_hashes.py`, ignored task report/raw logs. You are not alone; preserve others' changes. No child agents, SSH, GPU jobs or unrelated edits.

**Interfaces:** `_state_hashes(brain) -> dict` must hash the current complete 24-array state using the existing checked backend materialization API. Preserve cursor, total-spike metadata and current field names. CPU and Metal must remain supported. No host-state-validity bypass, upload, restore/reset, additional tick or weight-epoch increment.

- [ ] Write an authentic failing portable regression showing stale host state yields wrong hashes before implementation. Cover existing CPU behavior and materialization-error propagation.
- [ ] Add a real native Metal fixture regression that advances an owned small four-lane executor, confirms stale host state, and compares all terminal hashes with the subsequently saved 24-array checkpoint. Skip only absent Metal. This is numerical validation, not biological evidence.
- [ ] Run focused RED; retain command, actual exit and raw output. Fix fixture mistakes before accepting RED.
- [ ] Make the smallest checked synchronization change. Keep neural controls, traces, scoring, frozen weights and simulation clocks unchanged.
- [ ] On terminal materialization failure, keep the original primary error, persist accumulated frame traces with native terminal hashes explicitly unavailable, and attempt every available lane checkpoint through existing guarded paths. Never fall back to stale hashes, retry the download, reset/restore, bypass poison/ownership, or replace the primary error with a writer/cleanup failure. Add authentic failing regressions asserting persisted traces, unavailable-state flags and all checkpoint attempts for both an existing interruption and a newly primary download error.
- [ ] Run focused GREEN and relevant existing causal fixtures. Run the ordinary portable suite once before committing, record known warnings separately, and self-review. Do not run historical connectome matrices or scientific trajectories.
- [ ] Commit only owned source/tests with `Sergey Subbotin <ssubbotin@gmail.com>` and write the ignored report. Controller obtains fresh spec/quality review and later native focused evidence before future use.

## Controller follow-through

Preserve and report the measured source separately from the corrected source. Audit all saved checkpoints, baseline/filler equality, eligible-only weight changes and artifact integrity; explicitly flag unavailable authoritative terminal JSON hashes. Run no scientific replay merely to rewrite evidence. Keep all learning/launch claims false. Carry the already recorded direct-fit protocol-type Minor to final whole-subproject review.
