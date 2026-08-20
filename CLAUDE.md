# CPR Hybrid Compression-Rate Estimation

## Non-negotiable rules

1. **Development videos only** during implementation/tuning: `data/split/*_development.mp4`.
2. **Never run, inspect for performance, tune on, or use GT from `data/split/*_test.mp4`.** Test videos stay completely held out until the pipeline is finalized and frozen (there is a future, not-yet-written test-run script for that; do not create/run it now).
3. GT lives in `data/metadata/ground_truth_summary.csv`, keyed by `video{N}.mp4`. Split files map to it by stripping the `_development`/`_test` suffix (e.g. `video4_development.mp4` → `video4.mp4`). This mapping is confirmed unambiguous — see PROGRESS.md Phase 0 entry.
4. Do not fine-tune MediaPipe, CoTracker, or RepNet — pretrained checkpoints only.
5. GT is never an input to inference — only used post-prediction for evaluation/tuning. Never hard-code or nudge toward the known 30-compression count.

## Engineering standards (apply to every phase)

- **Config:** no hardcoded parameters — everything through the schema-validated config object (`src/hybrid/config.py`, `config/default.yaml`).
- **Logging:** shared logger only (`src/hybrid/logging_config.py`) — no bare `print`.
- **Errors:** typed exceptions (`src/hybrid/exceptions.py`) raised/logged at orchestration level — never a silently-returned `None`.
- **Testing:** every estimator gets a synthetic-signal unit test before its output is trusted on real video; full pipeline gets at least one integration test.
- **Caching:** expensive branch outputs (MediaPipe, CoTracker, RepNet) are cached per-video so tuning doesn't re-run them.
- **Reproducibility:** git commit hash, config hash, RNG/RANSAC seed recorded with every run.
- **Experiment ledger:** every dev run/tuning iteration appended to `runs/experiment_ledger.jsonl`.

## Environment constraints

- No discrete GPU on this machine (Intel Iris Xe integrated only, no CUDA) — CoTracker/RepNet run CPU-only. Expect this to be slow; build and validate correctness first, don't optimize runtime prematurely.
- Python 3.10.9, venv at `.venv/`. CoTracker (PyTorch) and RepNet (TensorFlow) may need isolated environments — see ADR 0003 once written.

## Hand-off

See `PROGRESS.md` for what's been done, current phase, and what's next. Read it at the start of every session before doing anything else.
