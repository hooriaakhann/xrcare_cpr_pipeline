# Progress Log

Append-only. One entry per finished phase (or meaningful chunk of one).

---

## Phase 0 — Inspect Before Coding (done)

**Development videos found (6):** video1, video2, video3, video4, video9, video10 (`data/split/video{N}_development.mp4`)
**Test videos found (4, untouched):** video5, video6, video7, video8 (`data/split/video{N}_test.mp4`)

**GT mapping:** unambiguous. Strip `_development`/`_test` suffix → matches `video{N}.mp4` row in `data/metadata/ground_truth_summary.csv` directly.

| split file | GT row | GT CPM |
|---|---|---:|
| video1_development.mp4 | video1.mp4 | 90.0 |
| video2_development.mp4 | video2.mp4 | 69.23 |
| video3_development.mp4 | video3.mp4 | 105.88 |
| video4_development.mp4 | video4.mp4 | 72.0 |
| video9_development.mp4 | video9.mp4 | 94.74 |
| video10_development.mp4 | video10.mp4 | 100.0 |

**Data integrity issue found and fixed:** cross-checked every `data/split/*.mp4` against its `data/processed/*_processed.mp4` source via `ffprobe`.
- `video4_development.mp4` was corrupt (`moov atom not found`, 7.0MB vs expected 27.9MB — truncated copy). **Fixed:** recopied from `data/processed/video4_processed.mp4`. Verified: 750 frames / 25.025s, matches.
- `video9_development.mp4` had a mismatched duration/frame count (19.74s/591 frames vs expected 19.00s/569 frames — not the raw video either, unexplained). **Fixed:** recopied from `data/processed/video9_processed.mp4`. Verified: 569 frames / 19.000s, matches.
- User approved both fixes as "recopy from processed" (2026-08-20). All 6 development videos are now verified valid and consistent with the GT-derivation source.

**Environment:**
- Python 3.10.9, venv at `.venv/`. Installed: `numpy 2.2.6`, `opencv-python 5.0.0.93`. Missing: mediapipe, torch, CoTracker, tensorflow, RepNet, scipy, pywt.
- No git repository yet.
- No discrete GPU — Intel Iris Xe integrated only, no CUDA. **Decision (user, 2026-08-20): proceed CPU-only for now**, revisit remote/cloud GPU later if CoTracker/RepNet runtime becomes a blocker.
- Existing repo already has a working preprocessing stage (`src/preprocessing/{inspect_videos,preprocess,compute_gt_cpm}.py`) writing to `data/processed/`, `data/metadata/timestamps/*_timestamp_map.csv` (per-frame original↔processed PTS mapping, reusable for Phase 1), and the three metadata CSVs. Videos confirmed VFR, HEVC 10-bit.

**Next phase:** Phase 0.5 — Project Scaffolding & Engineering Foundations (git init, `pyproject.toml`, config schema, logging, exceptions, caching skeleton, pre-commit, CI skeleton). Not yet started.

---

## Phase 0.5 — Project Scaffolding & Engineering Foundations (done)

**Git:** repo initialized (`git init`), `.gitignore` excludes `data/raw/`, `data/processed/`, `data/split/`, `data/cache/` (large video binaries/cache — not versioned) but tracks `data/metadata/**` (small, reproducibility-relevant, not regenerable without the raw videos). `runs/` is ignored except `.gitkeep` (the experiment ledger is local tuning history, not versioned). All scaffolding + existing `src/preprocessing/` + `data/metadata/` staged (40 files); **not yet committed** — awaiting explicit go-ahead per commit policy.

**Packaging:** `pyproject.toml` (hatchling, src-layout, package `hybrid` under `src/hybrid/`). Base deps: `numpy`, `opencv-python`, `pydantic>=2`, `pyyaml`. Dev extra (`.[dev]`): `pytest`, `ruff`, `pre-commit`. `requirements.txt` kept as-is for the existing preprocessing-only install path. Installed editable in `.venv` and verified.

**Config (`src/hybrid/config.py` + `config/default.yaml`):** Pydantic v2, `extra="forbid"` on every model (unknown YAML key fails validation, not silently ignored). Sections: `project` (seed=42), `paths` (data_root/raw/processed/split/metadata/ground_truth_csv/cache/runs — resolved to absolute at load time), `video` (dev/test glob patterns, matching CLAUDE.md rule 1-2 verbatim), `logging` (level, log_dir), `caching` (enabled). `load_config()` raises `ConfigError` (never returns partial config) on missing file / malformed YAML / schema violation. `HybridConfig.config_hash()` — sha256 of the resolved config, used for cache keys and the experiment ledger.

**Logging (`src/hybrid/logging_config.py`):** `setup_logging(level, log_dir)` configures the root logger once (idempotent), console + optional file handler. `get_logger(name)` everywhere else — no bare `print` in new code.

**Exceptions (`src/hybrid/exceptions.py`):** `HybridError` base, `ConfigError`, `CacheError`. More subclasses to be added by each phase as needed (e.g. branch-execution errors in Phase 1+).

**Caching (`src/hybrid/caching.py`):** `CacheManager(cache_dir)` — per-video branch-output cache keyed by `(stage, video_id, config_hash)` via `make_key()`, pickle-backed, atomic write (tmp file + replace), raises `CacheError` on corrupt/missing entries instead of returning `None`.

**Experiment ledger (`src/hybrid/experiment_ledger.py`):** `log_run(phase, config, metrics, video_id, extra)` appends one JSON line to `<runs_dir>/experiment_ledger.jsonl` with timestamp, git commit hash (`get_git_commit_hash()`), config hash, and seed auto-filled — satisfies the reproducibility + experiment ledger standards for every future tuning run.

**Tooling:** `.pre-commit-config.yaml` (trailing-whitespace, end-of-file-fixer, check-yaml, check-added-large-files @1MB — guards against accidentally committing raw video, mixed-line-ending, `ruff` lint + `ruff-format`). `Makefile` (install/lint/format/test/precommit/clean targets, all via `.venv/Scripts/python`). `.github/workflows/ci.yml` — ubuntu-latest, Python 3.10, installs `.[dev]`, runs `ruff check`, `ruff format --check`, `pytest`.

**Tests:** `tests/test_{config,caching,exceptions,experiment_ledger,logging_config}.py` — 18 tests, all passing, covering the scaffolding itself (schema validation incl. rejection of unknown fields, cache round-trip/corruption/key-scoping, ledger append-not-overwrite, logging idempotency). No real video I/O in these — pure synthetic/tmp-path fixtures, so CI never touches `data/`.

**Side effect — pre-existing `src/preprocessing/` cleanup:** introducing ruff surfaced 4 real lint errors in the Phase-0 preprocessing scripts (2 dead variables in `inspect_videos.py`, 1 unused import + 1 unnecessary `open()` mode arg in `preprocess.py`/tributaries) — fixed, zero behavior change, verified via `py_compile`. Since the repo had zero commits at the time, also let `ruff-format` normalize whitespace/line-wrapping across all of `src/preprocessing/` once, before any git history existed, rather than deferring it to a disruptive later reformat. Logic untouched; not re-run against real videos (would violate the "never run on test videos" rule anyway — these are formatting-only changes, verified by compiling, not by re-running the pipeline).

**Next phase:** Phase 1 (per `cpr_hybrid_pipeline_prompt_v2.md` — not present in this repo/filesystem; if it's meant to be the source of truth for phase specs, it needs to be added to the repo or its contents pasted in, since this session had no access to it beyond the phase list already summarized in prior sessions).
