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

**Correction (next session):** this entry originally said the Phase 0.5 work was staged but not committed, "awaiting explicit go-ahead." It was in fact committed and pushed at the start of the Phase 1 session below — see that entry for the git/GitHub setup story (no commits existed yet at that point, and no `origin` remote existed either, despite an assumption that it did).

---

## Phase 1 — Shared Video / Timestamp Handling + Dev-Set/GT Loader (done)

**Repo/GitHub reconciliation (before any Phase 1 code):** the repo had zero commits despite PROGRESS.md's Phase 0.5 entry describing everything as ready; `git status` confirmed "No commits yet." Also, no `origin` remote existed at all. Resolved in order: (1) committed the already-staged Phase 0.5 tree as the first commit; (2) `git ls-remote` on the assumed URL (`.../cpr-hybrid-pipeline.git`) failed auth entirely — no `gh` CLI, no SSH key, stale/missing Windows Credential Manager entry — user fixed the credential; (3) retried and got "Repository not found" — the repo didn't exist under that name; user confirmed the actual (already-created) repo is `hooriaakhan6-prog/xrcare_cpr_pipeline`; (4) pointed `origin` at that URL, fetched, found a single GitHub-auto-generated "Initial commit" (bare `README.md` with just a title) already on `main`; (5) renamed local `master` -> `main`, merged with `-X ours` (our fuller README wins on conflict), pushed. `origin/main` is now live and tracked.

**`src/hybrid/video_io.py`:** `VideoReader` (context-manager, single-pass iterator) + `Frame` (index, timestamp_sec, full uncropped BGR image) + `VideoMeta`/`probe_video_meta()`. Timestamps come from `ffprobe -show_entries frame=pts_time` (frame-accurate, VFR-aware), never from `cv2.CAP_PROP_POS_MSEC` (unreliable on these HEVC/VFR files) — OpenCV (`cv2.VideoCapture`) is used only to decode pixel data. Frame index and real PTS are paired 1:1 by decode order. Cross-checks cv2's reported frame count against ffprobe's PTS count and logs a warning (not silent) if they disagree by >1%. Raises `VideoReadError` (missing file, unopenable, zero decodable frames, re-iteration of a single-pass reader) or `TimestampError` (ffprobe missing/failed, malformed output, no timestamps, non-monotonic sequence) — never returns `None`.

**`src/hybrid/dataset.py`:** `discover_development_videos(config)` globs `config.paths.split_dir` for `config.video.development_glob`'s filename pattern (directory from resolved paths config, name pattern from video config — both config-driven, no hardcoded path), maps each `video{N}_development.mp4` to its `video{N}.mp4` ground-truth row via `map_split_filename_to_gt_key()` (strips `_development`/`_test` suffix per CLAUDE.md rule 3), and returns `DevVideo` records (video_id, split_path, GroundTruth). Raises `GroundTruthMappingError` on any unrecognized filename pattern, missing/duplicate GT row, or duplicate video id — nothing is silently skipped or guessed. `*_test.mp4` files are never matched by the development glob, so they can't leak in even if present in the same directory (verified by test).

**Exceptions added** (`src/hybrid/exceptions.py`): `VideoReadError`, `TimestampError`, `GroundTruthMappingError`, plus the Phase 2-10 exceptions from the spec skeleton added now so later phases just import them — `HandNotDetectedError`, `TrackLostError`, `EgoMotionUnreliableError`, `OpticalFlowUnstableError`, `RepNetUnavailableError`.

**Tests:** `tests/test_video_io.py` (13 tests — synthetic-video read-through, frame/timestamp pairing, meta-vs-reader agreement, single-pass guard, missing/corrupt file errors, mocked ffprobe failure modes: not-on-PATH, malformed JSON, empty output, non-monotonic timestamps) and `tests/test_dataset.py` (13 tests — filename mapping incl. multi-underscore video ids, GT CSV parsing/duplicate/malformed-row errors, dev-video discovery incl. sort order, missing-GT-row error, no-files-found error, and an explicit test that a co-located `*_test.mp4` is never picked up). All synthetic/tmp-path — no real video I/O in CI. Full suite: **44/44 passing**, `ruff check` clean.

**Real-data sanity check** (read-only, no algorithms — Phase 1 I/O only, per CLAUDE.md rule 1): `discover_development_videos()` against the real config found and correctly GT-mapped all 6 development videos (video1/2/3/4/9/10, matching the Phase 0 table). Read `video10_development.mp4` end-to-end via `VideoReader`: 539 frames, timestamps 0.0330s -> 17.9747s, inter-frame interval **0.0333s-0.0417s (not constant)** — confirms VFR timing is being preserved rather than assumed. `video7_test.mp4` (present in the same `data/split/` directory) was correctly excluded by the development glob.

**Decisions:** (1) dev-video sort order is lexical by filename (`video10` sorts before `video1`), not numeric by video id — fine since nothing downstream depends on ordering, noted here in case it matters later. (2) `VideoReader` is deliberately single-pass/forward-only (matches sequential per-frame processing needs of Phases 2-7; re-reading requires a new instance) — re-iterating raises `VideoReadError` rather than silently returning nothing.

**Next phase:** Phase 2 — MediaPipe Hand/Wrist Localization.

---

## Phase 2 — MediaPipe Hand/Wrist Localization (done)

**Dependency note:** the installed `mediapipe` (1.0.1, current PyPI release) only ships the newer Tasks API (`mediapipe.tasks.python.vision.HandLandmarker`) — the legacy `mp.solutions.hands` API used in most older tutorials/examples does not exist in this build (`mediapipe.python` submodule is gone). Implemented against the Tasks API, verified its exact option/field names and result shape empirically against the installed package before writing the module (`HandLandmarkerOptions` fields, `RunningMode.VIDEO`, `detect_for_video(image, timestamp_ms)`, `result.hand_landmarks` / `result.handedness` shapes) rather than assuming from memory. `mediapipe` pulls in `opencv-contrib-python` as its own cv2 dependency; `pyproject.toml` now depends on `opencv-contrib-python` instead of `opencv-python` (superset, `pip check` clean, `cv2.__version__` unchanged at 5.0.0).

**Pretrained checkpoint (`src/hybrid/models.py`, new):** `HAND_LANDMARKER` pins the official MediaPipe `hand_landmarker.task` (float16, Google's model repo) by exact URL + sha256. `ensure_model()` downloads into `paths.models_dir` (new config path, default `models/`, gitignored — large binary) only if missing, and verifies the checksum on every call, raising `ModelDownloadError` on failure/mismatch rather than silently using a possibly-wrong file. Pretrained only, never fine-tuned (CLAUDE.md rule 4) — the checksum is what makes "which checkpoint produced this run" reproducible without committing the binary.

**`src/hybrid/mediapipe_roi.py`:** `MediaPipeHandLocalizer` (VIDEO-mode `HandLandmarker`, one hand by default) detects per-frame: 21 hand landmarks (pixel coords), a tight bounding box, a **dynamic CPR-foreground ROI**, and a handedness-derived confidence score. The ROI is the tight bbox padded by `roi_padding_ratio`, then extended toward the forearm — continuing the middle-finger-MCP-to-wrist direction past the wrist by `forearm_extension_ratio x` the bbox's long side — so the region covers wrist + part of forearm, not just the hand, per spec. Never crops the frame itself; `HandDetection.image` is never stored, only coordinates, so the full frame stays available for Phase 4 ego-motion. `run_mediapipe_on_video()` iterates a whole video via the Phase 1 `VideoReader`, computes detection rate and longest no-detection gap (frames + seconds), logs a warning (not silent) when the gap exceeds `mediapipe.max_detection_gap_sec`, and raises `HandNotDetectedError` if the hand is never found in the entire video. `get_cotracker_init_points(detection, num_points)` returns the 21 landmarks first, then fills any remainder with a grid sampled over the ROI — this is what Phase 3 will call to seed CoTracker.

**Caching:** `run_mediapipe_on_video_cached()` wraps the above through the Phase 0.5 `CacheManager`, keyed by `(video_id, hash(mediapipe sub-config + pinned model sha256))` — scoped to only the MediaPipe config, so tuning unrelated parameters (Butterworth, fusion weights, ...) in later phases never invalidates this branch's cache.

**Config (`config/default.yaml` + `config.py`):** new `mediapipe:` section — `min_hand_detection_confidence`/`min_hand_presence_confidence`/`min_tracking_confidence` (0.5 each), `max_num_hands` (1), `roi_padding_ratio` (0.5), `forearm_extension_ratio` (1.2), `max_detection_gap_sec` (1.0s) — all schema-validated (confidences in [0,1], counts >=1, ratios/thresholds >=0). New `paths.models_dir` for checkpoint storage.

**Tests (`tests/test_mediapipe_roi.py`, 20 tests):** 18 fast/pure-logic tests (ROI geometry — clip/pad/forearm-extension math, longest-gap detection incl. start/end/multiple-gap cases, CoTracker init-point generation incl. landmark-then-grid fill and error cases, config-hash sensitivity, and the caching wrapper's hit/miss/disabled/config-change behavior via a monkeypatched `run_mediapipe_on_video` — no model needed, run in CI) + 2 `@pytest.mark.slow` tests that download/run the real model (blank-video → `HandNotDetectedError`; real dev-video frames → hand found, 21 landmarks, ROI wider than bbox — skips gracefully if `data/split/` isn't present locally, e.g. in CI). Introduced the `slow` pytest marker now (registered in `pyproject.toml`, `make test` excludes it, new `make test-all` includes it, CI updated to `pytest -m "not slow"`) since this is the first phase with a real network/model dependency — anticipates the Phase 19 test-tiering requirement rather than leaving CI to silently start downloading a model on every push. Full suite: **64/64 passing** (62 fast + 2 slow, run separately here), ruff clean.

**Real-data sanity check** (2 of 6 dev videos, via the real config/cache): `video10_development.mp4` — **539/539 frames (100.0%) detected, 0 gap**. `video1_development.mp4` — **597/599 frames (99.7%) detected, longest gap 1 frame**. CoTracker init points (40 requested) generated correctly for both (21 landmarks + 19 grid-filled). Cache verified: cold run ~21-23s/video, cached re-run **0.05s**.

**Decisions:** (1) Tasks API over legacy `solutions.hands` — not a choice, the installed package doesn't have the legacy API at all. (2) `max_num_hands=1` default — CPR compressions are single-handed (or two hands stacked, tracked as one region) in this dataset; configurable if that assumption needs revisiting. (3) ROI/forearm-extension parameters (`roi_padding_ratio`, `forearm_extension_ratio`) are first-pass defensible defaults, not yet tuned — real tuning happens in Phase 15 against downstream CPM accuracy, not in isolation here.

**Next phase:** Phase 3 — CoTracker Multi-Point Tracking.

---

## Phase 3 — CoTracker Multi-Point Tracking (done)

**Dependency:** installed CPU-only PyTorch (`torch==2.13.0+cpu`, via the `download.pytorch.org/whl/cpu` index — avoids pulling multi-GB CUDA wheels on this GPU-less machine) into the *same* venv as MediaPipe. `pip check` clean, no conflicts — so ADR 0003's "isolated venv + subprocess" default (Phase 0.5) was **not needed** for CoTracker; deferred to Phase 10/RepNet, where a TF1/TF2-era research repo is more likely to actually collide with the main env. CoTracker3 ("scaled offline", the pretrained checkpoint, never fine-tuned — CLAUDE.md rule 4) loads via `torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")`; verified its exact predictor API (`CoTrackerPredictor.forward(video, queries=...)`, `queries` shape `(B,N,3)` as `(frame_idx,x,y)`, output `(tracks (B,T,N,2), visibility (B,T,N) bool)`) by reading the actual cached source (`~/.cache/torch/hub/facebookresearch_co-tracker_main/{hubconf,demo,cotracker/predictor}.py`) rather than assuming from memory. CI installs the CPU torch wheel explicitly too (same reasoning). `pyproject.toml` now lists `torch`.

**Memory-driven design constraint (found before writing any orchestration code):** the offline model holds its whole input video tensor in memory at once. At this dataset's native resolution (~2000x1500), a ~600-frame video would need **~20GB** just for the float32 tensor — this machine has 16GB total RAM (~8.9GB free). Confirmed via a timed smoke test on real frames (90 frames/40 points: 66s at native res, 55s resized to 2x model resolution — resizing barely helps *speed*, since CoTracker downsamples internally regardless, but it's a **hard memory necessity**, not just a speed optimization). This is why `src/hybrid/cotracker_tracker.py` processes videos in memory-bounded temporal windows (`cotracker.window_frames`, default 100) instead of Phase 3's naive one-call-per-video reading of the spec.

**`src/hybrid/cotracker_tracker.py`:** `run_cotracker_on_video()` iterates the Phase 1 `VideoReader` in windows. Each window's query points are either (a) the previous window's last tracked positions (continuing the same point identities — the normal case), or (b) freshly reseeded from MediaPipe's ROI/landmarks at the new window's start (a **reinit event**, logged) when the previous window ended with a visible-point ratio below `cotracker.reinit_visibility_threshold`. This directly implements the spec's "MediaPipe may periodically reinitialize CoTracker if tracking quality becomes poor" — the trigger is quality-based, not a hardcoded cadence, and doubles as the window-stitching mechanism (no overlap frames needed). Per-point-per-frame `tracks_x`/`tracks_y`/`visibility` are kept in full (spec: "x trajectory, y trajectory, visibility... for every tracked point over time"), plus `num_visible_per_frame`, `valid_ratio_per_frame`, `reinit_events`, and `track_loss_periods` (longest-contiguous-run detection, same pattern as Phase 2's `_longest_gap`, generalized to a threshold condition). `tracker_motion_y(t)` = **median Y of currently-visible points per frame** (per spec's suggested strategy, robust to outlier tracks vs. a mean) — this is the raw signal Phase 6-9 will filter/analyze for periodicity. Raises `TrackLostError` (logged) only if the longest track-loss period exceeds `max_track_loss_sec` **and** `max_reinits` is already exhausted — a transient dip that a reinit fixes is not a failure.

**Caching:** `run_cotracker_on_video_cached()`, same pattern as Phase 2, keyed on `(video_id, hash(cotracker sub-config + mediapipe sub-config + pinned HandLandmarker sha256))` — deliberately includes the MediaPipe sub-config too, since CoTracker's seed points depend on MediaPipe's output; a MediaPipe config change must invalidate CoTracker's cache even if no CoTracker parameter changed (verified by test).

**Config:** new `cotracker:` section matching the Phase 0.5 skeleton's example almost exactly (`num_points: 40`, `visibility_threshold: 0.6`, `reinit_visibility_threshold: 0.4`) plus the memory/windowing parameters this phase's real-data investigation showed were necessary: `max_reinits: 10`, `max_track_loss_sec: 2.0`, `window_frames: 100`, `working_max_dim: 1024`. All schema-validated (thresholds in [0,1], counts >=1, `max_reinits` >=0 — 0 is legitimate, meaning "never reinitialize," useful for later ablations).

**Tests (`tests/test_cotracker_tracker.py`, 15 tests; `tests/test_config.py` +13 for the new schema fields):** 14 fast tests — resize-for-memory geometry, longest-track-loss-run detection (mirrors Phase 2's gap tests), median-visible-Y aggregation (incl. all-invisible -> NaN), config-hash sensitivity, and 4 orchestration tests that **monkeypatch `_run_window`** (not the real model) to deterministically drive the windowing/reinit/track-loss state machine across multiple windows — verifying reinit-then-recovery, `TrackLostError` when sustained loss exhausts `max_reinits`, `TrackLostError` when a reinit is needed but no MediaPipe detection exists in that window, and immediate failure when MediaPipe never detected anything at all — plus 2 caching-wrapper tests. 1 `@pytest.mark.slow` test loads the real CoTracker checkpoint and runs it on a tiny synthetic clip with a locally-distinctive moving pattern, asserting well-formed output shapes/dtype and that query-frame points are returned unchanged. Full suite: **92/92 passing**, ruff clean.

**Real-data sanity check** (`video3_development.mp4`, the shortest dev video — 17s/509 frames, chosen deliberately to keep this phase's wall-clock reasonable per the spec's own "sanity check on one development video" guidance, not the full set): MediaPipe 100% detection (23.1s). CoTracker: **509/509 frames tracked, 40 points, mean valid ratio 0.998, zero reinit events, zero track loss**, `tracker_motion_y` ranges ~1202-1703px (a clean ~500px oscillation band, no NaNs) — plausible for a repetitive up/down CPR hand motion, though its periodicity isn't analyzed until Phase 8-9's filtering/estimators. Runtime: **349.1s (~5.8 min) cold**, **0.085s cached** (cache hit confirmed for both branches together).

**Decisions:** (1) Single shared venv (no `.venv-torch` isolation) since torch+mediapipe coexist cleanly — revisit only if RepNet (Phase 10) actually conflicts. (2) Memory-bounded windowing was not in the original Phase 3 text but is a necessary consequence of this machine's RAM vs. the offline model's whole-video-in-memory design — documented here rather than silently deviating from the spec. (3) `window_frames=100` / `working_max_dim=1024` are first-pass defensible defaults (chosen from the timing/memory investigation above), not yet tuned — real tuning is Phase 15's job. (4) CPU wall-clock for CoTracker (~6min/500 frames) means the full 6-video dev set will take roughly 30-45 min total whenever it's run end-to-end (Phase 13) — acceptable per CLAUDE.md's "expect this to be slow... build and validate correctness first" environment constraint.

**Next phase:** Phase 4 — RANSAC Affine Ego-Motion Compensation.
