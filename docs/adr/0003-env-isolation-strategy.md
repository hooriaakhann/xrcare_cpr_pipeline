# ADR 0003: virtualenv + subprocess for RepNet, not Docker; not needed at all for CoTracker

**Status:** Accepted (Phase 0.5 default; confirmed in Phase 3 for CoTracker, applied in
Phase 10 for RepNet)

**Decision:** RepNet (TensorFlow) runs in its own isolated virtualenv (`.venv-tf/`),
invoked from the main pipeline via `subprocess.run` (`src/hybrid/repnet_branch.py` ->
`src/repnet_env/run_repnet.py`) rather than merged into the main environment or
containerized with Docker. CoTracker (PyTorch/`torch`), by contrast, was **not** given its
own environment — it was installed directly into the main venv alongside MediaPipe.

**Why isolate RepNet but not CoTracker:** Phase 0.5 set virtualenv+subprocess as the default
"unless there's time for Docker later," but left the actual need for isolation an open
question per branch. Phase 3 tested installing `torch` (CPU wheel) straight into the main
venv and ran `pip check` — clean, no conflicts with `mediapipe`/`opencv-contrib-python`/
`numpy`. There was no reason to pay subprocess-boundary complexity (serialization, process
startup, error-passing across a process boundary) for a dependency that coexists fine.
RepNet was different: its reference implementation only exists as a Colab notebook and pulls
in a full TensorFlow install (~2GB+ of dependencies) plus a checkpoint-loading path
(`tf.train.Checkpoint`, `py_checkpoint_reader`) that was never checked against `mediapipe`'s
bundled TFLite runtime for conflicts — vendoring it into the main venv risked finding out
mid-pipeline that TF and TFLite collide, with a much larger, harder-to-diagnose blast radius
than CoTracker's clean `pip check`. Isolation sidesteps the question entirely rather than
requiring an answer.

**Why subprocess+venv over Docker:** Docker adds real value mainly for deployment
portability and stronger isolation guarantees; neither was the priority here — this is a
development/tuning pipeline run locally, not a deployed service. A second virtualenv plus a
`subprocess.run` call is far less setup cost (no Docker Desktop / WSL2 backend dependency on
this Windows machine) for the same practical outcome: RepNet's dependencies can never
collide with the main environment's.

**Consequence:** every RepNet inference call pays a subprocess-startup and TensorFlow
import cost (observed ~30s once the checkpoint is on disk, Phase 10 PROGRESS.md) that an
in-process call wouldn't have; acceptable since RepNet is already cached per-video like every
other expensive branch, so this cost is paid once per video, not per pipeline run.
