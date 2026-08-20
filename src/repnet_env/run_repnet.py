"""RepNet subprocess entrypoint (Phase 10).

Invoked by `hybrid.repnet_branch` (main venv) via subprocess, running under
the isolated `.venv-tf` interpreter -- kept dependency-free of the main
`hybrid` package (mediapipe/torch/pydantic aren't installed here).

Usage:
    <venv-tf-python> run_repnet.py --video PATH --checkpoint-dir DIR
        --output-json PATH [--threshold F] [--within-period-threshold F]
        [--strides "1,2,3,4"] [--batch-size N] [--constant-speed]
        [--median-filter] [--fully-periodic]

Writes a single JSON object to --output-json:
    {"cpm": float | null, "confidence": float, "pred_period_frames": float,
     "chosen_stride": int, "fps": float, "num_frames": int,
     "reason": str | null}
`cpm`/`reason` follow CLAUDE.md's "never nudge toward a known count" rule --
this script only ever reports what RepNet itself predicted; a null `cpm`
with `confidence: 0.0` means RepNet found no clear periodicity, which is a
valid low-confidence result for Phase 11/12 fusion to downweight, not a
crash. A non-zero exit code means an actual infrastructure failure (bad
checkpoint, unreadable video, ...), for the caller to raise
RepNetUnavailableError on.
"""

import argparse
import json
import sys

import numpy as np
import tensorflow.compat.v2 as tf
from checkpoint_mapping import MAPPING_NEW_TO_OLD_LAYER_NAMES, load_ckpt_with_custom_layer_mapping
from model import get_counts, get_repnet_model, read_video


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--within-period-threshold", type=float, default=0.5)
    parser.add_argument("--strides", default="1,2,3,4")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--constant-speed", action="store_true")
    parser.add_argument("--median-filter", action="store_true")
    parser.add_argument("--fully-periodic", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    strides = [int(s) for s in args.strides.split(",")]

    frames, fps = read_video(args.video)
    if len(frames) == 0:
        raise RuntimeError(f"read_video returned 0 frames for {args.video}")
    if not fps or fps <= 0:
        raise RuntimeError(f"read_video returned an invalid fps ({fps}) for {args.video}")

    model = get_repnet_model(args.checkpoint_dir)
    load_ckpt_with_custom_layer_mapping(model, args.checkpoint_dir, MAPPING_NEW_TO_OLD_LAYER_NAMES)

    frames_tensor = tf.convert_to_tensor(frames)
    pred_period, pred_score, _within_period, _per_frame_counts, chosen_stride = get_counts(
        model,
        frames_tensor,
        strides=strides,
        batch_size=args.batch_size,
        threshold=args.threshold,
        within_period_threshold=args.within_period_threshold,
        constant_speed=args.constant_speed,
        median_filter=args.median_filter,
        fully_periodic=args.fully_periodic,
    )

    pred_score = float(pred_score)
    pred_period = float(pred_period)

    cpm = None
    reason = None
    if pred_score < args.threshold:
        reason = f"pred_score ({pred_score:.3f}) below threshold ({args.threshold})"
    elif not np.isfinite(pred_period) or pred_period <= 0:
        reason = f"non-finite or non-positive pred_period ({pred_period})"
        pred_score = 0.0
    else:
        cpm = 60.0 * fps / pred_period

    result = {
        "cpm": cpm,
        "confidence": max(0.0, min(1.0, pred_score)),
        "pred_period_frames": pred_period,
        "chosen_stride": int(chosen_stride),
        "fps": float(fps),
        "num_frames": len(frames),
        "reason": reason,
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001 -- surface any failure with a clear message to the caller's stderr
        print(f"run_repnet.py failed: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
