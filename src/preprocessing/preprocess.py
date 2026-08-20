"""
Preprocessing pipeline: temporal trim (manual, per-video) + optional
orientation correction (manual, per-video). No cropping, no resizing,
no FPS resampling. Full frame is preserved.

Reads:
    data/metadata/trim_times.csv
        columns: filename, cpr_start_sec, cpr_end_sec, rotate_degrees, notes
        Leave cpr_start_sec/cpr_end_sec blank to keep the full video
        (no trim). rotate_degrees defaults to 0 (no rotation) and only
        accepts 0/90/180/270.

Writes:
    data/processed/<name>_processed.mp4
    data/metadata/timestamps/<name>_timestamp_map.csv
        per-frame mapping: original_frame_index, original_pts_sec,
        processed_frame_index, processed_pts_sec, offset_sec
    data/metadata/preprocessing_metadata.csv
        one summary row per video

Source videos in data/raw/ are never modified.

Usage:
    python src/preprocessing/preprocess.py
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
TIMESTAMPS_DIR = METADATA_DIR / "timestamps"
TRIM_TIMES_PATH = METADATA_DIR / "trim_times.csv"
OUTPUT_METADATA_PATH = METADATA_DIR / "preprocessing_metadata.csv"

ROTATE_FILTERS = {
    0: None,
    90: "transpose=1",
    180: "transpose=1,transpose=1",
    270: "transpose=2",
}

PTS_MATCH_TOLERANCE_SEC = 1.0  # window padding when re-reading original PTS near a trim point


def ffprobe_json(path: Path) -> dict:
    cmd = ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def parse_frac(s):
    if not s or s == "0/0":
        return None
    n, d = s.split("/")
    n, d = float(n), float(d)
    return n / d if d else None


def video_summary(path: Path) -> dict:
    probe = ffprobe_json(path)
    vstream = next(s for s in probe["streams"] if s["codec_type"] == "video")
    fmt = probe["format"]
    return {
        "width": vstream["width"],
        "height": vstream["height"],
        "avg_fps": parse_frac(vstream.get("avg_frame_rate")),
        "duration_sec": float(fmt.get("duration", 0.0)),
    }


def get_frame_pts(path: Path, start: float = None, duration: float = None) -> list[float]:
    # JSON output (not csv=p=0) because re-encoded streams can attach an empty
    # side_data_list to each frame, which breaks naive csv column parsing.
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0"]
    if start is not None and duration is not None:
        cmd += ["-read_intervals", f"{start}%+{duration}"]
    cmd += ["-show_entries", "frame=pts_time", "-print_format", "json", str(path)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    return sorted(float(fr["pts_time"]) for fr in data.get("frames", []) if fr.get("pts_time") is not None)


def run_ffmpeg_trim(src: Path, dst: Path, start, end, rotate_degrees: int) -> tuple[bool, str]:
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src)]

    if start is not None:
        cmd += ["-ss", str(start)]
    if end is not None:
        cmd += ["-to", str(end)]

    cmd += ["-map", "0:v:0", "-map", "0:a:0?"]

    vf = ROTATE_FILTERS.get(rotate_degrees)
    if vf:
        cmd += ["-vf", vf]

    cmd += [
        "-c:v",
        "libx265",
        "-pix_fmt",
        "yuv420p10le",
        "-crf",
        "18",
        "-preset",
        "medium",
        "-tag:v",
        "hvc1",
        "-vsync",
        "vfr",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stderr


def run_ffmpeg_copy(src: Path, dst: Path) -> tuple[bool, str]:
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(src),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-c",
        "copy",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stderr


def build_timestamp_sidecar(
    video_name: str, original_path: Path, processed_path: Path, requested_start, original_duration: float
) -> tuple[Path, float, str]:
    """Returns (sidecar_path, actual_trim_start_sec, note)."""
    processed_pts = get_frame_pts(processed_path)

    if requested_start is None:
        window_start = 0.0
        window_dur = original_duration
    else:
        window_start = max(0.0, requested_start - PTS_MATCH_TOLERANCE_SEC)
        window_dur = (processed_pts[-1] if processed_pts else 0.0) + 2 * PTS_MATCH_TOLERANCE_SEC

    all_original_pts = get_frame_pts(original_path, start=window_start, duration=window_dur)

    if requested_start is None:
        start_index_in_window = 0
    else:
        start_index_in_window = next((i for i, t in enumerate(all_original_pts) if t >= requested_start - 1e-6), 0)
    original_window_from_start = all_original_pts[start_index_in_window:]

    note = ""
    n = min(len(processed_pts), len(original_window_from_start))
    if len(processed_pts) != len(original_window_from_start):
        note = (
            f"frame count mismatch during timestamp mapping: "
            f"processed={len(processed_pts)} original_window={len(original_window_from_start)}; "
            f"mapped first {n} frames only"
        )

    actual_trim_start = original_window_from_start[0] if original_window_from_start else (requested_start or 0.0)

    sidecar_path = TIMESTAMPS_DIR / f"{video_name}_timestamp_map.csv"
    with open(sidecar_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "processed_frame_index",
                "processed_pts_sec",
                "original_frame_index_in_video",
                "original_pts_sec",
                "offset_sec",
            ]
        )
        for i in range(n):
            proc_t = processed_pts[i]
            orig_t = original_window_from_start[i]
            orig_frame_index = start_index_in_window + i
            writer.writerow([i, f"{proc_t:.6f}", orig_frame_index, f"{orig_t:.6f}", f"{orig_t - proc_t:.6f}"])

    return sidecar_path, actual_trim_start, note


def load_trim_config() -> list[dict]:
    with open(TRIM_TIMES_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_float_or_none(s: str):
    s = (s or "").strip()
    return float(s) if s else None


def parse_rotate(s: str) -> int:
    s = (s or "").strip()
    val = int(s) if s else 0
    if val not in ROTATE_FILTERS:
        raise ValueError(f"rotate_degrees must be one of {sorted(ROTATE_FILTERS)}, got {val}")
    return val


def process_one(row: dict) -> dict:
    filename = row["filename"].strip()
    src = RAW_DIR / filename
    result = {"original_filename": filename, "notes": ""}

    if not src.exists():
        result["notes"] = f"SKIPPED: source file not found at {src}"
        return result

    orig = video_summary(src)
    result.update(
        {
            "original_fps": round(orig["avg_fps"], 4) if orig["avg_fps"] else "",
            "original_width": orig["width"],
            "original_height": orig["height"],
            "original_duration_sec": round(orig["duration_sec"], 4),
        }
    )

    cpr_start = parse_float_or_none(row.get("cpr_start_sec"))
    cpr_end = parse_float_or_none(row.get("cpr_end_sec"))
    try:
        rotate_degrees = parse_rotate(row.get("rotate_degrees"))
    except ValueError as e:
        result["notes"] = f"SKIPPED: {e}"
        return result

    notes = []
    if cpr_start is not None and (cpr_start < 0 or cpr_start >= orig["duration_sec"]):
        notes.append(f"invalid cpr_start_sec={cpr_start} (duration={orig['duration_sec']:.2f}s); ignoring trim")
        cpr_start = None
    if cpr_end is not None and (cpr_end <= (cpr_start or 0) or cpr_end > orig["duration_sec"] + 0.5):
        notes.append(f"invalid cpr_end_sec={cpr_end}; ignoring trim end")
        cpr_end = None

    trimmed = cpr_start is not None or cpr_end is not None
    rotated = rotate_degrees != 0

    stem = Path(filename).stem
    dst = PROCESSED_DIR / f"{stem}_processed.mp4"

    if not trimmed and not rotated:
        ok, err = run_ffmpeg_copy(src, dst)
        notes.append("stream copy (no trim, no rotation requested)")
    else:
        ok, err = run_ffmpeg_trim(src, dst, cpr_start, cpr_end, rotate_degrees)
        ops = []
        if trimmed:
            ops.append(
                f"trimmed [{cpr_start if cpr_start is not None else 0:.3f}s, "
                f"{cpr_end if cpr_end is not None else orig['duration_sec']:.3f}s]"
            )
        if rotated:
            ops.append(f"rotated {rotate_degrees} deg")
        notes.append("re-encoded (" + "; ".join(ops) + ")")

    if not ok:
        result["notes"] = "FFMPEG FAILED: " + err.strip()[-500:]
        return result

    proc = video_summary(dst)
    result.update(
        {
            "processed_filename": dst.name,
            "processed_fps": round(proc["avg_fps"], 4) if proc["avg_fps"] else "",
            "processed_width": proc["width"],
            "processed_height": proc["height"],
            "processed_duration_sec": round(proc["duration_sec"], 4),
            "cpr_start_timestamp_sec": cpr_start if cpr_start is not None else "",
            "cpr_end_timestamp_sec": cpr_end if cpr_end is not None else "",
            "rotation_applied_deg": rotate_degrees,
            "resizing_applied": False,
        }
    )

    try:
        sidecar_path, actual_trim_start, sidecar_note = build_timestamp_sidecar(
            stem, src, dst, cpr_start, orig["duration_sec"]
        )
        result["timestamp_sidecar"] = str(sidecar_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        result["actual_trim_start_sec"] = round(actual_trim_start, 6)
        if sidecar_note:
            notes.append(sidecar_note)
    except subprocess.CalledProcessError as e:
        notes.append(f"timestamp sidecar generation failed: {e.stderr.strip()[-300:]}")

    result["notes"] = "; ".join(notes)
    return result


def main() -> None:
    if not TRIM_TIMES_PATH.exists():
        print(f"Trim config not found: {TRIM_TIMES_PATH}", file=sys.stderr)
        sys.exit(1)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TIMESTAMPS_DIR.mkdir(parents=True, exist_ok=True)

    rows_in = load_trim_config()
    results = []
    for row in rows_in:
        print(f"Processing {row['filename']} ...")
        res = process_one(row)
        results.append(res)
        print(f"  -> {res.get('notes', '')}")

    fieldnames = [
        "original_filename",
        "processed_filename",
        "original_fps",
        "processed_fps",
        "original_width",
        "original_height",
        "processed_width",
        "processed_height",
        "original_duration_sec",
        "processed_duration_sec",
        "cpr_start_timestamp_sec",
        "cpr_end_timestamp_sec",
        "actual_trim_start_sec",
        "rotation_applied_deg",
        "resizing_applied",
        "timestamp_sidecar",
        "notes",
    ]
    with open(OUTPUT_METADATA_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            writer.writerow({k: res.get(k, "") for k in fieldnames})

    print(f"\nWrote {OUTPUT_METADATA_PATH}")
    failures = [
        r["original_filename"] for r in results if "FAILED" in r.get("notes", "") or "SKIPPED" in r.get("notes", "")
    ]
    if failures:
        print(f"Videos with issues: {failures}")
    else:
        print("All videos processed successfully.")


if __name__ == "__main__":
    main()
