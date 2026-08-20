"""
Inspect all videos in data/raw/ and report FPS, resolution, frame count,
duration, codec, and frame-timing consistency (CFR vs VFR).

Does not modify or move any files. Writes a report to
data/metadata/video_inspection_report.csv.

Usage:
    python src/preprocessing/inspect_videos.py
"""

import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
REPORT_PATH = METADATA_DIR / "video_inspection_report.csv"

VFR_STD_THRESHOLD_MS = 1.0  # stdev of frame intervals above this => VFR


def run_ffprobe_json(path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def get_frame_pts(path: Path) -> list[float]:
    # JSON output (not csv=p=0) because streams can attach an empty
    # side_data_list to each frame, which breaks naive csv column parsing.
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=pts_time",
        "-print_format",
        "json",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    return sorted(float(fr["pts_time"]) for fr in data.get("frames", []) if fr.get("pts_time") is not None)


def parse_frac(s: str | None) -> float | None:
    if not s or s == "0/0":
        return None
    n, d = s.split("/")
    n, d = float(n), float(d)
    return n / d if d else None


def inspect_one(path: Path) -> dict:
    row: dict = {"filename": path.name, "decode_ok": False, "error": ""}
    try:
        probe = run_ffprobe_json(path)
    except subprocess.CalledProcessError as e:
        row["error"] = f"ffprobe failed: {e.stderr.strip()}"
        return row

    vstream = next((s for s in probe.get("streams", []) if s.get("codec_type") == "video"), None)
    astreams = [s for s in probe.get("streams", []) if s.get("codec_type") == "audio"]
    fmt = probe.get("format", {})

    if vstream is None:
        row["error"] = "no video stream found"
        return row

    row.update(
        {
            "width": vstream.get("width"),
            "height": vstream.get("height"),
            "codec": vstream.get("codec_name"),
            "pix_fmt": vstream.get("pix_fmt"),
            "container": fmt.get("format_name"),
            "declared_fps_tbr": parse_frac(vstream.get("r_frame_rate")),
            "avg_fps": parse_frac(vstream.get("avg_frame_rate")),
            "nb_frames_container": vstream.get("nb_frames"),
            "duration_sec": float(fmt.get("duration", 0.0)),
            "has_audio": len(astreams) > 0,
            "audio_codec": astreams[0].get("codec_name") if astreams else "",
            "rotate_tag": (vstream.get("tags") or {}).get("rotate"),
            "rotation_side_data": next(
                (sd.get("rotation") for sd in vstream.get("side_data_list", []) if "rotation" in sd),
                None,
            ),
        }
    )

    try:
        pts = get_frame_pts(path)
    except subprocess.CalledProcessError as e:
        row["error"] = f"frame decode failed: {e.stderr.strip()}"
        return row

    row["nb_frames_decoded"] = len(pts)
    row["decode_ok"] = len(pts) > 0 and (
        row["nb_frames_container"] is None or int(row["nb_frames_container"]) == len(pts)
    )

    if len(pts) > 1:
        dts_ms = [(pts[i + 1] - pts[i]) * 1000 for i in range(len(pts) - 1)]
        row["mean_frame_interval_ms"] = statistics.mean(dts_ms)
        row["stdev_frame_interval_ms"] = statistics.pstdev(dts_ms)
        row["min_frame_interval_ms"] = min(dts_ms)
        row["max_frame_interval_ms"] = max(dts_ms)
        row["timing"] = "VFR" if row["stdev_frame_interval_ms"] > VFR_STD_THRESHOLD_MS else "CFR"
    else:
        row["timing"] = "unknown"

    return row


def main() -> None:
    if not RAW_DIR.exists():
        print(f"Raw video directory not found: {RAW_DIR}", file=sys.stderr)
        sys.exit(1)

    videos = sorted(RAW_DIR.glob("*.mp4"))
    if not videos:
        print(f"No .mp4 files found in {RAW_DIR}", file=sys.stderr)
        sys.exit(1)

    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for video_path in videos:
        print(f"Inspecting {video_path.name} ...")
        rows.append(inspect_one(video_path))

    fieldnames = [
        "filename",
        "decode_ok",
        "error",
        "width",
        "height",
        "codec",
        "pix_fmt",
        "container",
        "declared_fps_tbr",
        "avg_fps",
        "timing",
        "mean_frame_interval_ms",
        "stdev_frame_interval_ms",
        "min_frame_interval_ms",
        "max_frame_interval_ms",
        "nb_frames_container",
        "nb_frames_decoded",
        "duration_sec",
        "has_audio",
        "audio_codec",
        "rotate_tag",
        "rotation_side_data",
    ]
    with open(REPORT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"\nWrote {REPORT_PATH}")

    fps_rounded = {round(r["avg_fps"], 2) if r.get("avg_fps") else None for r in rows}
    timings = {r.get("timing") for r in rows}
    rotations = {r.get("rotate_tag") for r in rows} | {r.get("rotation_side_data") for r in rows}
    failed = [r["filename"] for r in rows if not r.get("decode_ok")]

    print("\n--- Consistency summary ---")
    print(f"Unique resolutions (W x H combos): {sorted((r.get('width'), r.get('height')) for r in rows)}")
    print(f"Unique avg FPS (rounded): {fps_rounded}")
    print(f"Frame timing types present: {timings}")
    print(f"Rotation tags/side_data present: {rotations}")
    if failed:
        print(f"FILES WITH DECODE ISSUES: {failed}")
    else:
        print("All files decoded successfully.")


if __name__ == "__main__":
    main()
