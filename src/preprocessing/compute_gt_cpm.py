"""
Compute clip-level ground-truth CPM (compressions per minute) from the
manually marked CPR-active window in data/metadata/trim_times.csv.

This is a COUNT-BASED ACTIVE-SEGMENT ground truth, not an event-by-event
median-interval ground truth: it assumes exactly 30 chest compressions occur
between cpr_start_sec (start of the 1st compression) and cpr_end_sec (end of
the 30th compression), and derives:

    gt_cpm = (known_compression_count * 60) / (cpr_end_sec - cpr_start_sec)

No per-compression annotation, MediaPipe, CoTracker, RepNet, optical flow, or
hybrid algorithm is used here. Raw videos are not modified.

Usage:
    python src/preprocessing/compute_gt_cpm.py
"""

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
TRIM_TIMES_PATH = METADATA_DIR / "trim_times.csv"
INSPECTION_REPORT_PATH = METADATA_DIR / "video_inspection_report.csv"
OUTPUT_PATH = METADATA_DIR / "ground_truth_summary.csv"

KNOWN_COMPRESSION_COUNT = 30
DURATION_TOLERANCE_SEC = 0.05  # allow small slack for timestamp rounding


def load_durations(path: Path) -> dict[str, float]:
    durations: dict[str, float] = {}
    if not path.exists():
        return durations
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            filename = row.get("filename", "").strip()
            duration_str = row.get("duration_sec", "").strip()
            if filename and duration_str:
                try:
                    durations[filename] = float(duration_str)
                except ValueError:
                    pass
    return durations


def validate(
    filename: str, start_str: str, end_str: str, durations: dict[str, float]
) -> tuple[float | None, float | None, str]:
    """Return (start, end, error_message). error_message is "" if valid."""
    start_str = (start_str or "").strip()
    end_str = (end_str or "").strip()

    if not start_str or not end_str:
        return None, None, "missing cpr_start_sec or cpr_end_sec"

    try:
        start = float(start_str)
        end = float(end_str)
    except ValueError:
        return None, None, "non-numeric cpr_start_sec or cpr_end_sec"

    if start < 0:
        return None, None, "cpr_start_sec is negative"

    if end <= start:
        return None, None, "cpr_end_sec is not greater than cpr_start_sec"

    duration = durations.get(filename)
    if duration is not None and end > duration + DURATION_TOLERANCE_SEC:
        return None, None, f"cpr_end_sec ({end}) exceeds original video duration ({duration})"

    return start, end, ""


def main() -> None:
    if not TRIM_TIMES_PATH.exists():
        print(f"Trim times file not found: {TRIM_TIMES_PATH}", file=sys.stderr)
        sys.exit(1)

    durations = load_durations(INSPECTION_REPORT_PATH)
    if not durations:
        print(
            f"Warning: no durations loaded from {INSPECTION_REPORT_PATH}; skipping the end-of-video validation check.",
            file=sys.stderr,
        )

    with open(TRIM_TIMES_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    out_rows = []
    valid_count = 0
    for row in rows:
        filename = row.get("filename", "").strip()
        start, end, error = validate(filename, row.get("cpr_start_sec", ""), row.get("cpr_end_sec", ""), durations)

        if error:
            out_rows.append(
                {
                    "filename": filename,
                    "cpr_start_sec": row.get("cpr_start_sec", ""),
                    "cpr_end_sec": row.get("cpr_end_sec", ""),
                    "cpr_duration_sec": "",
                    "known_compression_count": "",
                    "gt_cpm": "",
                    "notes": f"invalid: {error}",
                }
            )
            continue

        valid_count += 1
        cpr_duration_sec = end - start
        gt_cpm = (KNOWN_COMPRESSION_COUNT * 60) / cpr_duration_sec
        out_rows.append(
            {
                "filename": filename,
                "cpr_start_sec": round(start, 4),
                "cpr_end_sec": round(end, 4),
                "cpr_duration_sec": round(cpr_duration_sec, 4),
                "known_compression_count": KNOWN_COMPRESSION_COUNT,
                "gt_cpm": round(gt_cpm, 4),
                "notes": "",
            }
        )

    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "filename",
        "cpr_start_sec",
        "cpr_end_sec",
        "cpr_duration_sec",
        "known_compression_count",
        "gt_cpm",
        "notes",
    ]
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote {OUTPUT_PATH}")
    print(f"{valid_count}/{len(rows)} videos have valid cpr_start_sec/cpr_end_sec")
    invalid = [r["filename"] for r in out_rows if r["notes"]]
    if invalid:
        print(f"Invalid (gt_cpm left blank): {invalid}")


if __name__ == "__main__":
    main()
