"""Small shared signal-analysis helpers used across branch modules."""

from __future__ import annotations

import numpy as np


def longest_bad_run(is_bad: np.ndarray, timestamps: np.ndarray) -> tuple[int, float, list[tuple[int, int]]]:
    """Longest contiguous run where `is_bad` is True, plus every such run's
    (start_idx, end_idx) inclusive into the same axis as `is_bad`/`timestamps`.

    Used for detection-gap / track-loss / unreliable-transform diagnostics
    (Phase 2 hand-detection gaps, Phase 3 track-loss periods, Phase 4
    unreliable ego-motion spans, ...).
    """
    periods: list[tuple[int, int]] = []
    longest_frames = 0
    longest_sec = 0.0
    run_start: int | None = None
    for i, bad in enumerate(is_bad):
        if bad:
            if run_start is None:
                run_start = i
            continue
        if run_start is not None:
            periods.append((run_start, i - 1))
            run_len = i - run_start
            run_sec = float(timestamps[i - 1] - timestamps[run_start])
            if run_len > longest_frames:
                longest_frames, longest_sec = run_len, run_sec
            run_start = None
    if run_start is not None:
        periods.append((run_start, len(is_bad) - 1))
        run_len = len(is_bad) - run_start
        run_sec = float(timestamps[-1] - timestamps[run_start])
        if run_len > longest_frames:
            longest_frames, longest_sec = run_len, run_sec
    return longest_frames, longest_sec, periods
