"""CoTracker Multi-Point Tracking (Phase 3).

Tracks a set of MediaPipe-seeded points (hand + wrist + forearm) across a
whole development video using the pretrained CoTracker3 "offline" model
(torch.hub, never fine-tuned -- CLAUDE.md rule 4). The offline model holds
its entire input video tensor in memory at once; at native resolution a
600-frame video would need ~20GB, more than this machine has (16GB, no GPU).
So videos are processed in memory-bounded temporal windows instead of one
call over the full video.

At each window boundary, if the previous window ended with a degraded
visible-point ratio (below `reinit_visibility_threshold`), MediaPipe
re-derives a fresh point set for the next window -- a "reinit event". If
tracking is still healthy, the next window continues the same point
identities from their last known positions. This is the "MediaPipe may
periodically reinitialize CoTracker" behavior from the Phase 3 spec, and
also doubles as the mechanism that stitches windows back into one continuous
per-point trajectory.
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass

import cv2
import numpy as np
import torch

from hybrid.caching import CacheManager
from hybrid.config import CoTrackerConfig, HybridConfig
from hybrid.exceptions import TrackLostError
from hybrid.logging_config import get_logger
from hybrid.mediapipe_roi import MediaPipeVideoResult, get_cotracker_init_points
from hybrid.models import HAND_LANDMARKER
from hybrid.signal_utils import longest_bad_run
from hybrid.video_io import VideoReader

logger = get_logger(__name__)

_MODEL_CACHE: dict[str, torch.nn.Module] = {}

# Pinned via torch.hub (facebookresearch/co-tracker); not a project-local
# checkpoint file like MediaPipe's, so recorded here for the run
# summary/ledger rather than in hybrid.models.
COTRACKER3_OFFLINE_SOURCE = "facebookresearch/co-tracker (torch.hub, cotracker3_offline, scaled_offline.pth)"


def _load_cotracker_model() -> torch.nn.Module:
    if "offline" not in _MODEL_CACHE:
        logger.info("Loading pretrained CoTracker3 offline model via torch.hub (cached locally after first download)")
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
        model.eval()
        _MODEL_CACHE["offline"] = model
    return _MODEL_CACHE["offline"]


@dataclass
class CoTrackerVideoResult:
    video_id: str
    num_points: int
    frame_indices: np.ndarray  # (T,) int
    timestamps_sec: np.ndarray  # (T,) float
    tracks_x: np.ndarray  # (T, N) float, original-resolution pixel coords
    tracks_y: np.ndarray  # (T, N) float
    visibility: np.ndarray  # (T, N) bool
    num_visible_per_frame: np.ndarray  # (T,) int
    valid_ratio_per_frame: np.ndarray  # (T,) float
    reinit_events: list[tuple[int, float, str]]  # (frame_index, timestamp_sec, reason)
    track_loss_periods: list[tuple[int, int]]  # (start_idx, end_idx) into the T axis, inclusive
    longest_track_loss_sec: float
    tracker_motion_y: np.ndarray  # (T,) median Y of currently-visible points; NaN if none visible


def _resize_for_cotracker(frames: list[np.ndarray], max_dim: int) -> tuple[np.ndarray, float, float]:
    h, w = frames[0].shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = [cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_AREA) for f in frames]
    return np.stack(resized), new_w / w, new_h / h


def _frames_to_tensor(frames_bgr: np.ndarray) -> torch.Tensor:
    rgb = frames_bgr[..., ::-1].copy()
    return torch.from_numpy(rgb).permute(0, 3, 1, 2)[None].float()


def _run_window(
    model,
    frames_bgr: list[np.ndarray],
    query_frame_local: int,
    query_points_xy: np.ndarray,
    ct_config: CoTrackerConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Runs CoTracker on one window. Returns (tracks_xy, visibility) at
    ORIGINAL resolution: shapes (T_window, N, 2) float and (T_window, N) bool.
    """
    resized, scale_x, scale_y = _resize_for_cotracker(frames_bgr, ct_config.working_max_dim)
    video = _frames_to_tensor(resized)

    scaled_points = query_points_xy.copy()
    scaled_points[:, 0] *= scale_x
    scaled_points[:, 1] *= scale_y

    n = scaled_points.shape[0]
    queries = torch.zeros(1, n, 3, dtype=torch.float32)
    queries[0, :, 0] = float(query_frame_local)
    queries[0, :, 1] = torch.from_numpy(scaled_points[:, 0])
    queries[0, :, 2] = torch.from_numpy(scaled_points[:, 1])

    with torch.no_grad():
        tracks, visibility = model(video, queries=queries)

    tracks_np = tracks[0].numpy().copy()  # (T, N, 2)
    visibility_np = visibility[0].numpy().copy()  # (T, N) bool

    tracks_np[:, :, 0] /= scale_x
    tracks_np[:, :, 1] /= scale_y
    return tracks_np, visibility_np


def _longest_run_below_threshold(
    valid_ratio: np.ndarray, timestamps: np.ndarray, threshold: float
) -> tuple[int, float, list[tuple[int, int]]]:
    """Longest contiguous run where `valid_ratio < threshold`. Thin wrapper
    around the shared `longest_bad_run` helper (also used by Phase 4).
    """
    return longest_bad_run(valid_ratio < threshold, timestamps)


def _median_visible_y(tracks_y: np.ndarray, visibility: np.ndarray) -> np.ndarray:
    """Median Y pixel coordinate of currently-visible points per frame --
    the raw tracker motion waveform (robust to outlier points vs. a mean).
    NaN for a frame with zero visible points.
    """
    t_len = tracks_y.shape[0]
    motion = np.full(t_len, np.nan)
    for t in range(t_len):
        visible_y = tracks_y[t, visibility[t]]
        if visible_y.size > 0:
            motion[t] = float(np.median(visible_y))
    return motion


def run_cotracker_on_video(
    video_path, mediapipe_result: MediaPipeVideoResult, config: HybridConfig, video_id: str
) -> CoTrackerVideoResult:
    ct_config = config.cotracker
    model = _load_cotracker_model()
    detections_by_index = {d.frame_index: d for d in mediapipe_result.detections if d.detected}

    if not detections_by_index:
        raise TrackLostError(f"{video_id}: cannot seed CoTracker -- MediaPipe never detected a hand")

    reinit_events: list[tuple[int, float, str]] = []
    reinits_used = 0
    current_points: np.ndarray | None = None
    window_number = 0

    out_frame_indices: list[int] = []
    out_timestamps: list[float] = []
    out_tracks_x: list[np.ndarray] = []
    out_tracks_y: list[np.ndarray] = []
    out_visibility: list[np.ndarray] = []

    with VideoReader(video_path) as reader:
        frame_iter = iter(reader)
        while True:
            window_frames = list(itertools.islice(frame_iter, ct_config.window_frames))
            if not window_frames:
                break

            window_images = [f.image for f in window_frames]
            window_indices = [f.index for f in window_frames]
            window_timestamps = [f.timestamp_sec for f in window_frames]

            if current_points is None:
                seed_detection = next(
                    (detections_by_index[i] for i in window_indices if i in detections_by_index), None
                )
                if seed_detection is None:
                    raise TrackLostError(
                        f"{video_id}: no MediaPipe detection available to (re)seed CoTracker in window "
                        f"starting at frame {window_indices[0]}"
                    )
                query_points = get_cotracker_init_points(seed_detection, ct_config.num_points)
                query_frame_local = window_indices.index(seed_detection.frame_index)
                if window_number > 0:
                    reinit_events.append(
                        (seed_detection.frame_index, seed_detection.timestamp_sec, "visibility_below_threshold")
                    )
                    reinits_used += 1
                    logger.info(
                        "%s: reinitializing CoTracker at frame %d (t=%.2fs) -- reinit %d/%d",
                        video_id,
                        seed_detection.frame_index,
                        seed_detection.timestamp_sec,
                        reinits_used,
                        ct_config.max_reinits,
                    )
            else:
                query_points = current_points
                query_frame_local = 0

            tracks_xy, visibility = _run_window(model, window_images, query_frame_local, query_points, ct_config)

            out_frame_indices.extend(window_indices)
            out_timestamps.extend(window_timestamps)
            out_tracks_x.append(tracks_xy[:, :, 0])
            out_tracks_y.append(tracks_xy[:, :, 1])
            out_visibility.append(visibility)

            last_visible_ratio = float(visibility[-1].mean())
            if last_visible_ratio < ct_config.reinit_visibility_threshold and reinits_used < ct_config.max_reinits:
                current_points = None
            else:
                current_points = tracks_xy[-1]

            window_number += 1

    tracks_x = np.concatenate(out_tracks_x, axis=0)
    tracks_y = np.concatenate(out_tracks_y, axis=0)
    visibility_arr = np.concatenate(out_visibility, axis=0)
    frame_indices = np.array(out_frame_indices)
    timestamps = np.array(out_timestamps)

    num_visible_per_frame = visibility_arr.sum(axis=1)
    valid_ratio_per_frame = num_visible_per_frame / visibility_arr.shape[1]

    longest_loss_frames, longest_loss_sec, track_loss_periods = _longest_run_below_threshold(
        valid_ratio_per_frame, timestamps, ct_config.visibility_threshold
    )

    if longest_loss_sec > 0:
        logger.info(
            "%s: longest track-loss period is %.2fs (%d frames) below visibility_threshold=%.2f",
            video_id,
            longest_loss_sec,
            longest_loss_frames,
            ct_config.visibility_threshold,
        )
    if longest_loss_sec > ct_config.max_track_loss_sec and reinits_used >= ct_config.max_reinits:
        raise TrackLostError(
            f"{video_id}: longest track-loss period ({longest_loss_sec:.2f}s) exceeds "
            f"max_track_loss_sec={ct_config.max_track_loss_sec}s, and max_reinits "
            f"({ct_config.max_reinits}) already exhausted"
        )

    tracker_motion_y = _median_visible_y(tracks_y, visibility_arr)

    logger.info(
        "%s: CoTracker tracked %d points across %d frames in %d window(s), %d reinit event(s), mean valid ratio %.2f",
        video_id,
        ct_config.num_points,
        len(frame_indices),
        window_number,
        len(reinit_events),
        float(valid_ratio_per_frame.mean()),
    )

    return CoTrackerVideoResult(
        video_id=video_id,
        num_points=ct_config.num_points,
        frame_indices=frame_indices,
        timestamps_sec=timestamps,
        tracks_x=tracks_x,
        tracks_y=tracks_y,
        visibility=visibility_arr,
        num_visible_per_frame=num_visible_per_frame,
        valid_ratio_per_frame=valid_ratio_per_frame,
        reinit_events=reinit_events,
        track_loss_periods=track_loss_periods,
        longest_track_loss_sec=longest_loss_sec,
        tracker_motion_y=tracker_motion_y,
    )


def _cotracker_config_hash(config: HybridConfig) -> str:
    """Scoped to cotracker + mediapipe sub-configs (+ the pinned HandLandmarker
    checksum) since CoTracker's query points are seeded from MediaPipe's
    output -- a MediaPipe config change must also invalidate this cache.
    """
    payload = config.cotracker.model_dump_json() + config.mediapipe.model_dump_json() + HAND_LANDMARKER.sha256
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def run_cotracker_on_video_cached(
    video_path,
    mediapipe_result: MediaPipeVideoResult,
    config: HybridConfig,
    video_id: str,
    cache_manager: CacheManager,
) -> CoTrackerVideoResult:
    config_hash = _cotracker_config_hash(config)
    key = cache_manager.make_key("cotracker", video_id, config_hash)

    if config.caching.enabled and cache_manager.exists(key):
        logger.info("%s: loaded CoTracker result from cache (key=%s)", video_id, key)
        return cache_manager.load(key)

    result = run_cotracker_on_video(video_path, mediapipe_result, config, video_id)
    if config.caching.enabled:
        cache_manager.save(key, result)
    return result
