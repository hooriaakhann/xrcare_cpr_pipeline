import subprocess

import cv2
import numpy as np
import pytest

from hybrid.exceptions import TimestampError, VideoReadError
from hybrid.video_io import VideoReader, probe_video_meta


def _write_synthetic_video(path, num_frames=8, fps=10.0, width=64, height=48):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), i * 10 % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()


@pytest.fixture
def synthetic_video(tmp_path):
    path = tmp_path / "synthetic.mp4"
    _write_synthetic_video(path, num_frames=8, fps=10.0)
    return path


def test_reads_all_frames_with_paired_index_and_timestamp(synthetic_video):
    with VideoReader(synthetic_video) as reader:
        frames = list(reader)

    assert len(frames) == 8
    for expected_index, frame in enumerate(frames):
        assert frame.index == expected_index
        assert frame.image.shape == (48, 64, 3)
    timestamps = [f.timestamp_sec for f in frames]
    assert timestamps == sorted(timestamps)
    assert timestamps[0] >= 0.0


def test_frame_count_and_duration_properties(synthetic_video):
    with VideoReader(synthetic_video) as reader:
        assert reader.frame_count == 8
        frames = list(reader)
    assert reader.duration_sec == frames[-1].timestamp_sec


def test_probe_video_meta_matches_reader_frame_count(synthetic_video):
    meta = probe_video_meta(synthetic_video)
    with VideoReader(synthetic_video) as reader:
        assert meta.frame_count == reader.frame_count
    assert meta.width == 64
    assert meta.height == 48


def test_missing_file_raises_video_read_error(tmp_path):
    with pytest.raises(VideoReadError):
        VideoReader(tmp_path / "does_not_exist.mp4")


def test_probe_missing_file_raises_video_read_error(tmp_path):
    with pytest.raises(VideoReadError):
        probe_video_meta(tmp_path / "does_not_exist.mp4")


def test_single_pass_reiteration_raises(synthetic_video):
    with VideoReader(synthetic_video) as reader:
        list(reader.frames())
        with pytest.raises(VideoReadError):
            list(reader.frames())


def test_corrupt_video_raises_video_read_error(tmp_path):
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"not a real video file")
    with pytest.raises((VideoReadError, TimestampError)):
        VideoReader(corrupt)


def test_ffprobe_not_on_path_raises_timestamp_error(synthetic_video, monkeypatch):
    def fake_run(*args, **kwargs):
        raise FileNotFoundError("ffprobe not found")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(TimestampError):
        VideoReader(synthetic_video)


def test_ffprobe_malformed_json_raises_timestamp_error(synthetic_video, monkeypatch):
    class FakeResult:
        stdout = "not json"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeResult())
    with pytest.raises(TimestampError):
        VideoReader(synthetic_video)


def test_ffprobe_no_timestamps_raises_timestamp_error(synthetic_video, monkeypatch):
    class FakeResult:
        stdout = '{"frames": []}'

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeResult())
    with pytest.raises(TimestampError):
        VideoReader(synthetic_video)


def test_ffprobe_non_monotonic_timestamps_raises_timestamp_error(synthetic_video, monkeypatch):
    class FakeResult:
        stdout = '{"frames": [{"pts_time": "1.0"}, {"pts_time": "0.5"}]}'

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeResult())
    with pytest.raises(TimestampError):
        VideoReader(synthetic_video)
