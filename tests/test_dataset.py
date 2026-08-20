import pytest

from hybrid.config import HybridConfig
from hybrid.dataset import (
    discover_development_videos,
    load_ground_truth,
    map_split_filename_to_gt_key,
)
from hybrid.exceptions import GroundTruthMappingError, VideoReadError

GT_CSV_HEADER = "filename,cpr_start_sec,cpr_end_sec,cpr_duration_sec,known_compression_count,gt_cpm,notes\n"


def _write_gt_csv(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write(GT_CSV_HEADER)
        for row in rows:
            f.write(row + "\n")


@pytest.mark.parametrize(
    ("split_filename", "expected"),
    [
        ("video4_development.mp4", "video4.mp4"),
        ("video10_development.mp4", "video10.mp4"),
        ("video7_test.mp4", "video7.mp4"),
        ("personA_session2_development.mp4", "personA_session2.mp4"),
    ],
)
def test_map_split_filename_to_gt_key(split_filename, expected):
    assert map_split_filename_to_gt_key(split_filename) == expected


@pytest.mark.parametrize(
    "bad_filename",
    ["video4.mp4", "video4_dev.mp4", "development.mp4"],
)
def test_map_split_filename_unrecognized_pattern_raises(bad_filename):
    with pytest.raises(GroundTruthMappingError):
        map_split_filename_to_gt_key(bad_filename)


def test_load_ground_truth_parses_rows(tmp_path):
    csv_path = tmp_path / "gt.csv"
    _write_gt_csv(csv_path, ["video1.mp4,4.0,24.0,20.0,30,90.0,"])
    rows = load_ground_truth(csv_path)
    assert set(rows) == {"video1.mp4"}
    gt = rows["video1.mp4"]
    assert gt.gt_cpm == pytest.approx(90.0)
    assert gt.known_compression_count == 30


def test_load_ground_truth_missing_file_raises(tmp_path):
    with pytest.raises(GroundTruthMappingError):
        load_ground_truth(tmp_path / "missing.csv")


def test_load_ground_truth_duplicate_row_raises(tmp_path):
    csv_path = tmp_path / "gt.csv"
    _write_gt_csv(
        csv_path,
        ["video1.mp4,4.0,24.0,20.0,30,90.0,", "video1.mp4,4.0,24.0,20.0,30,90.0,"],
    )
    with pytest.raises(GroundTruthMappingError):
        load_ground_truth(csv_path)


def test_load_ground_truth_malformed_row_raises(tmp_path):
    csv_path = tmp_path / "gt.csv"
    _write_gt_csv(csv_path, ["video1.mp4,not_a_number,24.0,20.0,30,90.0,"])
    with pytest.raises(GroundTruthMappingError):
        load_ground_truth(csv_path)


def _make_config(tmp_path):
    split_dir = tmp_path / "split"
    metadata_dir = tmp_path / "metadata"
    split_dir.mkdir()
    metadata_dir.mkdir()
    gt_csv = metadata_dir / "ground_truth_summary.csv"

    config = HybridConfig()
    config.paths.split_dir = split_dir
    config.paths.ground_truth_csv = gt_csv
    return config, split_dir, gt_csv


def test_discover_development_videos_maps_and_sorts(tmp_path):
    config, split_dir, gt_csv = _make_config(tmp_path)
    _write_gt_csv(
        gt_csv,
        [
            "video1.mp4,4.0,24.0,20.0,30,90.0,",
            "video2.mp4,3.0,29.0,26.0,30,69.2308,",
        ],
    )
    (split_dir / "video2_development.mp4").write_bytes(b"")
    (split_dir / "video1_development.mp4").write_bytes(b"")
    (split_dir / "video7_test.mp4").write_bytes(b"")  # must be ignored — test split

    dev_videos = discover_development_videos(config)

    assert [v.video_id for v in dev_videos] == ["video1", "video2"]
    assert dev_videos[0].gt.gt_cpm == pytest.approx(90.0)
    assert dev_videos[1].split_path.name == "video2_development.mp4"


def test_discover_development_videos_missing_gt_row_raises(tmp_path):
    config, split_dir, gt_csv = _make_config(tmp_path)
    _write_gt_csv(gt_csv, ["video1.mp4,4.0,24.0,20.0,30,90.0,"])
    (split_dir / "video99_development.mp4").write_bytes(b"")

    with pytest.raises(GroundTruthMappingError):
        discover_development_videos(config)


def test_discover_development_videos_no_files_raises(tmp_path):
    config, split_dir, gt_csv = _make_config(tmp_path)
    _write_gt_csv(gt_csv, ["video1.mp4,4.0,24.0,20.0,30,90.0,"])

    with pytest.raises(VideoReadError):
        discover_development_videos(config)


def test_discover_development_videos_never_touches_test_split(tmp_path):
    config, split_dir, gt_csv = _make_config(tmp_path)
    _write_gt_csv(gt_csv, ["video1.mp4,4.0,24.0,20.0,30,90.0,"])
    (split_dir / "video1_development.mp4").write_bytes(b"")
    (split_dir / "video1_test.mp4").write_bytes(b"")

    dev_videos = discover_development_videos(config)

    assert len(dev_videos) == 1
    assert "test" not in dev_videos[0].split_path.name
