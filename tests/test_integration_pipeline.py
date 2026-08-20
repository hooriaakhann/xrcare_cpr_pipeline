"""Phase 19 integration test: the real, unmocked Phase 1-12 pipeline
end-to-end on one development video.

Every other test in this suite verifies one phase in isolation (mocking
its neighbors); this is the one place the whole chain runs for real and is
checked as a system. Uses `video3_development.mp4` -- already validated
piece-by-piece against real footage throughout Phases 2-13's development
(see PROGRESS.md) -- rather than a new committed fixture: by the time this
test runs in a real dev environment, every branch is already cached from
that development work, so it runs in seconds, not the ~15-20 minutes a
fully cold run would take. Marked slow and skips gracefully when
`data/split/` isn't present (e.g. in CI, which never has the raw videos),
matching the pattern already used for Phase 2/3's real-model tests.
"""

from pathlib import Path

import numpy as np
import pytest

from hybrid.caching import CacheManager
from hybrid.config import load_config
from hybrid.dataset import discover_development_videos
from hybrid.evaluation import run_full_pipeline_on_video


@pytest.mark.slow
def test_full_pipeline_end_to_end_on_real_video():
    split_dir = Path(__file__).resolve().parents[1] / "data" / "split"
    if not split_dir.exists():
        pytest.skip("real split videos not present locally (data/split is gitignored)")

    config = load_config()
    cache_manager = CacheManager(config.paths.cache_dir)
    dev_videos = discover_development_videos(config)
    dev_video = next((v for v in dev_videos if v.video_id == "video3"), dev_videos[0])

    result = run_full_pipeline_on_video(dev_video, config, cache_manager)

    assert np.isfinite(result.final_cpm)
    assert 20.0 < result.final_cpm < 300.0  # broad physiological sanity bound, not an accuracy claim
    assert 0.0 <= result.overall_confidence <= 1.0
    assert result.runtime_sec >= 0.0
    # this video's real GT is well-established from Phase 0's mapping (see
    # PROGRESS.md); a loose bound here catches a badly broken pipeline
    # without re-asserting Phase 13's precise per-video accuracy numbers
    assert abs(result.final_cpm - result.gt_cpm) < 30.0
