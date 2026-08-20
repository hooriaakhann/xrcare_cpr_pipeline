# Hybrid CPR Compression Rate Estimation

Egocentric smart-glasses CPR video → hybrid CV pipeline for compression rate
estimation. This repo currently implements only the **video preprocessing
stage**. Detection/estimation algorithms (MediaPipe, CoTracker, ego-motion
compensation, optical flow, RepNet, CWT, autocorrelation, peak detection,
fusion) are not implemented yet.

## Project structure

```
data/
├── raw/                  # original videos — never modified
├── processed/             # trimmed / orientation-corrected full-frame videos
└── metadata/
    ├── trim_times.csv               # you fill this in (CPR start/end per video)
    ├── video_inspection_report.csv  # output of inspect_videos.py
    ├── preprocessing_metadata.csv   # output of preprocess.py
    └── timestamps/                  # per-video original<->processed frame PTS maps

src/
└── preprocessing/
    ├── inspect_videos.py   # probes data/raw/, reports FPS/res/duration/timing
    └── preprocess.py       # trims + optionally rotates, writes metadata

runs/                       # scratch space for future pipeline runs
.venv/                      # local virtualenv (opencv-python, numpy)
```

## Setup

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

## Usage

1. Inspect the raw videos (read-only, writes a report):

   ```
   .venv\Scripts\python src\preprocessing\inspect_videos.py
   ```

2. Edit `data/metadata/trim_times.csv` — set `cpr_start_sec` / `cpr_end_sec`
   per video to mark where CPR actually starts/ends (leave blank to keep the
   full video). Set `rotate_degrees` (0/90/180/270) only for videos you've
   confirmed are actually mis-rotated — default is 0 (no rotation).

3. Run the preprocessing pipeline:

   ```
   .venv\Scripts\python src\preprocessing\preprocess.py
   ```

   This writes trimmed/rotated full-frame videos to `data/processed/`, a
   per-video frame-timestamp mapping sidecar to
   `data/metadata/timestamps/`, and a summary row per video to
   `data/metadata/preprocessing_metadata.csv`.

## Design notes

- **No spatial cropping / ROI at this stage.** Full frame is preserved so
  later stages (MediaPipe hand/wrist detection, CoTracker point tracking,
  affine/RANSAC ego-motion compensation using background features, RepNet)
  all have the complete frame to work with.
- **No resolution/FPS standardization.** Source videos share a consistent
  aspect ratio (~3:4 portrait) but not identical pixel dimensions, and all
  average ~30fps — neither is forced to match since none of the downstream
  algorithms require it.
- **Source videos are variable frame rate (VFR).** Frame-to-frame timing is
  not constant (varies roughly 25-42ms between frames even though the
  average is ~30fps). Trimming uses ffmpeg's accurate/frame-exact seeking on
  real per-frame timestamps rather than frame-index arithmetic, and the
  per-video sidecar in `data/metadata/timestamps/` records the actual
  original-video timestamp for every retained frame so later ground-truth
  annotation can be mapped back precisely:
  `processed_time = original_time - actual_trim_start_sec`.
- **Processed videos stay HEVC 10-bit (`yuv420p10le`)**, matching the
  source. Verified that OpenCV (FFMPEG backend) decodes both the raw source
  and this re-encoded output correctly (all frames, standard `uint8` BGR
  arrays) before committing to this codec.
