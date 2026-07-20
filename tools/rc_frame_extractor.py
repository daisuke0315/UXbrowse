#!/usr/bin/env python3
"""Extract RealityCapture-ready stills from a MOV/MP4 video.

Direct fixed-fps JPEG extraction is the default. An optional smart mode can
remove blurry and near-duplicate frames.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

try:
    import numpy as np
    from PIL import Image
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit("Pillow and NumPy are required: python3 -m pip install pillow numpy") from exc

MAX_CANDIDATES = 8000
MAX_DIRECT_FRAMES = 30000


@dataclass(frozen=True)
class FrameMetric:
    path: Path
    index: int
    seconds: float
    sharpness: float
    preview: np.ndarray


def command_path(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise SystemExit(
            f"{name} was not found. On macOS install it with: brew install ffmpeg"
        )
    return path


def run(command: list[str]) -> None:
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"FFmpeg failed with exit code {exc.returncode}") from exc


def video_duration(video: Path, ffprobe: str) -> float:
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def extract_frames_direct(
    video: Path,
    output: Path,
    fps: float,
    ffmpeg: str,
    quality: int = 1,
) -> list[Path]:
    """Reproduce the proven one-pass `fps=N -q:v 1` workflow."""
    output.mkdir(parents=True, exist_ok=True)
    pattern = output / "output_image_%06d.jpg"
    run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-nostdin",
            "-i",
            str(video),
            "-q:v",
            str(quality),
            "-vf",
            f"fps={fps}",
            "-fps_mode",
            "vfr",
            str(pattern),
        ]
    )
    return sorted(output.glob("output_image_*.jpg"))


def write_direct_report(
    video: Path,
    output: Path,
    fps: float,
    quality: int,
    frame_count: int,
) -> None:
    with (output / "extraction_report.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["source", "fps", "jpeg_q", "frame_count"])
        writer.writerow([str(video), f"{fps:g}", quality, frame_count])


def gray_preview(path: Path, width: int = 320) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L")
        height = max(1, round(image.height * width / image.width))
        return np.asarray(image.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32)


def sharpness_score(gray: np.ndarray) -> float:
    """Variance of a discrete Laplacian; larger values are generally sharper."""
    center = gray[1:-1, 1:-1]
    laplacian = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - 4.0 * center
    )
    return float(np.var(laplacian))


def difference_score(left: np.ndarray, right: np.ndarray) -> float:
    """Mean absolute luminance difference, normalized to the range 0..1."""
    if left.shape != right.shape:
        height = min(left.shape[0], right.shape[0])
        left = left[:height]
        right = right[:height]
    return float(np.mean(np.abs(left - right)) / 255.0)


def extract_candidates(video: Path, directory: Path, fps: float, ffmpeg: str) -> list[Path]:
    # Only small previews are needed for blur/similarity analysis. Writing every
    # candidate at source resolution can consume tens or hundreds of gigabytes
    # for long 4K videos.
    pattern = directory / "candidate_%07d.jpg"
    run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-i",
            str(video),
            "-vf",
            f"fps={fps},scale=640:-2",
            "-fps_mode",
            "vfr",
            "-q:v",
            "6",
            str(pattern),
        ]
    )
    return sorted(directory.glob("candidate_*.jpg"))


def analyze(
    paths: list[Path],
    fps: float,
    progress: Optional[Callable[[int, int], None]] = None,
) -> list[FrameMetric]:
    metrics = []
    total = len(paths)
    for index, path in enumerate(paths):
        preview = gray_preview(path)
        metrics.append(FrameMetric(path, index, index / fps, sharpness_score(preview), preview))
        if progress:
            progress(index + 1, total)
    return metrics


def choose_frames(
    metrics: list[FrameMetric],
    min_sharpness: float,
    min_difference: float,
    max_gap: float,
) -> tuple[list[FrameMetric], list[tuple[FrameMetric, str, float]]]:
    kept: list[FrameMetric] = []
    decisions: list[tuple[FrameMetric, str, float]] = []
    for metric in metrics:
        difference = 1.0 if not kept else difference_score(metric.preview, kept[-1].preview)
        gap = float("inf") if not kept else metric.seconds - kept[-1].seconds
        if not kept:
            reason = "first"
        elif metric.sharpness >= min_sharpness and difference >= min_difference:
            reason = "changed"
        elif gap >= max_gap and metric.sharpness >= min_sharpness * 0.6:
            reason = "gap_guard"
        elif metric.sharpness < min_sharpness:
            decisions.append((metric, "blur", difference))
            continue
        else:
            decisions.append((metric, "duplicate", difference))
            continue
        kept.append(metric)
        decisions.append((metric, reason, difference))
    return kept, decisions


def output_frames(frames: list[FrameMetric], output: Path, quality: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for number, frame in enumerate(frames, start=1):
        destination = output / f"rc_{number:05d}_{frame.seconds:010.3f}s.jpg"
        with Image.open(frame.path) as image:
            image.convert("RGB").save(destination, quality=quality, subsampling=0)


def output_frames_from_video(
    video: Path,
    frames: list[FrameMetric],
    output: Path,
    sample_fps: float,
    ffmpeg: str,
) -> None:
    """Decode the source again and write only selected frames at full resolution."""
    if not frames:
        return
    output.mkdir(parents=True, exist_ok=True)
    expression = "+".join(f"eq(n\\,{frame.index})" for frame in frames)
    temporary_pattern = output / "selected_%07d.jpg"
    run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-i",
            str(video),
            "-vf",
            f"fps={sample_fps},select={expression}",
            "-fps_mode",
            "vfr",
            "-q:v",
            "2",
            str(temporary_pattern),
        ]
    )
    decoded = sorted(output.glob("selected_*.jpg"))
    if len(decoded) != len(frames):
        raise RuntimeError(
            f"Expected {len(frames)} selected frames, but FFmpeg wrote {len(decoded)}"
        )
    for number, (path, frame) in enumerate(zip(decoded, frames), start=1):
        path.rename(output / f"rc_{number:05d}_{frame.seconds:010.3f}s.jpg")


def write_report(decisions: list[tuple[FrameMetric, str, float]], output: Path) -> None:
    with (output / "selection_report.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["time_seconds", "sharpness", "difference", "decision"])
        for metric, reason, difference in decisions:
            writer.writerow([f"{metric.seconds:.3f}", f"{metric.sharpness:.3f}", f"{difference:.5f}", reason])


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("video", type=Path, help="Input .mov/.mp4 file")
    result.add_argument("-o", "--output", type=Path, default=Path("rc_frames"))
    result.add_argument("--mode", choices=("direct", "smart"), default="direct")
    result.add_argument("--sample-fps", type=float, default=10.0, help="Frames per second")
    result.add_argument("--min-sharpness", type=float, default=18.0, help="Reject blur below this score")
    result.add_argument("--min-difference", type=float, default=0.018, help="Reject near-duplicates below this score")
    result.add_argument("--max-gap", type=float, default=1.5, help="Maximum desired gap between kept frames")
    result.add_argument("--quality", type=int, default=1, choices=range(1, 32), metavar="1..31")
    result.add_argument(
        "--max-frames", type=int, default=MAX_DIRECT_FRAMES,
        help="Direct mode safety limit; raise explicitly for longer jobs",
    )
    result.add_argument(
        "--max-candidates", type=int, default=MAX_CANDIDATES,
        help="Abort before decoding when the estimated candidate count exceeds this limit",
    )
    return result


def main() -> None:
    args = parser().parse_args()
    if not args.video.is_file():
        raise SystemExit(f"Video not found: {args.video}")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(
            f"Output directory is not empty: {args.output}\n"
            "Choose a new directory so frames from separate runs are not mixed."
        )
    if args.sample_fps <= 0 or args.max_gap <= 0:
        raise SystemExit("--sample-fps and --max-gap must be positive")
    ffmpeg = command_path("ffmpeg")
    ffprobe = command_path("ffprobe")
    duration = video_duration(args.video, ffprobe)
    print(f"Video: {duration:.1f}s; sampling at {args.sample_fps:g} fps")
    estimated_candidates = math.ceil(duration * args.sample_fps)
    if args.mode == "direct":
        if estimated_candidates > args.max_frames:
            raise SystemExit(
                f"Estimated frames ({estimated_candidates}) exceed the safety limit "
                f"({args.max_frames}). Lower --sample-fps, split the video, or raise --max-frames."
            )
        frames = extract_frames_direct(
            args.video, args.output, args.sample_fps, ffmpeg, args.quality
        )
        write_direct_report(
            args.video, args.output, args.sample_fps, args.quality, len(frames)
        )
        print(f"Wrote {len(frames)} frames to {args.output}")
        return
    if estimated_candidates > args.max_candidates:
        raise SystemExit(
            f"Estimated candidates ({estimated_candidates}) exceed the smart-mode limit "
            f"({args.max_candidates}). Lower --sample-fps or split the video."
        )
    with tempfile.TemporaryDirectory(prefix="rc-frames-") as temporary:
        candidates = extract_candidates(args.video, Path(temporary), args.sample_fps, ffmpeg)
        if not candidates:
            raise SystemExit("No frames could be decoded")
        metrics = analyze(candidates, args.sample_fps)
        kept, decisions = choose_frames(
            metrics, args.min_sharpness, args.min_difference, args.max_gap
        )
        output_frames_from_video(args.video, kept, args.output, args.sample_fps, ffmpeg)
        write_report(decisions, args.output)
    print(f"Kept {len(kept)} of {len(decisions)} candidates in {args.output}")


if __name__ == "__main__":
    main()
