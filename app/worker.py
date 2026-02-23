"""AutoFlip processing engine — refactored from autoflip.py for service use.

Loads YOLOv8n once at module level and exposes run_autoflip() for reuse.
"""

import cv2
import numpy as np
import subprocess
import os
import tempfile
from typing import Callable, Optional

import torch
from ultralytics import YOLO

# ── Constants ────────────────────────────────────────────────────────────────
DEFAULT_TARGET_ASPECT = 9 / 16
SMOOTHING_WINDOW = 30
DETECTION_INTERVAL = 3
BORDER_MARGIN = 0.05
BATCH_SIZE = 32
PERSON_CLASS = 0
VIDEO_FORMAT_TO_ASPECT = {
    "PORTRAIT": 9 / 16,
    "WIDESCREEN": 16 / 9,
    "SQUARE": 1.0,
}
# Standard output resolutions per format (width x height).
VIDEO_FORMAT_TO_RESOLUTION = {
    "PORTRAIT": (1080, 1920),
    "WIDESCREEN": (1920, 1080),
    "SQUARE": (1080, 1080),
}
# Tolerance for comparing source aspect to target aspect.
_ASPECT_TOLERANCE = 0.08

# ── Singleton model ──────────────────────────────────────────────────────────
_yolo_model: Optional[YOLO] = None
_device: str = "cpu"


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def load_model() -> YOLO:
    """Load YOLOv8n once and cache it."""
    global _yolo_model, _device
    if _yolo_model is None:
        _device = get_device()
        print(f"[AutoFlip] Loading YOLOv8n on {_device}...")
        _yolo_model = YOLO("yolov8n.pt")
        _yolo_model.to(_device)
        print("[AutoFlip] Model ready.")
    return _yolo_model


def has_nvenc() -> bool:
    """Check if ffmpeg supports h264_nvenc."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
        return "h264_nvenc" in result.stdout
    except Exception:
        return False


def _resolve_target_aspect(
    *,
    video_format: Optional[str],
    target_aspect_ratio: Optional[float],
) -> float:
    if target_aspect_ratio is not None:
        value = float(target_aspect_ratio)
        if value <= 0:
            raise ValueError("'target_aspect_ratio' must be greater than 0.")
        return value
    normalized_format = str(video_format or "PORTRAIT").strip().upper()
    return VIDEO_FORMAT_TO_ASPECT.get(normalized_format, DEFAULT_TARGET_ASPECT)


def _resolve_output_resolution(
    video_format: Optional[str],
    crop_w: int,
    crop_h: int,
) -> tuple[int, int]:
    """Determine the final output resolution.

    Uses the standard resolution for the format when the crop is smaller,
    otherwise keeps the native crop size to avoid unnecessary upscaling of
    already-large sources.
    """
    normalized_format = str(video_format or "PORTRAIT").strip().upper()
    standard = VIDEO_FORMAT_TO_RESOLUTION.get(normalized_format)
    if standard is None:
        return crop_w, crop_h
    std_w, std_h = standard
    # Only upscale when the crop is smaller than the standard resolution.
    # If the crop is already larger, keep it to preserve quality.
    if crop_w < std_w or crop_h < std_h:
        return std_w, std_h
    return crop_w, crop_h


def _source_matches_target(frame_w: int, frame_h: int, target_aspect: float) -> bool:
    """Check if the source video already matches the target aspect ratio."""
    if frame_w <= 0 or frame_h <= 0:
        return False
    source_aspect = float(frame_w) / float(frame_h)
    return abs(source_aspect - target_aspect) <= _ASPECT_TOLERANCE


def _compute_crop_size(frame_w: int, frame_h: int, target_aspect: float) -> tuple[int, int]:
    crop_h = frame_h
    crop_w = int(round(crop_h * target_aspect))
    if crop_w > frame_w:
        crop_w = frame_w
        crop_h = int(round(crop_w / target_aspect))
    crop_w = max(2, crop_w)
    crop_h = max(2, crop_h)
    return crop_w, crop_h


def _compute_crop_from_center(
    frame_w: int,
    frame_h: int,
    center_x: float,
    center_y: float,
    crop_w: int,
    crop_h: int,
) -> tuple[int, int, int, int]:
    x = int(round(center_x - crop_w / 2))
    y = int(round(center_y - crop_h / 2))
    x = max(0, min(x, frame_w - crop_w))
    y = max(0, min(y, frame_h - crop_h))
    return x, y, crop_w, crop_h


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) <= 1:
        return values.astype(np.float32)
    window = max(1, min(window, len(values)))
    if window % 2 == 0:
        window = max(1, window - 1)
    if window <= 1:
        return values.astype(np.float32)
    pad = window // 2
    padded = np.pad(values.astype(np.float32), (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def _smooth_positions(positions, window):
    if not positions:
        return positions
    arr = np.array(positions, dtype=np.float32)
    smoothed_x = np.round(_moving_average(arr[:, 0], window)).astype(int)
    smoothed_y = np.round(_moving_average(arr[:, 1], window)).astype(int)
    return list(
        zip(
            smoothed_x.tolist(),
            smoothed_y.tolist(),
            arr[:, 2].tolist(),
            arr[:, 3].tolist(),
        )
    )


def _interpolate_centers(
    detection_results: dict[int, Optional[tuple[float, float, float, float]]],
    all_frame_count: int,
    frame_w: int,
    frame_h: int,
) -> list[tuple[float, float]]:
    default_center = (frame_w / 2.0, frame_h / 2.0)
    keyframes: list[tuple[int, float, float]] = []
    for frame_idx in sorted(detection_results.keys()):
        box = detection_results.get(frame_idx)
        if box is None:
            continue
        sx1, sy1, sx2, sy2 = box
        center_x = ((sx1 + sx2) / 2.0) * frame_w
        center_y = ((sy1 + sy2) / 2.0) * frame_h
        keyframes.append((frame_idx, center_x, center_y))

    if not keyframes:
        return [default_center for _ in range(all_frame_count)]

    centers: list[tuple[float, float]] = [default_center for _ in range(all_frame_count)]
    first_idx, first_x, first_y = keyframes[0]
    for frame_idx in range(0, min(first_idx + 1, all_frame_count)):
        centers[frame_idx] = (first_x, first_y)

    for left_idx in range(len(keyframes) - 1):
        start_frame, start_x, start_y = keyframes[left_idx]
        end_frame, end_x, end_y = keyframes[left_idx + 1]
        span = max(1, end_frame - start_frame)
        for frame_idx in range(start_frame, min(end_frame + 1, all_frame_count)):
            alpha = (frame_idx - start_frame) / float(span)
            x = start_x + (end_x - start_x) * alpha
            y = start_y + (end_y - start_y) * alpha
            centers[frame_idx] = (x, y)

    last_idx, last_x, last_y = keyframes[-1]
    for frame_idx in range(max(0, last_idx), all_frame_count):
        centers[frame_idx] = (last_x, last_y)

    return centers


def run_autoflip(
    input_path: str,
    output_path: str,
    progress_callback: Optional[Callable[[float], None]] = None,
    video_format: str = "PORTRAIT",
    target_aspect_ratio: Optional[float] = None,
) -> None:
    """Run AutoFlip on input_path and write result to output_path.

    Raises on failure. Calls progress_callback(0.0–1.0) during processing.
    """
    yolo = load_model()
    device = _device
    use_nvenc = has_nvenc()
    target_aspect = _resolve_target_aspect(
        video_format=video_format,
        target_aspect_ratio=target_aspect_ratio,
    )

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise RuntimeError("Video has no frames")

    print(
        f"[AutoFlip] Source: {frame_w}x{frame_h} @ {fps}fps, "
        f"{total_frames} frames (~{total_frames / fps:.0f}s)"
    )
    print(
        f"[AutoFlip] Target format={str(video_format or 'PORTRAIT').upper()}, "
        f"aspect={target_aspect:.4f}"
    )

    # ── Fast path: source already matches the target aspect ──────────────
    if _source_matches_target(frame_w, frame_h, target_aspect):
        cap.release()
        out_w, out_h = _resolve_output_resolution(video_format, frame_w, frame_h)
        print(
            f"[AutoFlip] Source already matches target aspect ratio "
            f"({frame_w}x{frame_h} ≈ {target_aspect:.4f}). "
            f"Scaling to {out_w}x{out_h}."
        )
        encoder = "h264_nvenc" if use_nvenc else "libx264"
        cmd = ["ffmpeg", "-y"]
        if use_nvenc:
            cmd += ["-hwaccel", "cuda"]
        cmd += ["-i", input_path]
        if out_w != frame_w or out_h != frame_h:
            cmd += ["-vf", f"scale={out_w}:{out_h}"]
        if use_nvenc:
            cmd += ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "23"]
        else:
            cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "23"]
        cmd += ["-c:a", "copy", output_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg scaling failed: {result.stderr[-1000:]}")
        if progress_callback:
            progress_callback(1.0)
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[AutoFlip] Done (pass-through) — {output_path} ({size_mb:.1f} MB, {out_w}x{out_h})")
        return

    # ── Pass 1: Read frames for detection ────────────────────────────────
    detection_frames = []
    all_frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if all_frame_count % DETECTION_INTERVAL == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detection_frames.append((all_frame_count, rgb))
        all_frame_count += 1

    print(f"[AutoFlip] Pass 1: {len(detection_frames)} frames to detect")

    # ── Run YOLO in batches ──────────────────────────────────────────────
    detection_results = {}
    detected_count = 0
    for batch_start in range(0, len(detection_frames), BATCH_SIZE):
        batch = detection_frames[batch_start : batch_start + BATCH_SIZE]
        batch_imgs = [f[1] for f in batch]

        results = yolo.predict(
            batch_imgs,
            classes=[PERSON_CLASS],
            verbose=False,
            device=device,
            imgsz=320,
        )

        for (fidx, _), result in zip(batch, results):
            boxes = []
            for det in result.boxes:
                x1, y1, x2, y2 = det.xyxy[0].cpu().numpy()
                nx1, ny1 = x1 / frame_w, y1 / frame_h
                nx2, ny2 = x2 / frame_w, y2 / frame_h
                boxes.append((nx1, ny1, nx2, ny2))

            if boxes:
                mx1 = min(b[0] for b in boxes)
                my1 = min(b[1] for b in boxes)
                mx2 = max(b[2] for b in boxes)
                my2 = max(b[3] for b in boxes)
                bw, bh = mx2 - mx1, my2 - my1
                mx1 = max(0, mx1 - bw * BORDER_MARGIN)
                my1 = max(0, my1 - bh * BORDER_MARGIN)
                mx2 = min(1, mx2 + bw * BORDER_MARGIN)
                my2 = min(1, my2 + bh * BORDER_MARGIN)
                detection_results[fidx] = (mx1, my1, mx2, my2)
                detected_count += 1
            else:
                detection_results[fidx] = None

        done = min(batch_start + BATCH_SIZE, len(detection_frames))
        if progress_callback and len(detection_frames) > 0:
            # Pass 1 is 0–50%
            progress_callback(done / len(detection_frames) * 0.5)

    detection_rate = (detected_count / len(detection_frames) * 100) if detection_frames else 0
    print(
        f"[AutoFlip] Detection complete: {detected_count}/{len(detection_frames)} "
        f"frames with person ({detection_rate:.1f}%)"
    )
    if detection_rate < 10:
        print("[AutoFlip] WARNING: Very low person detection rate — output will use center crop.")

    del detection_frames
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Build & smooth crop positions ────────────────────────────────────
    crop_w, crop_h = _compute_crop_size(frame_w, frame_h, target_aspect)
    frame_centers = _interpolate_centers(detection_results, all_frame_count, frame_w, frame_h)
    raw_positions = [
        _compute_crop_from_center(frame_w, frame_h, cx, cy, crop_w, crop_h)
        for cx, cy in frame_centers
    ]

    del detection_results
    crop_positions = _smooth_positions(raw_positions, SMOOTHING_WINDOW)

    # ── Pass 2: Write cropped video ──────────────────────────────────────
    crop_w, crop_h = int(crop_positions[0][2]), int(crop_positions[0][3])
    out_w, out_h = _resolve_output_resolution(video_format, crop_w, crop_h)
    needs_scale = (out_w != crop_w or out_h != crop_h)
    print(
        f"[AutoFlip] Pass 2: Cropping to {crop_w}x{crop_h}"
        + (f", scaling to {out_w}x{out_h}" if needs_scale else "")
    )

    tmp_video = tempfile.mktemp(suffix=".mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_video, fourcc, fps, (crop_w, crop_h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        x, y, cw, ch = [int(v) for v in crop_positions[frame_idx]]
        cropped = frame[y : y + ch, x : x + cw]
        if cropped.shape[1] != crop_w or cropped.shape[0] != crop_h:
            cropped = cv2.resize(cropped, (crop_w, crop_h))
        writer.write(cropped)
        frame_idx += 1

        if progress_callback and frame_idx % 500 == 0:
            # Pass 2 is 50–90%
            progress_callback(0.5 + (frame_idx / total_frames) * 0.4)

    writer.release()
    cap.release()

    if progress_callback:
        progress_callback(0.9)

    # ── Mux audio with encoding (+ optional scale to standard resolution) ─
    encoder = "h264_nvenc" if use_nvenc else "libx264"
    print(f"[AutoFlip] Encoding with {encoder}")

    cmd = [
        "ffmpeg", "-y",
    ]
    if use_nvenc:
        cmd += ["-hwaccel", "cuda"]
    cmd += [
        "-i", tmp_video,
        "-i", input_path,
    ]
    # Apply scaling to standard resolution if needed.
    if needs_scale:
        cmd += ["-vf", f"scale={out_w}:{out_h}"]
    if use_nvenc:
        cmd += ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "23"]
    else:
        cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "23"]
    cmd += [
        "-c:a", "copy",
        "-map", "0:v:0", "-map", "1:a:0?",
        "-shortest",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    os.unlink(tmp_video)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg encoding failed: {result.stderr[-1000:]}")

    if progress_callback:
        progress_callback(1.0)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[AutoFlip] Done — {output_path} ({size_mb:.1f} MB, {out_w}x{out_h})")
