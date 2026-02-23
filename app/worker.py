"""AutoFlip processing engine — refactored from autoflip.py for service use.

Loads YOLOv8n once at module level and exposes run_autoflip() for reuse.
"""

import cv2
import numpy as np
import subprocess
import os
import tempfile
import time
import shutil
from typing import Callable, Optional

import torch
from ultralytics import YOLO

# ── Constants ────────────────────────────────────────────────────────────────
TARGET_ASPECT = 9 / 16
SMOOTHING_WINDOW = 30
DETECTION_INTERVAL = 3
BORDER_MARGIN = 0.05
BATCH_SIZE = 32
PERSON_CLASS = 0

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


def _compute_crop(frame_w, frame_h, salient_box, target_aspect):
    crop_h = frame_h
    crop_w = int(crop_h * target_aspect)
    if crop_w > frame_w:
        crop_w = frame_w
        crop_h = int(crop_w / target_aspect)

    if salient_box is not None:
        sx1, sy1, sx2, sy2 = salient_box
        center_x = (sx1 + sx2) / 2 * frame_w
        center_y = (sy1 + sy2) / 2 * frame_h
    else:
        center_x = frame_w / 2
        center_y = frame_h / 2

    x = int(center_x - crop_w / 2)
    y = int(center_y - crop_h / 2)
    x = max(0, min(x, frame_w - crop_w))
    y = max(0, min(y, frame_h - crop_h))
    return x, y, crop_w, crop_h


def _smooth_positions(positions, window):
    if not positions:
        return positions
    arr = np.array(positions)
    kernel = np.ones(window) / window
    smoothed_x = np.convolve(arr[:, 0], kernel, mode="same").astype(int)
    smoothed_y = np.convolve(arr[:, 1], kernel, mode="same").astype(int)
    return list(
        zip(
            smoothed_x.tolist(),
            smoothed_y.tolist(),
            arr[:, 2].tolist(),
            arr[:, 3].tolist(),
        )
    )


def run_autoflip(
    input_path: str,
    output_path: str,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> None:
    """Run AutoFlip on input_path and write result to output_path.

    Raises on failure. Calls progress_callback(0.0–1.0) during processing.
    """
    yolo = load_model()
    device = _device
    use_nvenc = has_nvenc()

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
            else:
                detection_results[fidx] = None

        done = min(batch_start + BATCH_SIZE, len(detection_frames))
        if progress_callback:
            # Pass 1 is 0–50%
            progress_callback(done / len(detection_frames) * 0.5)

    del detection_frames
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Build & smooth crop positions ────────────────────────────────────
    crop_positions = []
    last_box = None
    for fidx in range(all_frame_count):
        if fidx in detection_results and detection_results[fidx] is not None:
            last_box = detection_results[fidx]
        x, y, cw, ch = _compute_crop(frame_w, frame_h, last_box, TARGET_ASPECT)
        crop_positions.append((x, y, cw, ch))

    del detection_results
    crop_positions = _smooth_positions(crop_positions, SMOOTHING_WINDOW)

    # ── Pass 2: Write cropped video ──────────────────────────────────────
    crop_w, crop_h = int(crop_positions[0][2]), int(crop_positions[0][3])
    print(f"[AutoFlip] Pass 2: Writing {crop_w}x{crop_h} cropped video")

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

    # ── Mux audio with encoding ──────────────────────────────────────────
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
    print(f"[AutoFlip] Done — {output_path} ({size_mb:.1f} MB, {crop_w}x{crop_h})")
