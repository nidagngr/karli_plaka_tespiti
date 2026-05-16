from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class LetterboxInfo:
    scale: float
    pad_x: int
    pad_y: int
    original_size: Tuple[int, int]
    target_size: int


def letterbox_to_square(image_bgr: np.ndarray, target_size: int, pad_value: int = 114) -> Tuple[np.ndarray, LetterboxInfo]:
    h, w = image_bgr.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("Invalid image size.")

    scale = min(target_size / w, target_size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)

    canvas = np.full((target_size, target_size, 3), pad_value, dtype=np.uint8)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    info = LetterboxInfo(scale=scale, pad_x=pad_x, pad_y=pad_y, original_size=(w, h), target_size=target_size)
    return canvas, info


def apply_clahe(image_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    enhanced = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def adjust_contrast(image_bgr: np.ndarray, alpha: float = 1.16, beta: int = 0) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    bright_ratio = float(np.mean(l_channel > 215))
    mean_l = float(np.mean(l_channel))

    if bright_ratio > 0.24:
        gamma = 1.25
        normalized = l_channel.astype(np.float32) / 255.0
        compressed = np.power(normalized, gamma) * 255.0
        l_channel = np.clip(compressed, 0, 255).astype(np.uint8)
        l_channel = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(8, 8)).apply(l_channel)
    elif bright_ratio > 0.12 or mean_l > 172:
        l_channel = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(l_channel)
    else:
        l_channel = cv2.convertScaleAbs(l_channel, alpha=alpha, beta=beta)

    return cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)


def sharpen_image(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    bright_ratio = float(np.mean(gray > 220))
    if bright_ratio > 0.12:
        strength = 0.28
        sigma = 0.8
    else:
        strength = 0.45
        sigma = 1.0
    blurred = cv2.GaussianBlur(image_bgr, (0, 0), sigmaX=sigma)
    return cv2.addWeighted(image_bgr, 1.0 + strength, blurred, -strength, 0)


def morphological_filter(image_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(l_channel, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    filtered_l = cv2.addWeighted(l_channel, 0.65, closed, 0.35, 0)
    return cv2.cvtColor(cv2.merge((filtered_l, a_channel, b_channel)), cv2.COLOR_LAB2BGR)


def gaussian_sharpen(image_bgr: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(image_bgr, (0, 0), sigmaX=1.4)
    return cv2.addWeighted(image_bgr, 1.7, blurred, -0.7, 0)


def build_yolo_detection_image(image_bgr: np.ndarray) -> np.ndarray:
    clahe = apply_clahe(image_bgr)
    filtered = morphological_filter(clahe)
    return gaussian_sharpen(filtered)


def clean_noise(image_bgr: np.ndarray) -> np.ndarray:
    denoised = cv2.fastNlMeansDenoisingColored(image_bgr, None, 7, 7, 7, 21)
    return cv2.bilateralFilter(denoised, 5, 45, 45)


def suppress_bright_snow(image_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    bright_ratio = float(np.mean(l_channel > 190))

    if bright_ratio < 0.18:
        return image_bgr.copy()

    normalized = l_channel.astype(np.float32) / 255.0
    compressed = np.power(normalized, 1.45) * 255.0
    compressed = np.clip(compressed, 0, 255).astype(np.uint8)
    enhanced = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8)).apply(compressed)
    return cv2.cvtColor(cv2.merge((enhanced, a_channel, b_channel)), cv2.COLOR_LAB2BGR)


def dark_text_emphasis(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5)))
    dark = cv2.subtract(background, gray)
    dark = cv2.normalize(dark, None, 0, 255, cv2.NORM_MINMAX)
    dark = cv2.GaussianBlur(dark, (3, 3), 0)
    return cv2.cvtColor(dark, cv2.COLOR_GRAY2BGR)


def rescue_snow_plate(image_bgr: np.ndarray) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    lower = image_bgr[int(h * 0.25) : h, int(w * 0.02) : int(w * 0.92)]
    if lower.size == 0:
        lower = image_bgr

    up = cv2.resize(lower, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (61, 21)))
    normalized = cv2.divide(gray, background, scale=255)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(normalized)
    dark = cv2.subtract(
        cv2.morphologyEx(clahe, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 9))),
        clahe,
    )
    dark = cv2.normalize(dark, None, 0, 255, cv2.NORM_MINMAX)
    adaptive = cv2.adaptiveThreshold(
        dark,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        41,
        -3,
    )
    return cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)


def ocr_preprocess(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(gray)
    denoised = cv2.bilateralFilter(clahe, 5, 45, 45)
    normalized = cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX)
    blurred = cv2.GaussianBlur(normalized, (0, 0), sigmaX=0.8)
    enhanced = cv2.addWeighted(normalized, 1.35, blurred, -0.35, 0)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def preprocess_for_plate_ui(image_bgr: np.ndarray, target_size: int) -> Tuple[Dict[str, np.ndarray], LetterboxInfo]:
    resized, info = letterbox_to_square(image_bgr, target_size)
    clahe = apply_clahe(resized)
    contrast = adjust_contrast(clahe)
    sharpened = sharpen_image(contrast)
    cleaned = clean_noise(sharpened)
    snow_suppressed = suppress_bright_snow(clahe)
    dark_text = dark_text_emphasis(snow_suppressed)
    ocr_ready = ocr_preprocess(snow_suppressed)

    return (
        {
            "01 YOLO model girisi": resized,
            "02 CLAHE": clahe,
            "03 Contrast adjustment": contrast,
            "04 Sharpening": sharpened,
            "05 Noise cleaning": cleaned,
            "06 Snow/light suppression": snow_suppressed,
            "07 Dark text emphasis": dark_text,
            "08 OCR preprocessing": ocr_ready,
        },
        info,
    )
