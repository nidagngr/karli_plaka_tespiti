from __future__ import annotations

import base64
import cgi
import html
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from gan_restoration import build_restorer_from_config
from ocr_utils import (
    choose_best_supported_plate_candidate,
    format_supported_plate_candidates,
    read_degraded_plate_candidates,
    read_plate_candidates_from_stages,
)
from preprocess_utils import (
    adjust_contrast,
    apply_clahe,
    clean_noise,
    dark_text_emphasis,
    letterbox_to_square,
    ocr_preprocess,
    rescue_snow_plate,
    sharpen_image,
    suppress_bright_snow,
)
from utils import add_padding_to_box, load_yaml, read_image, resolve_project_root, to_absolute, write_image


PROJECT_ROOT = resolve_project_root()
PIPELINE_CFG = load_yaml(PROJECT_ROOT / "configs" / "pipeline_config.yaml")
YOLO_CFG = load_yaml(PROJECT_ROOT / "configs" / "yolo_config.yaml")
TARGET_SIZE = int(YOLO_CFG["train"]["imgsz"])
UPLOAD_DIR = PROJECT_ROOT / "outputs" / "web_ui"

yolo_model = None
ocr_reader = None
restorer = None
restorer_loaded = False


def image_to_data_uri(image_bgr) -> str:
    ok, encoded = cv2.imencode(".png", image_bgr)
    if not ok:
        return ""
    data = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def get_yolo_model():
    global yolo_model
    if yolo_model is None:
        from ultralytics import YOLO

        weights = to_absolute(PROJECT_ROOT, PIPELINE_CFG["paths"]["yolo_weights"])
        if not weights.exists():
            raise FileNotFoundError(f"YOLO agirlik dosyasi bulunamadi: {weights}")
        yolo_model = YOLO(str(weights))
    return yolo_model


def get_ocr_reader():
    global ocr_reader
    if ocr_reader is None:
        from easyocr import Reader

        ocr_cfg = PIPELINE_CFG["ocr"]
        ocr_reader = Reader(ocr_cfg["languages"], gpu=bool(ocr_cfg["gpu"]))
    return ocr_reader


def get_restorer():
    global restorer, restorer_loaded
    if not bool(PIPELINE_CFG.get("gan", {}).get("enabled", False)):
        return None
    if not restorer_loaded:
        restorer_loaded = True
        try:
            restorer = build_restorer_from_config(PROJECT_ROOT, PIPELINE_CFG)
        except Exception:
            restorer = None
    return restorer


def run_plate_pipeline(image_path: Path) -> Dict[str, Any]:
    original = read_image(image_path)
    model_input, _ = letterbox_to_square(original, TARGET_SIZE)
    stages = {"01 YOLO model girisi": model_input}

    yolo_cfg = PIPELINE_CFG["yolo"]
    prediction = get_yolo_model().predict(
        source=model_input,
        conf=float(yolo_cfg["conf_threshold"]),
        iou=float(yolo_cfg["iou_threshold"]),
        max_det=int(yolo_cfg["max_det"]),
        verbose=False,
    )[0]
    h, w = model_input.shape[:2]
    detection_label = "YOLO"
    yolo_confidence: Optional[float] = None

    if prediction.boxes is not None and len(prediction.boxes) > 0:
        box = sorted(prediction.boxes, key=lambda item: float(item.conf[0]), reverse=True)[0]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = add_padding_to_box(
            int(x1), int(y1), int(x2), int(y2), float(yolo_cfg["crop_padding"]), w, h
        )
        crop = model_input[y1:y2, x1:x2]
        yolo_confidence = float(box.conf[0])
    else:
        fallback = find_plate_like_crop(model_input)
        if fallback is None:
            return {
                "plate": "-",
                "details": "Plaka tespit edilemedi. YOLO kutu bulamadi ve fallback plaka bolgesi de bulunamadi.",
                "stages": stages,
            }
        crop = fallback
        detection_label = "Fallback: plaka benzeri bolge"

    refined_plate = refine_inner_plate_crop(crop)
    if refined_plate is not None:
        crop = refined_plate
        detection_label = f"{detection_label} -> ic plaka refine"

    clahe = apply_clahe(crop)
    contrast = adjust_contrast(clahe)
    sharpened = sharpen_image(contrast)
    cleaned = clean_noise(sharpened)
    snow_suppressed = suppress_bright_snow(clahe)
    dark_text = dark_text_emphasis(snow_suppressed)
    rescue_plate = rescue_snow_plate(crop)
    ocr_ready = ocr_preprocess(snow_suppressed)

    stages.update(
        {
            "02 Crop": crop,
            "03 CLAHE": clahe,
            "04 Contrast adjustment": contrast,
            "05 Sharpening": sharpened,
            "06 Noise cleaning": cleaned,
            "07 Snow/light suppression": snow_suppressed,
            "08 Dark text emphasis": dark_text,
            "09 Snow rescue OCR view": rescue_plate,
            "10 OCR preprocessing": ocr_ready,
        }
    )

    ocr_image = ocr_ready
    gan_status = "GAN Restoration: kapali"
    model_restorer = get_restorer()
    if model_restorer is not None:
        ocr_image = model_restorer.restore(ocr_ready)
        stages["08 GAN Restoration"] = ocr_image
        gan_status = "GAN Restoration: aktif"

    ocr_cfg = PIPELINE_CFG["ocr"]
    ocr_sources = [
        ("crop", crop),
        ("clahe", clahe),
        ("snow_suppression", snow_suppressed),
        ("dark_text", dark_text),
        ("snow_rescue", rescue_plate),
        ("contrast", contrast),
        ("sharpening", sharpened),
        ("noise_cleaning", cleaned),
        ("ocr_preprocessing", ocr_image),
    ]
    candidates = read_plate_candidates_from_stages(
        get_ocr_reader(),
        ocr_sources,
        str(ocr_cfg["allowlist"]),
        float(ocr_cfg.get("min_confidence", 0.0)),
    )
    if not candidates:
        candidates = read_degraded_plate_candidates(get_ocr_reader(), crop, str(ocr_cfg["allowlist"]))
    best = choose_best_supported_plate_candidate(candidates, beam_width=1)
    if best and (not best.regex_ok or best.confidence < 0.20 or len(best.normalized) < 6):
        best = type(best)("", 0.0, "none", "")
    raw = " | ".join(candidate.raw_text for candidate in candidates)
    ranked = format_supported_plate_candidates(candidates, beam_width=1, limit=8)

    details = "\n".join(
        [
            f"Tespit kaynagi: {detection_label}",
            f"YOLO confidence: {yolo_confidence:.4f}" if yolo_confidence is not None else "YOLO confidence: yok",
            gan_status,
            "Akis: YOLO -> Crop -> CLAHE -> Contrast Adjustment -> Sharpening -> Noise cleaning -> OCR preprocessing -> OCR",
            f"Secilen plaka: {best.normalized or '-'}",
            f"OCR kaynagi: {best.source}",
            f"OCR confidence: {best.confidence:.4f}",
            f"Regex uygun: {'evet' if best.regex_ok else 'hayir'}",
            "",
            f"Ham OCR sonucu: {raw or '-'}",
            "",
            "OCR adaylari:",
            ranked or "-",
        ]
    )

    display_plate = best.normalized
    if not display_plate and detection_label.startswith("Fallback"):
        hard_dir = PROJECT_ROOT / "data" / "hard_plates"
        hard_dir.mkdir(parents=True, exist_ok=True)
        source_name = image_path.stem
        write_image(hard_dir / f"{source_name}_crop.png", crop)
        write_image(hard_dir / f"{source_name}_rescue.png", rescue_plate)
        display_plate = "OCR OKUNAMADI"
    return {"plate": display_plate or "-", "details": details, "stages": stages}


def find_plate_like_crop(image_bgr):
    blue_crop = find_blue_strip_plate_crop(image_bgr)
    if blue_crop is not None and is_plausible_plate_crop(blue_crop):
        return blue_crop

    dirty_white_crop = find_dirty_white_plate_crop(image_bgr)
    if dirty_white_crop is not None:
        return dirty_white_crop

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    search_y1 = int(h * 0.32)
    search_y2 = int(h * 0.86)
    roi = gray[search_y1:search_y2, :]

    proposals = []
    for threshold in (90, 110, 130, 150):
        mask = cv2.inRange(roi, threshold, 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 3))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            y += search_y1
            aspect = bw / max(bh, 1)
            area = bw * bh
            if not (45 <= bw <= 280 and 12 <= bh <= 75 and 2.0 <= aspect <= 8.5 and area >= 700):
                continue

            cx = x + bw / 2.0
            cy = y + bh / 2.0
            center_bonus = 1.0 - min(abs(cx - w / 2.0) / max(w / 2.0, 1), 1.0)
            lower_bonus = min(max((cy - h * 0.38) / max(h * 0.34, 1), 0.0), 1.0)
            aspect_bonus = 1.0 - min(abs(aspect - 4.5) / 4.5, 1.0)
            score = area * (1.0 + center_bonus * 0.9 + lower_bonus * 0.6 + aspect_bonus * 0.7)
            proposals.append((score, x, y, bw, bh))

    if not proposals:
        return None

    _, x, y, bw, bh = sorted(proposals, key=lambda item: item[0], reverse=True)[0]
    aspect = bw / max(bh, 1)
    adjusted_lower_plate = False
    if aspect < 3.0 and bh > 34:
        shift = int(bh * 0.82)
        y += shift
        bh = max(22, int(bh * 0.45))
        adjusted_lower_plate = True

    pad_x = int(bw * 0.18)
    pad_y = int(bh * (0.85 if adjusted_lower_plate else 0.28))
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + bw + pad_x)
    y2 = min(h, y + bh + pad_y)
    return image_bgr[y1:y2, x1:x2]


def is_plausible_plate_crop(crop_bgr):
    if crop_bgr is None or crop_bgr.size == 0:
        return False
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    white_plate_ratio = float(np.mean((hsv[:, :, 1] < 115) & (hsv[:, :, 2] > 95)))
    very_dark_ratio = float(np.mean(gray < 45))
    return white_plate_ratio >= 0.04 and very_dark_ratio <= 0.92


def find_dirty_white_plate_crop(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    mask = ((hsv[:, :, 1] < 105) & (hsv[:, :, 2] > 85)).astype(np.uint8) * 255
    mask[: int(h * 0.42), :] = 0
    mask[:, : int(w * 0.05)] = 0
    mask[:, int(w * 0.95) :] = 0
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    proposals = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        aspect = bw / max(bh, 1)
        area = bw * bh
        if not (55 <= bw <= 180 and 18 <= bh <= 70 and 1.45 <= aspect <= 5.2 and area >= 900):
            continue

        roi = gray[y : y + bh, x : x + bw]
        dark_ratio = float(np.mean(roi < 95))
        bright_ratio = float(np.mean(roi > 140))
        if dark_ratio < 0.34:
            continue

        lower_bonus = y / max(h, 1)
        aspect_bonus = 1.0 - min(abs(aspect - 2.3) / 2.3, 1.0)
        score = area * (1.0 + dark_ratio * 2.0 + bright_ratio * 0.6 + lower_bonus * 2.0 + aspect_bonus * 0.5)
        proposals.append((score, x, y, bw, bh))

    if not proposals:
        return None

    _, x, y, bw, bh = sorted(proposals, key=lambda item: item[0], reverse=True)[0]
    pad_x = max(8, int(bw * 0.18))
    pad_y = max(6, int(bh * 0.32))
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + bw + pad_x)
    y2 = min(h, y + bh + pad_y)
    return image_bgr[y1:y2, x1:x2]


def find_blue_strip_plate_crop(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    blue_mask = cv2.inRange(hsv, np.array([85, 30, 20]), np.array([145, 255, 255]))
    blue_mask[: int(h * 0.58), :] = 0
    blue_mask[:, : int(w * 0.12)] = 0
    blue_mask[:, int(w * 0.95) :] = 0

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(blue_mask, 8)
    proposals = []
    for idx in range(1, num_labels):
        x, y, bw, bh, area = stats[idx]
        if not (int(h * 0.58) <= y <= h - 20 and 5 <= bw <= 60 and 10 <= bh <= 52 and 25 <= area <= 1600):
            continue
        aspect = bw / max(bh, 1)
        if not (0.15 <= aspect <= 1.75):
            continue

        cx, cy = centroids[idx]
        plate_h = max(34, int(bh * 2.15))
        plate_w = int(plate_h * 5.6)
        x1 = int(cx - bw * 0.9)
        y1 = int(cy - plate_h * 0.50)
        x2 = int(x1 + plate_w)
        y2 = int(y1 + plate_h)

        if x2 > w:
            overflow = x2 - w
            x1 -= overflow
            x2 -= overflow

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        crop_w = x2 - x1
        crop_h = y2 - y1
        if crop_w < 90 or crop_h < 26:
            continue

        lower_bonus = y / max(h, 1)
        center_bonus = 1.0 - min(abs((x1 + crop_w / 2) - w / 2) / max(w / 2, 1), 1.0)
        stripe_bonus = bh * 12.0 + area
        score = stripe_bonus * (1.0 + lower_bonus * 0.8 + center_bonus * 0.4)
        proposals.append((score, x1, y1, x2, y2))

    if not proposals:
        return None

    _, x1, y1, x2, y2 = sorted(proposals, key=lambda item: item[0], reverse=True)[0]
    return image_bgr[y1:y2, x1:x2]


def refine_inner_plate_crop(crop_bgr):
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    blue_mask = cv2.inRange(hsv, np.array([85, 30, 20]), np.array([145, 255, 255]))
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(blue_mask, 8)
    blue_proposals = []
    for idx in range(1, num_labels):
        x, y, bw, bh, area = stats[idx]
        aspect = bw / max(bh, 1)
        if 15 <= bw <= 65 and 12 <= bh <= 52 and 0.4 <= aspect <= 1.9 and 60 <= area <= 1600:
            cx, cy = centroids[idx]
            plate_h = max(36, int(bh * 1.9))
            x1 = int(cx - bw * 0.85)
            y1 = int(cy - plate_h * 0.55)
            x2 = int(x1 + plate_h * 6.2)
            y2 = int(y1 + plate_h)
            if x2 > w:
                overflow = x2 - w
                x1 -= overflow
                x2 -= overflow
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            if x2 - x1 >= 80 and y2 - y1 >= 24:
                blue_proposals.append((area + bh * 20, x1, y1, x2, y2))

    if blue_proposals:
        _, x1, y1, x2, y2 = sorted(blue_proposals, key=lambda item: item[0], reverse=True)[0]
        return crop_bgr[y1:y2, x1:x2]

    white_mask = ((hsv[:, :, 1] < 95) & (hsv[:, :, 2] > 110)).astype(np.uint8) * 255
    white_mask[: int(h * 0.15), :] = 0
    closed = cv2.morphologyEx(
        white_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)),
    )
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    proposals = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        aspect = bw / max(bh, 1)
        area = bw * bh
        if 18 <= bw <= w * 0.75 and 8 <= bh <= h * 0.45 and 1.2 <= aspect <= 5.8 and area >= 140:
            center_bonus = 1.0 - min(abs((x + bw / 2) - w / 2) / max(w / 2, 1), 1.0)
            proposals.append((area * (1.0 + center_bonus), x, y, bw, bh))

    if not proposals:
        return None

    _, x, y, bw, bh = sorted(proposals, key=lambda item: item[0], reverse=True)[0]
    pad_x = max(6, int(bw * 0.25))
    pad_y = max(5, int(bh * 0.65))
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + bw + pad_x)
    y2 = min(h, y + bh + pad_y)
    refined = crop_bgr[y1:y2, x1:x2]
    if refined.shape[0] < 12 or refined.shape[1] < 24:
        return None
    return refined


def render_page(result: Optional[Dict[str, Any]] = None, error: str = "") -> bytes:
    result_html = ""
    if error:
        result_html = f"<section class='alert'>{html.escape(error)}</section>"
    elif result is not None:
        cards = []
        for title, image in result["stages"].items():
            cards.append(
                "<article class='card'>"
                f"<h3>{html.escape(title)}</h3>"
                f"<img src='{image_to_data_uri(image)}' alt='{html.escape(title)}'>"
                "</article>"
            )
        result_html = (
            "<section class='result'>"
            f"<div class='plate'>{html.escape(str(result['plate']))}</div>"
            f"<pre>{html.escape(str(result['details']))}</pre>"
            "</section>"
            f"<section class='grid'>{''.join(cards)}</section>"
        )

    page = f"""<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Plaka Tanima Arayuzu</title>
  <style>
    body {{ margin: 0; font-family: Segoe UI, Arial, sans-serif; background: #f5f7fb; color: #111827; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
    header {{ display: flex; align-items: center; justify-content: space-between; gap: 16px; margin-bottom: 18px; }}
    h1 {{ font-size: 24px; margin: 0; }}
    form {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
    input, button {{ font: inherit; }}
    button {{ border: 0; background: #0f766e; color: white; padding: 9px 14px; border-radius: 6px; cursor: pointer; }}
    .hint {{ color: #4b5563; margin: 0 0 18px; }}
    .result, .alert {{ background: white; border: 1px solid #d1d5db; border-radius: 8px; padding: 16px; margin-bottom: 18px; }}
    .alert {{ color: #991b1b; }}
    .plate {{ font-size: 40px; font-weight: 800; color: #0f766e; margin-bottom: 10px; }}
    pre {{ white-space: pre-wrap; font-family: Consolas, monospace; font-size: 14px; margin: 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; }}
    .card {{ background: white; border: 1px solid #d1d5db; border-radius: 8px; padding: 12px; }}
    .card h3 {{ font-size: 15px; margin: 0 0 8px; }}
    img {{ width: 100%; max-height: 260px; object-fit: contain; background: #fff; }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Plaka Tanima Arayuzu</h1>
      <form method="post" enctype="multipart/form-data">
        <input type="file" name="image" accept="image/*" required>
        <button type="submit">Plakayi Oku</button>
      </form>
    </header>
    <p class="hint">Akis: YOLO -> Crop -> CLAHE -> Contrast adjustment -> Sharpening -> Noise cleaning -> OCR preprocessing -> OCR</p>
    {result_html}
  </main>
</body>
</html>"""
    return page.encode("utf-8")


class PlateWebHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self._send_html(render_page())

    def do_POST(self) -> None:
        try:
            form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST"})
            item = form["image"] if "image" in form else None
            if item is None or not item.filename:
                self._send_html(render_page(error="Resim secilmedi."))
                return

            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            suffix = Path(item.filename).suffix or ".png"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=UPLOAD_DIR) as tmp:
                tmp.write(item.file.read())
                upload_path = Path(tmp.name)

            result = run_plate_pipeline(upload_path)
            self._send_html(render_page(result=result))
        except Exception as exc:
            self._send_html(render_page(error=str(exc)))

    def _send_html(self, content: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 7860), PlateWebHandler)
    print("Web arayuz hazir: http://127.0.0.1:7860")
    server.serve_forever()


if __name__ == "__main__":
    main()
