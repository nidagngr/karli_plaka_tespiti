from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

from gan_restoration import build_restorer_from_config
from ocr_utils import (
    choose_best_supported_plate_candidate,
    format_supported_plate_candidates,
    read_plate_candidates_from_stages,
)
from preprocess_utils import LetterboxInfo, preprocess_for_plate_ui
from preprocess_utils import (
    adjust_contrast,
    apply_clahe,
    clean_noise,
    dark_text_emphasis,
    ocr_preprocess,
    rescue_snow_plate,
    sharpen_image,
    suppress_bright_snow,
)
from utils import add_padding_to_box, ensure_dir, load_yaml, read_image, resolve_project_root, to_absolute, write_image


class PlateRecognitionApp:
    def __init__(self, root_window: tk.Tk) -> None:
        self.root_window = root_window
        self.project_root = resolve_project_root()
        self.pipeline_cfg = load_yaml(self.project_root / "configs" / "pipeline_config.yaml")
        self.yolo_cfg = load_yaml(self.project_root / "configs" / "yolo_config.yaml")
        self.target_size = int(self.yolo_cfg["train"]["imgsz"])

        self.original_image = None
        self.preprocessed_images: Dict[str, object] = {}
        self.letterbox_info: Optional[LetterboxInfo] = None
        self.selected_path: Optional[Path] = None
        self.yolo_model = None
        self.ocr_reader = None
        self.restorer = None
        self.restorer_loaded = False
        self.is_running = False
        self.photo_refs = []

        self.root_window.title("Plaka Tanima Arayuzu")
        self.root_window.geometry("1220x820")
        self.root_window.minsize(1040, 720)
        self.root_window.configure(bg="#f5f7fb")

        self._build_ui()

    def _build_ui(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#f5f7fb")
        style.configure("TLabel", background="#f5f7fb", foreground="#111827", font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 17, "bold"))
        style.configure("Meta.TLabel", foreground="#4b5563")
        style.configure("Result.TLabel", font=("Segoe UI", 26, "bold"), foreground="#0f766e")
        style.configure("TButton", font=("Segoe UI", 10), padding=(12, 7))

        outer = ttk.Frame(self.root_window, padding=18)
        outer.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(outer)
        header.pack(fill=tk.X)
        ttk.Label(header, text="Plaka Tanima Arayuzu", style="Title.TLabel").pack(side=tk.LEFT)

        actions = ttk.Frame(header)
        actions.pack(side=tk.RIGHT)
        ttk.Button(actions, text="Resim Yukle", command=self.load_image).pack(side=tk.LEFT, padx=(0, 8))
        self.read_button = ttk.Button(actions, text="Plakayi Oku", command=self.start_recognition)
        self.read_button.pack(side=tk.LEFT)

        info_frame = ttk.Frame(outer)
        info_frame.pack(fill=tk.X, pady=(12, 10))
        self.file_label = ttk.Label(info_frame, text="Henuz resim yuklenmedi.", style="Meta.TLabel")
        self.file_label.pack(side=tk.LEFT)
        self.size_label = ttk.Label(info_frame, text=f"Model giris boyutu: {self.target_size}x{self.target_size}", style="Meta.TLabel")
        self.size_label.pack(side=tk.RIGHT)

        result_frame = ttk.Frame(outer, padding=(0, 6))
        result_frame.pack(fill=tk.X)
        ttk.Label(result_frame, text="Sonuc:").pack(side=tk.LEFT)
        self.result_label = ttk.Label(result_frame, text="-", style="Result.TLabel")
        self.result_label.pack(side=tk.LEFT, padx=(12, 0))
        self.status_label = ttk.Label(result_frame, text="", style="Meta.TLabel")
        self.status_label.pack(side=tk.RIGHT)

        main = ttk.PanedWindow(outer, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        left = ttk.Frame(main)
        right = ttk.Frame(main)
        main.add(left, weight=3)
        main.add(right, weight=2)

        self.canvas = tk.Canvas(left, bg="#ffffff", highlightthickness=1, highlightbackground="#d1d5db")
        scrollbar = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.canvas.yview)
        self.gallery = ttk.Frame(self.canvas)
        self.gallery.bind("<Configure>", lambda event: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas_window = self.canvas.create_window((0, 0), window=self.gallery, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.bind("<Configure>", self._resize_canvas_window)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        right_top = ttk.Frame(right)
        right_top.pack(fill=tk.BOTH, expand=True)
        ttk.Label(right_top, text="Plaka kirpimi / tespit").pack(anchor="w")
        self.crop_label = ttk.Label(right_top)
        self.crop_label.pack(fill=tk.BOTH, expand=True, pady=(8, 12))

        ttk.Label(right, text="Aday analizi").pack(anchor="w")
        self.detail_text = tk.Text(right, height=13, wrap=tk.WORD, font=("Consolas", 10), bg="#ffffff", fg="#111827")
        self.detail_text.pack(fill=tk.X)
        self.detail_text.insert(tk.END, "YOLO -> Crop -> CLAHE -> Contrast -> Sharpening -> Noise cleaning -> OCR preprocessing -> OCR sonucu burada gorunecek.")
        self.detail_text.configure(state=tk.DISABLED)

    def _resize_canvas_window(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self.canvas_window, width=event.width)

    def load_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Plaka resmi sec",
            filetypes=[
                ("Resim dosyalari", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"),
                ("Tum dosyalar", "*.*"),
            ],
        )
        if not path:
            return

        try:
            self.selected_path = Path(path)
            self.original_image = read_image(self.selected_path)
            self.preprocessed_images, self.letterbox_info = preprocess_for_plate_ui(self.original_image, self.target_size)
        except Exception as exc:
            messagebox.showerror("Resim okunamadi", str(exc))
            return

        h, w = self.original_image.shape[:2]
        self.file_label.configure(text=str(self.selected_path))
        self.size_label.configure(
            text=f"Orijinal: {w}x{h}  ->  Model: {self.target_size}x{self.target_size}"
        )
        self.result_label.configure(text="-")
        self.status_label.configure(text="On islemler hazir.")
        self._clear_details()
        self._render_gallery()
        self._set_crop_image(None)

    def start_recognition(self) -> None:
        if self.is_running:
            return
        if self.original_image is None or not self.preprocessed_images:
            messagebox.showwarning("Resim gerekli", "Once bir resim yukleyin.")
            return

        self.is_running = True
        self.read_button.configure(state=tk.DISABLED)
        self.status_label.configure(text="YOLO baslatiliyor...")
        self.result_label.configure(text="Okunuyor")
        thread = threading.Thread(target=self._recognize_worker, daemon=True)
        thread.start()

    def _set_status_async(self, text: str) -> None:
        self.root_window.after(0, lambda: self.status_label.configure(text=text))

    def _recognize_worker(self) -> None:
        try:
            result = self._recognize_plate()
        except Exception as exc:
            self.root_window.after(0, lambda: self._show_error(exc))
            return
        self.root_window.after(0, lambda: self._show_result(result))

    def _recognize_plate(self) -> Dict[str, object]:
        from ultralytics import YOLO

        if self.yolo_model is None:
            self._set_status_async("YOLO modeli yukleniyor...")
            weights = to_absolute(self.project_root, self.pipeline_cfg["paths"]["yolo_weights"])
            if not weights.exists():
                raise FileNotFoundError(f"YOLO agirlik dosyasi bulunamadi: {weights}")
            self.yolo_model = YOLO(str(weights))

        model_image = self.preprocessed_images["01 YOLO model girisi"]
        yolo_cfg = self.pipeline_cfg["yolo"]
        self._set_status_async("YOLO plaka ariyor...")
        prediction = self.yolo_model.predict(
            source=model_image,
            conf=float(yolo_cfg["conf_threshold"]),
            iou=float(yolo_cfg["iou_threshold"]),
            max_det=int(yolo_cfg["max_det"]),
            verbose=False,
        )[0]
        if prediction.boxes is None or len(prediction.boxes) == 0:
            return {
                "plate": "",
                "crop": None,
                "details": "\n".join(
                    [
                        "Plaka tespit edilemedi.",
                        "",
                        "Akis: YOLO -> Crop -> CLAHE -> Contrast Adjustment -> Sharpening -> Noise cleaning -> OCR preprocessing -> OCR",
                        "YOLO kutu bulamadigi icin OCR asamasina gecilmedi.",
                    ]
                ),
            }

        box = sorted(prediction.boxes, key=lambda item: float(item.conf[0]), reverse=True)[0]
        h, w = model_image.shape[:2]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = add_padding_to_box(
            int(x1), int(y1), int(x2), int(y2), float(yolo_cfg["crop_padding"]), w, h
        )
        crop = model_image[y1:y2, x1:x2]
        return self._ocr_crop(crop, yolo_confidence=float(box.conf[0]), source_label="YOLO")

    def _trim_neutral_padding(self, image_bgr):
        if image_bgr.size == 0:
            return image_bgr
        diff = np.abs(image_bgr.astype(np.int16) - 114)
        non_padding = np.any(diff > 18, axis=2)
        rows = np.where(np.any(non_padding, axis=1))[0]
        cols = np.where(np.any(non_padding, axis=0))[0]
        if len(rows) == 0 or len(cols) == 0:
            return image_bgr
        y1, y2 = int(rows[0]), int(rows[-1]) + 1
        x1, x2 = int(cols[0]), int(cols[-1]) + 1
        return image_bgr[y1:y2, x1:x2]

    def _get_restorer(self):
        if not bool(self.pipeline_cfg.get("gan", {}).get("enabled", False)):
            return None
        if not self.restorer_loaded:
            self.restorer_loaded = True
            try:
                self.restorer = build_restorer_from_config(self.project_root, self.pipeline_cfg)
            except Exception:
                self.restorer = None
        return self.restorer

    def _ocr_crop(self, crop, yolo_confidence: Optional[float] = None, source_label: str = "YOLO") -> Dict[str, object]:
        from easyocr import Reader

        crop = self._trim_neutral_padding(crop)
        ocr_cfg = self.pipeline_cfg["ocr"]
        if self.ocr_reader is None:
            self._set_status_async("OCR modeli yukleniyor...")
            self.ocr_reader = Reader(ocr_cfg["languages"], gpu=bool(ocr_cfg["gpu"]))

        self._set_status_async("On isleme uygulanıyor...")
        clahe = apply_clahe(crop)
        contrast = adjust_contrast(clahe)
        sharpened = sharpen_image(contrast)
        cleaned = clean_noise(sharpened)
        snow_suppressed = suppress_bright_snow(clahe)
        dark_text = dark_text_emphasis(snow_suppressed)
        rescue_plate = rescue_snow_plate(crop)
        ocr_ready = ocr_preprocess(snow_suppressed)
        ocr_image = ocr_ready
        stages = {
            "01 YOLO model girisi": self.preprocessed_images["01 YOLO model girisi"],
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
        gan_status = "GAN Restoration: kapali"

        if bool(self.pipeline_cfg.get("gan", {}).get("enabled", False)):
            self._set_status_async("GAN restoration uygulanıyor...")
            restorer = self._get_restorer()
            if restorer is None:
                gan_status = "GAN Restoration: yuklenemedi veya agirlik yok"
            else:
                ocr_image = restorer.restore(ocr_ready)
                stages["08 GAN Restoration"] = ocr_image
                gan_status = "GAN Restoration: aktif"

        self._set_status_async("OCR okunuyor...")
        min_confidence = float(ocr_cfg.get("min_confidence", 0.0))
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
            self.ocr_reader,
            ocr_sources,
            str(ocr_cfg["allowlist"]),
            min_confidence,
        )
        best = choose_best_supported_plate_candidate(candidates, beam_width=1)

        out_dir = ensure_dir(self.project_root / "outputs" / "ui")
        if self.selected_path is not None:
            write_image(out_dir / f"{self.selected_path.stem}_model_input.png", self.preprocessed_images["01 YOLO model girisi"])
            write_image(out_dir / f"{self.selected_path.stem}_plate_crop.png", crop)
            write_image(out_dir / f"{self.selected_path.stem}_ocr_ready.png", ocr_image)

        raw = " | ".join(candidate.raw_text for candidate in candidates)
        ranked = format_supported_plate_candidates(candidates, beam_width=1, limit=8)
        confidence_line = (
            f"YOLO confidence: {yolo_confidence:.4f}"
            if yolo_confidence is not None
            else "YOLO confidence: yok"
        )
        details = "\n".join(
            [
                confidence_line,
                f"Crop kaynagi: {source_label}",
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
        return {"plate": best.normalized, "crop": crop, "details": details, "stages": stages}

    def _show_error(self, exc: Exception) -> None:
        self.result_label.configure(text="-")
        self.status_label.configure(text="Hata olustu.")
        self.is_running = False
        self.read_button.configure(state=tk.NORMAL)
        messagebox.showerror("Islem tamamlanamadi", str(exc))

    def _show_result(self, result: Dict[str, object]) -> None:
        plate = str(result.get("plate") or "-")
        self.result_label.configure(text=plate)
        self.status_label.configure(text="Tamamlandi.")
        self.is_running = False
        self.read_button.configure(state=tk.NORMAL)
        self._set_crop_image(result.get("crop"))
        stages = result.get("stages")
        if isinstance(stages, dict):
            self.preprocessed_images = stages
            self._render_gallery()
        self._set_details(str(result.get("details") or ""))

    def _render_gallery(self) -> None:
        for child in self.gallery.winfo_children():
            child.destroy()
        self.photo_refs = []

        for index, (title, image_bgr) in enumerate(self.preprocessed_images.items()):
            cell = ttk.Frame(self.gallery, padding=8)
            cell.grid(row=index // 2, column=index % 2, sticky="nsew")
            ttk.Label(cell, text=title).pack(anchor="w", pady=(0, 4))
            label = ttk.Label(cell)
            label.pack(fill=tk.BOTH, expand=True)
            self._assign_image(label, image_bgr, max_size=(440, 245))

        self.gallery.columnconfigure(0, weight=1)
        self.gallery.columnconfigure(1, weight=1)

    def _assign_image(self, label: ttk.Label, image_bgr, max_size: tuple[int, int]) -> None:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        pil_image.thumbnail(max_size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(pil_image)
        label.configure(image=photo)
        self.photo_refs.append(photo)

    def _set_crop_image(self, image_bgr) -> None:
        self.crop_label.configure(image="")
        if image_bgr is None:
            self.crop_label.configure(text="Plaka tespitinden sonra burada gorunecek.")
            return
        self.crop_label.configure(text="")
        self._assign_image(self.crop_label, image_bgr, max_size=(430, 260))

    def _clear_details(self) -> None:
        self._set_details("YOLO -> Crop -> CLAHE -> Contrast -> Sharpening -> Noise cleaning -> OCR preprocessing -> OCR sonucu burada gorunecek.")

    def _set_details(self, text: str) -> None:
        self.detail_text.configure(state=tk.NORMAL)
        self.detail_text.delete("1.0", tk.END)
        self.detail_text.insert(tk.END, text)
        self.detail_text.configure(state=tk.DISABLED)


def main() -> None:
    root_window = tk.Tk()
    PlateRecognitionApp(root_window)
    root_window.mainloop()


if __name__ == "__main__":
    main()
