import re
from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from utils import normalize_plate_text

TURKISH_PLATE_RE = re.compile(r"^(0[1-9]|[1-7][0-9]|8[01])[A-Z]{1,3}[0-9]{2,4}$")
TURKISH_PLATE_SUBSTR_RE = re.compile(r"(0[1-9]|[1-7][0-9]|8[01])[A-Z]{1,3}[0-9]{2,4}")
DIGIT_MAP = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6", "T": "7"}
LETTER_MAP = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "7": "T", "8": "B"}
PLATE_LETTER_OPTIONS = {
    "0": ["D", "O"],
    "1": ["I", "L"],
    "2": ["Z"],
    "5": ["S"],
    "6": ["G"],
    "7": ["T"],
    "8": ["B"],
}
DIGITS = "0123456789"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CONTEXT_DIGIT_OPTIONS = {
    "B": ["8", "3"],
    "L": ["1", "4"],
    "I": ["1"],
    "O": ["0"],
    "Q": ["0"],
    "D": ["0"],
    "Z": ["2"],
    "S": ["5"],
    "G": ["6"],
    "T": ["7"],
}
AMBIGUOUS_DIGIT_OPTIONS = {
    "3": ["3", "5"],
}


@dataclass(frozen=True)
class PlateCandidate:
    text: str
    confidence: float = 0.0
    source: str = "ocr"
    raw_text: str = ""

    @property
    def normalized(self) -> str:
        return strip_plate_prefix(self.text)

    @property
    def regex_ok(self) -> bool:
        return bool(TURKISH_PLATE_RE.fullmatch(self.normalized))

    @property
    def score(self) -> float:
        return plate_candidate_score(self.text, self.confidence)


def strip_plate_prefix(text: str) -> str:
    normalized = normalize_plate_text(text)
    if normalized.startswith("TR") and len(normalized) > 2:
        return normalized[2:]
    return normalized


def upscale_for_ocr(image: np.ndarray, min_width: int = 320) -> np.ndarray:
    h, w = image.shape[:2]
    if w >= min_width:
        return image
    scale = float(min_width) / max(w, 1)
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def remove_vertical_snow_streaks(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    bright = cv2.inRange(gray, 185, 255)
    kernel_h = max(9, image_bgr.shape[0] // 3)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_h))
    streak_mask = cv2.morphologyEx(bright, cv2.MORPH_OPEN, vertical_kernel)
    streak_mask = cv2.dilate(streak_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    if int(streak_mask.sum()) == 0:
        return image_bgr
    return cv2.inpaint(image_bgr, streak_mask, 3, cv2.INPAINT_TELEA)


def enhance_small_plate_for_ocr(image_bgr: np.ndarray, target_width: int) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    scale = float(target_width) / max(w, 1)
    up = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    cleaned = remove_vertical_snow_streaks(up)
    lab = cv2.cvtColor(cleaned, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_channel = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(4, 4)).apply(l_channel)
    enhanced = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    return cv2.addWeighted(enhanced, 1.65, blurred, -0.65, 0)


def build_ocr_variants(image_bgr: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    base = upscale_for_ocr(image_bgr)
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    return [
        ("ocr_ready", base),
        ("gray", cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)),
    ]


def plate_candidate_score(text: str, confidence: float = 0.0) -> float:
    normalized = strip_plate_prefix(text)
    if not normalized:
        return -10.0

    score = float(confidence) * 8.0
    if TURKISH_PLATE_RE.fullmatch(normalized):
        score += 20.0

    length = len(normalized)
    if 7 <= length <= 8:
        score += 5.0
    else:
        score -= abs(length - 7.5)

    digit_count = sum(ch.isdigit() for ch in normalized)
    alpha_count = sum(ch.isalpha() for ch in normalized)
    score += min(digit_count, 6) * 0.5
    score += min(alpha_count, 3) * 0.75

    if len(normalized) >= 2 and normalized[:2].isdigit():
        city = int(normalized[:2])
        if 1 <= city <= 81:
            score += 4.0

    if alpha_count == 0 or digit_count == 0:
        score -= 4.0

    bad_substrings = ["TOT", "TOI", "TIT", "III", "OOO"]
    if any(token in normalized for token in bad_substrings):
        score -= 3.0

    return score


def _candidate_sort_key(candidate: PlateCandidate) -> Tuple[float, int, int, float]:
    source_priority = {
        "base": 5,
        "gray": 5,
        "denoised": 5,
        "otsu": 5,
        "adaptive": 5,
        "regex_repair": 4,
        "beam_search": 3,
    }
    priority = source_priority.get(candidate.source.split(":")[0], 1)
    return (candidate.score, int(candidate.regex_ok), priority, candidate.confidence)


def extract_plausible_plate(text: str) -> str:
    normalized = strip_plate_prefix(text)
    if not normalized:
        return ""

    matches = TURKISH_PLATE_SUBSTR_RE.findall(normalized)
    if matches:
        # findall with a capturing group only returns the city code; use finditer for the full match
        full_matches = [m.group(0) for m in TURKISH_PLATE_SUBSTR_RE.finditer(normalized)]
        full_matches = sorted(full_matches, key=lambda value: (plate_candidate_score(value), -len(value)), reverse=True)
        return full_matches[0]

    best = normalized
    best_score = plate_candidate_score(normalized)
    for start in range(len(normalized)):
        for end in range(start + 6, min(len(normalized), start + 9) + 1):
            candidate = normalized[start:end]
            for repaired in generate_plate_repairs(candidate):
                score = plate_candidate_score(repaired)
                if score > best_score:
                    best = repaired
                    best_score = score
    return best


def easyocr_result_to_candidate(result: Sequence[Any], variant_name: str) -> PlateCandidate:
    if not result:
        return PlateCandidate("", 0.0, variant_name, "")

    items = []
    for item in result:
        if isinstance(item, str):
            items.append((0.0, item, 0.0))
            continue

        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue

        bbox, text = item[0], item[1]
        confidence = float(item[2]) if len(item) > 2 and item[2] is not None else 0.0
        try:
            x_left = min(point[0] for point in bbox)
        except Exception:
            x_left = 0.0
        items.append((float(x_left), str(text), confidence))

    if not items:
        return PlateCandidate("", 0.0, variant_name, "")

    items.sort(key=lambda value: value[0])
    raw_text = "".join(text for _, text, _ in items)
    confidences = [confidence for _, _, confidence in items if confidence > 0]
    confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return PlateCandidate(extract_plausible_plate(raw_text), confidence, variant_name, raw_text)


def read_plate_candidates(reader: Any, image_bgr: np.ndarray, allowlist: str, min_confidence: float = 0.0) -> List[PlateCandidate]:
    candidates: List[PlateCandidate] = []
    for variant_name, variant in build_ocr_variants(image_bgr):
        result = reader.readtext(variant, detail=1, paragraph=False, allowlist=allowlist)
        candidate = easyocr_result_to_candidate(result, variant_name)
        if candidate.text and candidate.confidence >= min_confidence:
            candidates.append(candidate)
        elif candidate.text:
            candidates.append(PlateCandidate(candidate.text, candidate.confidence, f"{variant_name}:low_conf", candidate.raw_text))
    return candidates


def read_plate_candidates_from_stages(
    reader: Any,
    stages: Iterable[Tuple[str, np.ndarray]],
    allowlist: str,
    min_confidence: float = 0.0,
) -> List[PlateCandidate]:
    candidates: List[PlateCandidate] = []
    for stage_name, image_bgr in stages:
        for candidate in read_plate_candidates(reader, image_bgr, allowlist, min_confidence):
            candidates.append(
                PlateCandidate(
                    candidate.text,
                    candidate.confidence,
                    f"{stage_name}/{candidate.source}",
                    candidate.raw_text,
                )
            )
    return candidates


def read_degraded_plate_candidates(reader: Any, image_bgr: np.ndarray, allowlist: str) -> List[PlateCandidate]:
    candidates: List[PlateCandidate] = []
    h, w = image_bgr.shape[:2]
    regions = [
        ("lower", image_bgr[int(h * 0.25) : h, :]),
        ("tight", image_bgr[int(h * 0.30) : h, int(w * 0.02) : int(w * 0.92)]),
    ]

    for region_name, region in regions:
        if region.size == 0:
            continue
        for scale in (3, 5):
            up = cv2.resize(region, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
            denoised = cv2.bilateralFilter(clahe, 7, 60, 60)
            blurred = cv2.GaussianBlur(denoised, (0, 0), 1.0)
            sharp = cv2.addWeighted(denoised, 1.75, blurred, -0.75, 0)
            _, otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            adaptive = cv2.adaptiveThreshold(
                sharp,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                5,
            )

            variants = [
                ("sharp", cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)),
                ("adaptive", cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)),
            ]

            for variant_name, variant in variants:
                try:
                    result = reader.readtext(
                        variant,
                        detail=1,
                        paragraph=False,
                        allowlist=allowlist,
                        mag_ratio=2,
                        canvas_size=2560,
                        text_threshold=0.05,
                        low_text=0.03,
                        link_threshold=0.03,
                        width_ths=2.0,
                        add_margin=0.2,
                    )
                except Exception:
                    continue

                candidate = easyocr_result_to_candidate(result, f"degraded/{region_name}_{scale}_{variant_name}")
                if candidate.text:
                    normalized = candidate.normalized
                    raw = normalize_plate_text(candidate.raw_text)
                    if 7 <= len(normalized) <= 9 and len(raw) <= 14 and candidate.confidence >= 0.45:
                        candidates.append(candidate)

    return candidates


def _char_options(ch: str, expected: str) -> List[Tuple[str, float]]:
    normalized = normalize_plate_text(ch)
    ch = normalized[0] if normalized else ""
    options = []

    if expected == "digit":
        if ch.isdigit():
            options.append((ch, 1.0))
        mapped = DIGIT_MAP.get(ch)
        if mapped:
            options.append((mapped, 0.82))
        for alt in DIGITS:
            options.append((alt, 0.1 if alt != ch else 0.5))
    else:
        if ch.isalpha():
            options.append((ch, 1.0))
        mapped = LETTER_MAP.get(ch)
        if mapped:
            options.append((mapped, 0.82))
        for alt in LETTERS:
            options.append((alt, 0.06 if alt != ch else 0.5))

    dedup = {}
    for value, weight in options:
        if value not in dedup or weight > dedup[value]:
            dedup[value] = weight
    return sorted(dedup.items(), key=lambda item: item[1], reverse=True)[:4]


def beam_search_plate_candidates(text: str, confidence: float = 0.0, beam_width: int = 8) -> List[PlateCandidate]:
    normalized = strip_plate_prefix(text)
    if not normalized:
        return []

    candidates = {extract_plausible_plate(normalized)}
    windows = {normalized}
    for start in range(len(normalized)):
        for end in range(start + 5, min(len(normalized), start + 10) + 1):
            windows.add(normalized[start:end])

    for window in windows:
        for alpha_len in (1, 2, 3):
            for tail_digits in (2, 3, 4):
                pattern = ["digit", "digit"] + ["letter"] * alpha_len + ["digit"] * tail_digits
                if len(window) != len(pattern):
                    continue

                beams: List[Tuple[str, float]] = [("", 0.0)]
                for ch, expected in zip(window, pattern):
                    next_beams = []
                    for prefix, score in beams:
                        for replacement, weight in _char_options(ch, expected):
                            next_beams.append((prefix + replacement, score + weight))
                    beams = sorted(next_beams, key=lambda item: item[1], reverse=True)[:beam_width]

                for candidate_text, _ in beams:
                    if TURKISH_PLATE_RE.fullmatch(candidate_text):
                        candidates.add(candidate_text)

    return [
        PlateCandidate(candidate, confidence, "beam_search", normalized)
        for candidate in candidates
        if candidate
    ]


def generate_plate_repairs(text: str) -> List[str]:
    normalized = strip_plate_prefix(text)
    repairs = {normalized}

    for alpha_len in (1, 2, 3):
        for tail_digits in (2, 3, 4):
            total_len = 2 + alpha_len + tail_digits
            if len(normalized) != total_len:
                continue

            city_options = [_digit_repair_options(ch) for ch in normalized[:2]]
            letter_options = [_letter_repair_options(ch) for ch in normalized[2 : 2 + alpha_len]]
            tail_options = [_digit_repair_options(ch) for ch in normalized[2 + alpha_len :]]
            for values in product(*(city_options + letter_options + tail_options)):
                repairs.add("".join(values))

    return list(repairs)


def _digit_repair_options(ch: str) -> List[str]:
    if ch.isdigit():
        return AMBIGUOUS_DIGIT_OPTIONS.get(ch, [ch])
    return CONTEXT_DIGIT_OPTIONS.get(ch, [DIGIT_MAP.get(ch, ch)])


def _letter_repair_options(ch: str) -> List[str]:
    if ch.isalpha():
        return [ch]
    if ch in PLATE_LETTER_OPTIONS:
        return PLATE_LETTER_OPTIONS[ch]
    return [LETTER_MAP.get(ch, ch)]


def expand_plate_candidates(candidates: Iterable[str | PlateCandidate], beam_width: int = 8) -> List[PlateCandidate]:
    expanded: List[PlateCandidate] = []
    for item in candidates:
        if isinstance(item, PlateCandidate):
            base = item
        else:
            base = PlateCandidate(str(item), 0.0, "ocr", str(item))

        plausible = extract_plausible_plate(base.text or base.raw_text)
        if plausible:
            expanded.append(PlateCandidate(plausible, base.confidence, base.source, base.raw_text or base.text))

        repair_sources = [plausible, base.text, base.raw_text]
        for repair_source in repair_sources:
            if not repair_source:
                continue
            for repaired in generate_plate_repairs(repair_source):
                if repaired:
                    expanded.append(PlateCandidate(repaired, base.confidence, f"{base.source}:regex_repair", base.raw_text or base.text))

        if beam_width > 1:
            expanded.extend(beam_search_plate_candidates(base.raw_text or base.text, base.confidence, beam_width=beam_width))

    unique = {}
    for candidate in expanded:
        key = candidate.normalized
        if not key:
            continue
        if key not in unique or _candidate_sort_key(candidate) > _candidate_sort_key(unique[key]):
            unique[key] = candidate
    return sorted(unique.values(), key=_candidate_sort_key, reverse=True)


def choose_best_plate_candidate(candidates: Iterable[str | PlateCandidate], beam_width: int = 8) -> PlateCandidate:
    expanded = expand_plate_candidates(candidates, beam_width=beam_width)
    if not expanded:
        return PlateCandidate("", 0.0, "none", "")
    return expanded[0]


def choose_best_supported_plate_candidate(candidates: Iterable[str | PlateCandidate], beam_width: int = 1) -> PlateCandidate:
    source_candidates = list(candidates)
    expanded = expand_plate_candidates(source_candidates, beam_width=beam_width)
    if not expanded:
        return PlateCandidate("", 0.0, "none", "")

    raw_texts = _raw_candidate_texts(source_candidates)
    preferred_length = _preferred_plate_length(raw_texts)
    return sorted(
        expanded,
        key=lambda item: (
            int(item.regex_ok),
            int(len(item.normalized) == preferred_length),
            -abs(len(item.normalized) - preferred_length),
            _exact_city_variant_priority(item, expanded),
            _city_prefix_support(item, raw_texts),
            hamming_support_score(item, raw_texts, expanded),
            int(not _is_repaired_candidate(item)),
            item.score,
            item.confidence,
        ),
        reverse=True,
    )[0]


def choose_best_plate_text(candidates: Iterable[str | PlateCandidate], beam_width: int = 8) -> str:
    best = choose_best_plate_candidate(candidates, beam_width=beam_width)
    return best.normalized


def format_plate_candidates(candidates: Iterable[str | PlateCandidate], beam_width: int = 8, limit: int = 8) -> str:
    expanded = expand_plate_candidates(candidates, beam_width=beam_width)[:limit]
    if not expanded:
        return ""
    return " | ".join(
        f"{candidate.normalized}:{candidate.confidence:.2f}:{candidate.source}:score={candidate.score:.2f}"
        for candidate in expanded
    )


def format_supported_plate_candidates(candidates: Iterable[str | PlateCandidate], beam_width: int = 1, limit: int = 8) -> str:
    source_candidates = list(candidates)
    expanded = expand_plate_candidates(source_candidates, beam_width=beam_width)
    if not expanded:
        return ""

    raw_texts = _raw_candidate_texts(source_candidates)
    preferred_length = _preferred_plate_length(raw_texts)
    ranked = sorted(
        expanded,
        key=lambda item: (
            int(item.regex_ok),
            int(len(item.normalized) == preferred_length),
            -abs(len(item.normalized) - preferred_length),
            _exact_city_variant_priority(item, expanded),
            _city_prefix_support(item, raw_texts),
            hamming_support_score(item, raw_texts, expanded),
            int(not _is_repaired_candidate(item)),
            item.score,
            item.confidence,
        ),
        reverse=True,
    )[:limit]
    return " | ".join(
        f"{candidate.normalized}:{candidate.confidence:.2f}:{candidate.source}:support={hamming_support_score(candidate, raw_texts, expanded):.2f}:score={candidate.score:.2f}"
        for candidate in ranked
    )


def extract_exact_plate_substring(text: str) -> str:
    normalized = strip_plate_prefix(text)
    if not normalized:
        return ""
    matches = [match.group(0) for match in TURKISH_PLATE_SUBSTR_RE.finditer(normalized)]
    if matches:
        return sorted(matches, key=lambda value: (len(value), plate_candidate_score(value)), reverse=True)[0]
    return normalized


def choose_best_ocr_only_candidate(candidates: Iterable[PlateCandidate]) -> PlateCandidate:
    exact_candidates: List[PlateCandidate] = []
    fallback_candidates: List[PlateCandidate] = []
    repaired_candidates: List[PlateCandidate] = []

    for candidate in candidates:
        raw_text = candidate.raw_text or candidate.text
        exact_text = extract_exact_plate_substring(raw_text)
        if not exact_text:
            continue

        cleaned = PlateCandidate(exact_text, candidate.confidence, candidate.source, raw_text)
        if cleaned.regex_ok:
            exact_candidates.append(cleaned)
        else:
            fallback_candidates.append(cleaned)

        repaired_candidates.extend(generate_conservative_format_repairs(raw_text, candidate.confidence, candidate.source))

    if repaired_candidates:
        all_candidates = exact_candidates + repaired_candidates
        return sorted(
            all_candidates,
            key=lambda item: (len(item.normalized), item.confidence, int(item.source.endswith("format_repair"))),
            reverse=True,
        )[0]
    if exact_candidates:
        return sorted(exact_candidates, key=lambda item: (len(item.normalized), item.confidence), reverse=True)[0]
    return PlateCandidate("", 0.0, "none", "")


def format_ocr_only_candidates(candidates: Iterable[PlateCandidate], limit: int = 10) -> str:
    cleaned = []
    for candidate in candidates:
        text = extract_exact_plate_substring(candidate.raw_text or candidate.text)
        if text:
            cleaned.append(PlateCandidate(text, candidate.confidence, candidate.source, candidate.raw_text or candidate.text))
        cleaned.extend(generate_conservative_format_repairs(candidate.raw_text or candidate.text, candidate.confidence, candidate.source))

    cleaned = sorted(cleaned, key=lambda item: (int(item.regex_ok), len(item.normalized), item.confidence), reverse=True)
    if not cleaned:
        return ""
    return " | ".join(
        f"{candidate.normalized}:{candidate.confidence:.2f}:{candidate.source}:regex={'ok' if candidate.regex_ok else 'no'}"
        for candidate in cleaned[:limit]
    )


def hamming_distance(a: str, b: str) -> int:
    a = strip_plate_prefix(a)
    b = strip_plate_prefix(b)
    overlap = min(len(a), len(b))
    distance = sum(1 for idx in range(overlap) if a[idx] != b[idx])
    return distance + abs(len(a) - len(b))


def _raw_candidate_texts(candidates: Iterable[PlateCandidate]) -> List[str]:
    texts = []
    for candidate in candidates:
        for value in (candidate.raw_text, candidate.text, extract_exact_plate_substring(candidate.raw_text or candidate.text)):
            normalized = strip_plate_prefix(value)
            if 4 <= len(normalized) <= 10:
                texts.append(normalized)
    return list(dict.fromkeys(texts))


def _city_prefix_support(candidate: PlateCandidate, raw_texts: List[str]) -> int:
    text = candidate.normalized
    if len(text) < 2 or not text[:2].isdigit():
        return 0
    city = text[:2]
    support = 0
    for raw in raw_texts:
        if raw.startswith(city):
            support += 2
        elif len(raw) >= 2 and raw[0] == city[0] and city[1] in _digit_repair_options(raw[1]):
            support += 1
    return support


def _is_repaired_candidate(candidate: PlateCandidate) -> bool:
    return "regex_repair" in candidate.source or "format_repair" in candidate.source or candidate.source == "beam_search"


def _exact_city_variant_priority(candidate: PlateCandidate, expanded: List[PlateCandidate]) -> int:
    text = candidate.normalized
    if _is_repaired_candidate(candidate) or not candidate.regex_ok:
        return 0
    if not (candidate.source.startswith("ocr_preprocessing/") or candidate.source.startswith("dark_text/")):
        return 0
    if len(text) < 3 or not text[:2].isdigit():
        return 0
    for other in expanded:
        other_text = other.normalized
        if other_text == text or len(other_text) < 3 or not other_text[:2].isdigit():
            continue
        if other_text[0] == text[0] and other_text[1] != text[1] and other_text[2:] == text[2:]:
            return 4
    return 0


def _preferred_plate_length(raw_texts: List[str]) -> int:
    lengths = [len(text) for text in raw_texts if 7 <= len(text) <= 9]
    if not lengths:
        lengths = [len(text) for text in raw_texts if 6 <= len(text) <= 9]
    if not lengths:
        return 8
    counts = {}
    for length in lengths:
        counts[length] = counts.get(length, 0) + 1
    max_count = max(counts.values())
    near_best = [length for length, count in counts.items() if max_count - count <= 1]
    return sorted(near_best, key=lambda length: (length == 8, length), reverse=True)[0]


def hamming_support_score(candidate: PlateCandidate, raw_texts: List[str], generated: List[PlateCandidate]) -> float:
    text = candidate.normalized
    if not text:
        return 0.0

    support = 0.0
    for raw in raw_texts:
        distance = hamming_distance(text, raw)
        support += max(0.0, 4.0 - distance) * 0.6
        if text in raw or raw in text:
            support += 1.2

    for other in generated:
        other_text = other.normalized
        if not other_text or other_text == text:
            continue
        distance = hamming_distance(text, other_text)
        support += max(0.0, 3.0 - distance) * 0.25

    return support


def choose_best_prediction_fallback(candidates: Iterable[PlateCandidate], beam_width: int = 8) -> PlateCandidate:
    source_candidates = list(candidates)
    predicted = expand_plate_candidates(source_candidates, beam_width=beam_width)
    predicted = [candidate for candidate in predicted if candidate.regex_ok and len(candidate.normalized) >= 7]
    if not predicted:
        return PlateCandidate("", 0.0, "none", "")
    raw_texts = _raw_candidate_texts(source_candidates)
    return sorted(
        predicted,
        key=lambda item: (
            len(item.normalized),
            hamming_support_score(item, raw_texts, predicted),
            int(item.source in {"regex_repair", "beam_search"} or item.source.endswith("format_repair")),
            item.score,
            item.confidence,
        ),
        reverse=True,
    )[0]


def format_prediction_fallback(candidates: Iterable[PlateCandidate], beam_width: int = 8, limit: int = 8) -> str:
    source_candidates = list(candidates)
    predicted = expand_plate_candidates(source_candidates, beam_width=beam_width)
    predicted = [candidate for candidate in predicted if candidate.regex_ok and len(candidate.normalized) >= 7]
    if not predicted:
        return ""
    raw_texts = _raw_candidate_texts(source_candidates)
    predicted = sorted(
        predicted,
        key=lambda item: (len(item.normalized), hamming_support_score(item, raw_texts, predicted), item.score, item.confidence),
        reverse=True,
    )
    return " | ".join(
        f"{candidate.normalized}:{candidate.confidence:.2f}:{candidate.source}:hamming={hamming_support_score(candidate, raw_texts, predicted):.2f}:score={candidate.score:.2f}"
        for candidate in predicted[:limit]
    )


def _expected_char_options(ch: str, expected: str) -> List[str]:
    normalized = normalize_plate_text(ch)
    if not normalized:
        return []
    ch = normalized[0]

    if expected == "digit":
        values = [ch] if ch.isdigit() else []
        mapped = DIGIT_MAP.get(ch)
        if mapped:
            values.append(mapped)
        return list(dict.fromkeys(values))

    values = [ch] if ch.isalpha() else []
    values.extend(PLATE_LETTER_OPTIONS.get(ch, []))
    mapped = LETTER_MAP.get(ch)
    if mapped:
        values.append(mapped)
    return list(dict.fromkeys(values))


def generate_conservative_format_repairs(text: str, confidence: float, source: str) -> List[PlateCandidate]:
    normalized = strip_plate_prefix(text)
    repairs: List[PlateCandidate] = []

    for start in range(max(1, len(normalized))):
        for alpha_len in (1, 2, 3):
            for tail_len in (2, 3, 4):
                total_len = 2 + alpha_len + tail_len
                window = normalized[start : start + total_len]
                if len(window) != total_len:
                    continue

                pattern = ["digit", "digit"] + ["letter"] * alpha_len + ["digit"] * tail_len
                options = [_expected_char_options(ch, expected) for ch, expected in zip(window, pattern)]
                if any(not values for values in options):
                    continue

                variants = [""]
                for values in options:
                    variants = [prefix + value for prefix in variants for value in values[:2]]

                for repaired in variants:
                    if repaired == window:
                        continue
                    if TURKISH_PLATE_RE.fullmatch(repaired):
                        repairs.append(PlateCandidate(repaired, confidence, f"{source}:format_repair", text))

    unique = {}
    for candidate in repairs:
        key = candidate.normalized
        if key not in unique or candidate.confidence > unique[key].confidence:
            unique[key] = candidate
    return list(unique.values())
