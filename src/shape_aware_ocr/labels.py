from __future__ import annotations

import re

_NON_ALNUM = re.compile(r"[^A-Z0-9]+")


def normalize_label(text: str) -> str:
    raw = str(text or "").upper().strip()
    return _NON_ALNUM.sub("", raw)


def normalized_match_stem(stem: str) -> str:
    parts = [part for part in str(stem or "").split("_") if part]
    if not parts:
        return ""
    if len(parts) >= 3:
        tail = parts[1:]
        half = len(tail) // 2
        if len(tail) % 2 == 0 and tail[:half] == tail[half:]:
            return "_".join([parts[0], *tail[:half]])
    return "_".join(parts)


def label_from_filename_stem(stem: str) -> str:
    normalized = normalized_match_stem(stem)
    parts = [part for part in normalized.split("_") if part]
    if len(parts) >= 2:
        return normalize_label("".join(parts[1:]))
    return normalize_label(normalized)
