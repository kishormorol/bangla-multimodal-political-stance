"""Bangla text normalization.

The raw articles arrive with mixed punctuation, Latin-digit/Bangla-digit mixing,
zero-width joiners inside conjuncts, and the usual scraped-HTML debris. These
functions are deliberately conservative: they fix encoding-level inconsistency
and leave the wording alone, because the labels are about stance, not surface
form.
"""

from __future__ import annotations

import re
import unicodedata

# csebuetnlp's normalizer is what BanglaBERT/BanglaT5 were trained with; using it
# keeps our tokenization consistent with the checkpoints. It is optional so that
# ingest still runs on a machine without it.
try:
    from normalizer import normalize as _buet_normalize
except ImportError:  # pragma: no cover - optional dependency
    _buet_normalize = None

BANGLA_DIGITS = "০১২৩৪৫৬৭৮৯"
_DIGIT_MAP = {ord(b): str(i) for i, b in enumerate(BANGLA_DIGITS)}

_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_HTML = re.compile(r"<[^>]+>")
_ZERO_WIDTH = re.compile(r"[​‌‍﻿]")
_WHITESPACE = re.compile(r"\s+")
_REPEATED_PUNCT = re.compile(r"([।!?,.])\1{1,}")
_EMOJI = re.compile(
    "[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f1e6-\U0001f1ff]",
    flags=re.UNICODE,
)


def strip_markup(text: str) -> str:
    """Remove HTML tags, URLs and e-mail addresses left by the scraper."""
    text = _HTML.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _EMAIL.sub(" ", text)
    return text


def normalize_digits(text: str, to: str = "bangla") -> str:
    """Unify digit systems. Default keeps Bangla digits, which the encoders saw."""
    if to == "latin":
        return text.translate(_DIGIT_MAP)
    for latin, bangla in enumerate(BANGLA_DIGITS):
        text = text.replace(str(latin), bangla)
    return text


def normalize_text(
    text: str,
    *,
    remove_emoji: bool = False,
    unify_digits: str | None = "bangla",
    use_buet_normalizer: bool = True,
) -> str:
    """Canonical form of one article's text. Safe to call on already-clean text."""
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFC", text)
    text = strip_markup(text)
    text = _ZERO_WIDTH.sub("", text)

    if remove_emoji:
        text = _EMOJI.sub(" ", text)
    if unify_digits:
        text = normalize_digits(text, to=unify_digits)

    text = _REPEATED_PUNCT.sub(r"\1", text)
    text = _WHITESPACE.sub(" ", text).strip()

    if use_buet_normalizer and _buet_normalizer_available():
        text = _buet_normalize(text)
    return text


def _buet_normalizer_available() -> bool:
    return _buet_normalize is not None


def is_usable(text: str, min_chars: int = 20, min_bangla_ratio: float = 0.5) -> bool:
    """Reject rows that are empty, stub-length, or not actually Bangla."""
    if not text or len(text) < min_chars:
        return False
    bangla = sum(1 for ch in text if "ঀ" <= ch <= "৿")
    letters = sum(1 for ch in text if ch.isalpha())
    return letters > 0 and bangla / letters >= min_bangla_ratio
