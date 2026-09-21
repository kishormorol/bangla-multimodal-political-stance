"""Bangla normalization: fix encoding noise, leave the wording alone."""

from __future__ import annotations

from bmpb.data.preprocess import (
    is_usable,
    normalize_digits,
    normalize_text,
    strip_markup,
)

BANGLA = "বাংলাদেশের রাজনীতিতে নতুন মোড় এসেছে বলে মনে করছেন বিশ্লেষকেরা"


def test_markup_and_urls_are_stripped():
    text = f"<p>{BANGLA}</p> https://prothomalo.com/x আরও news@example.com"
    cleaned = strip_markup(text)
    assert "http" not in cleaned
    assert "@" not in cleaned
    assert "<p>" not in cleaned
    assert "রাজনীতিতে" in cleaned


def test_zero_width_joiners_are_removed():
    assert "‌" not in normalize_text(f"বাং‌লাদেশ {BANGLA}", use_buet_normalizer=False)


def test_digits_are_unified():
    assert normalize_digits("২০২৪ সালে", to="latin").startswith("2024")
    assert normalize_digits("2024 সালে", to="bangla").startswith("২০২৪")


def test_repeated_punctuation_collapses():
    assert normalize_text(f"{BANGLA}।।।", use_buet_normalizer=False).endswith("।")


def test_whitespace_is_collapsed():
    assert "  " not in normalize_text(f"{BANGLA}   \n\t {BANGLA}", use_buet_normalizer=False)


def test_normalization_is_idempotent():
    once = normalize_text(BANGLA, use_buet_normalizer=False)
    assert normalize_text(once, use_buet_normalizer=False) == once


def test_non_string_input_is_safe():
    assert normalize_text(None) == ""
    assert normalize_text(float("nan")) == ""


def test_usability_filter_rejects_stubs_and_non_bangla():
    assert is_usable(BANGLA)
    assert not is_usable("short")
    assert not is_usable("This is an English sentence that is long enough to pass length.")
    assert not is_usable("")
