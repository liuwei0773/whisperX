# furigana.py
"""Generate WebVTT ruby (furigana) annotations for Japanese subtitle text.

Ruby markup (``<ruby>漢字<rt>かんじ</rt></ruby>``) is HTML/WebVTT syntax and is
only meaningful in ``.vtt`` output rendered by a WebVTT-aware player; SRT players
display the raw tags, so callers must restrict this to VTT + Japanese.

Readings are produced with `pykakasi <https://github.com/miurahr/pykakasi>`_,
which is loaded lazily so the dependency is only required when furigana is
actually requested.
"""

from functools import lru_cache

# Kanji (CJK unified ideographs, incl. the common extension blocks) plus the
# iteration mark 々. Only fragments containing these need a reading.
_KANJI_RE = None


def _has_kanji(text: str) -> bool:
    global _KANJI_RE
    if _KANJI_RE is None:
        import re

        _KANJI_RE = re.compile(r"[㐀-鿿豈-﫿\U00020000-\U0002ffff々]")
    return bool(_KANJI_RE.search(text))


@lru_cache(maxsize=1)
def _get_converter():
    try:
        import pykakasi
    except ImportError as e:
        raise ImportError(
            "Furigana output requires the 'pykakasi' package. "
            "Install it with: pip install 'whisperx[ja]'  (or: pip install pykakasi)"
        ) from e
    return pykakasi.Kakasi()


def add_furigana(text: str) -> str:
    """Wrap kanji-containing fragments of ``text`` in WebVTT ruby markup.

    Kana-only fragments, punctuation and whitespace are passed through
    unchanged. Returns the original text unmodified when it contains no kanji.
    """
    if not text or not _has_kanji(text):
        return text

    kks = _get_converter()
    out = []
    for token in kks.convert(text):
        orig = token["orig"]
        reading = token["hira"]
        # Only annotate fragments that contain kanji and whose reading actually
        # differs from the surface form (skips kana, digits, punctuation).
        if _has_kanji(orig) and reading and reading != orig:
            out.append(f"<ruby>{orig}<rt>{reading}</rt></ruby>")
        else:
            out.append(orig)
    return "".join(out)
