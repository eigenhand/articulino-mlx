"""Characters the model reads, and the mapping from text to character ids and language ids.

The model reads lower-case letters, not phonemes. :func:`normalize_characters` lower-cases the text, maps typographic
quotes and dashes to plain ones, strips accents (except on ä, ö, ü, ß), drops every character the model does not know
and collapses white space. Ids are ``index in CHARSET + 1``; id 0 is padding (and the prompt placeholder).

Words in the other language can be marked inline: ``"Das [en]Meeting[/en] ist verschoben."``. Every character then
carries its own language id (0 = German, 1 = English).
"""
from __future__ import annotations

import re
import unicodedata

CHARSET = " abcdefghijklmnopqrstuvwxyzäöüß.,?!-'\":;()0123456789"
PAD_ID = 0
CHAR_TO_ID = {c: i + 1 for i, c in enumerate(CHARSET)}
VOCAB_SIZE = len(CHARSET) + 1
LANGUAGES = {"de": 0, "en": 1}
SPACE_ID = CHAR_TO_ID[" "]

# typographic characters that have a plain counterpart in CHARSET
_REPLACE = {"’": "'", "‘": "'", "`": "'", "´": "'", "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
            "–": "-", "—": "-", "…": ".", "ø": "ö", "æ": "ä", "ð": "d", "þ": "th"}
_TAG = re.compile(r"(\[/?(?:de|en)\])")
_TAG_PARTS = re.compile(r"\[(/?)(de|en)\]")


def normalize_characters(text: str) -> str:
    """Lower-case ``text`` and reduce it to the model's character set."""
    out = []
    for c in text.lower():
        c = _REPLACE.get(c, c)
        if c not in CHAR_TO_ID and c not in "äöüß":
            c = unicodedata.normalize("NFKD", c)
            c = "".join(x for x in c if not unicodedata.combining(x))
        for x in c:
            if x in CHAR_TO_ID:
                out.append(x)
            elif x.isspace():
                out.append(" ")
    return " ".join("".join(out).split())


def text_to_ids(text: str) -> list[int]:
    """Character ids of ``text`` after :func:`normalize_characters` (no language tags)."""
    return [CHAR_TO_ID[c] for c in normalize_characters(text)]


def language_id(language: str) -> int:
    try:
        return LANGUAGES[language]
    except KeyError:
        raise ValueError(f"language must be one of {sorted(LANGUAGES)}, not {language!r}") from None


def encode(text: str, language: str = "de") -> tuple[list[int], list[int]]:
    """Character ids and one language id per character.

    ``[en]…[/en]`` (or ``[de]…[/de]``) switches the language inside the text; ``language`` is the language outside
    the tags. An unclosed tag runs to the end of the text.
    """
    language_id(language)                          # validates
    base = language
    ids: list[int] = []
    langs: list[int] = []
    current = base
    for part in _TAG.split(text):
        tag = _TAG_PARTS.fullmatch(part)
        if tag:
            current = base if tag.group(1) else tag.group(2)
            continue
        chars = normalize_characters(part)
        if not chars:
            if part.isspace() and ids and ids[-1] != SPACE_ID:       # white space between two tags
                ids.append(SPACE_ID)
                langs.append(LANGUAGES[current])
            continue
        if ids and part[:1].isspace() and ids[-1] != SPACE_ID:
            ids.append(SPACE_ID)
            langs.append(LANGUAGES[current])
        ids += [CHAR_TO_ID[c] for c in chars]
        langs += [LANGUAGES[current]] * len(chars)
        if part[-1:].isspace():
            ids.append(SPACE_ID)
            langs.append(LANGUAGES[current])
    while ids and ids[-1] == SPACE_ID:
        ids.pop()
        langs.pop()
    return ids, langs


def open_language_tag(text: str) -> str | None:
    """The tag (``"[en]"`` or ``"[de]"``) still open at the end of ``text``, or None."""
    open_tag = None
    for tag in _TAG_PARTS.finditer(text):
        open_tag = None if tag.group(1) else tag.group(0)
    return open_tag


# ------------------------------------------------------------------------------------------------ foreign spans
_FOREIGN_SPAN = {lang: re.compile(rf"\[{lang}\].*?(?:\[/{lang}\]|$)", re.S) for lang in LANGUAGES}


def _placeholder(i: int) -> str:
    """A plain word in brackets that no stage of the German frontend changes (letters only, has vowels, not all
    caps); the brackets stand where the tags stood, so the text around it is treated as before."""
    letters = ""
    while True:
        letters = chr(97 + i % 26) + letters
        i //= 26
        if not i:
            break
    return "[Fremdwortstelle" + letters + "]"


def protect_foreign_spans(text: str, language: str = "de") -> tuple[str, list[str]]:
    """Replace every span tagged with the *other* language (``[en]…[/en]`` in German text) by a placeholder word,
    so a language-specific text frontend leaves it alone. Returns (masked text, original spans)."""
    other = [lang for lang in LANGUAGES if lang != language]
    spans: list[str] = []

    def keep(m: re.Match) -> str:
        spans.append(m.group(0))
        return _placeholder(len(spans) - 1)

    for lang in other:
        text = _FOREIGN_SPAN[lang].sub(keep, text)
    return text, spans


def restore_foreign_spans(text: str, spans: list[str]) -> str | None:
    """Put the spans back; None if a placeholder did not survive (the caller then falls back)."""
    for i, span in enumerate(spans):
        mark = _placeholder(i)
        if text.count(mark) != 1:
            return None
        text = text.replace(mark, span)
    return text
