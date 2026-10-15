"""Splitting text into the pieces the model speaks one at a time.

The model generates one sentence at a time. A sentence ends at ``. ! ? … : ;`` (plus closing quotes or brackets)
followed by white space. Sentences with fewer than 20 letters are joined to the next one (very short pieces sound
unnatural), and sentences longer than 180 characters are split at the last comma (or space) before 180 characters,
because the model was trained on clips of up to about 20 seconds including the 5 second voice prompt.

:class:`SentenceBuffer` does the same for text that arrives in pieces (for example tokens from a language model): it
returns a sentence as soon as it is complete and safe to speak.
"""
from __future__ import annotations

import re

from .text import open_language_tag

MAX_CHARS = 180
MIN_LETTERS = 20
_SENTENCE = re.compile(r"(.+?[.!?…:;]+[\"')\]]*)(?:\s+|$)", re.S)


def _letters(s: str) -> int:
    return sum(c.isalpha() for c in s)


def split_sentences(buffer: str, final: bool, max_chars: int = MAX_CHARS,
                    min_letters: int = MIN_LETTERS) -> tuple[list[str], str]:
    """Complete sentences in ``buffer`` and the rest that has to wait for more text.

    With ``final=True`` everything is returned (the text is complete). Without, a sentence mark at the very end of
    the buffer is not trusted yet (``"3."`` may continue as ``"3.5"``, ``"."`` as ``"..."``), and a last sentence
    with fewer than ``min_letters`` letters waits for the next one.
    """
    out, pos = [], 0
    for m in _SENTENCE.finditer(buffer):
        if m.end() == len(buffer) and not final and not buffer[-1:].isspace():
            break
        out.append(m.group(1).strip())
        pos = m.end()
    rest = buffer[pos:]
    if final and rest.strip():
        out.append(rest.strip())
        rest = ""
    joined: list[str] = []
    for s in out:                                   # short sentences go together with the next one
        if joined and _letters(joined[-1]) < min_letters:
            joined[-1] += " " + s
        else:
            joined.append(s)
    if joined and not final and _letters(joined[-1]) < min_letters:
        rest = joined.pop() + " " + rest
    pieces = []
    for s in joined:                                # very long sentences are split at a comma
        while len(s) > max_chars:
            k = s.rfind(", ", 60, max_chars)
            k = k + 1 if k > 0 else s.rfind(" ", 60, max_chars)
            if k <= 0:
                break
            pieces.append(s[:k].strip())
            s = s[k:].strip()
        pieces.append(s)
    return pieces, rest


def carry_language_tags(pieces: list[str]) -> list[str]:
    """Re-open a ``[en]``/``[de]`` span that a sentence split cut in two, so every piece is tagged on its own."""
    out, open_tag = [], None
    for piece in pieces:
        if open_tag:
            piece = open_tag + piece
        open_tag = open_language_tag(piece)
        out.append(piece)
    return out


class SentenceBuffer:
    """Collects text pieces and hands out sentences once they are complete.

    >>> buf = SentenceBuffer()
    >>> buf.push("Guten Mor"), buf.push("gen! Heute regnet es ab achtzehn Uhr. ")
    ([], ['Guten Morgen! Heute regnet es ab achtzehn Uhr.'])
    >>> buf.flush()
    []
    """

    def __init__(self, max_chars: int = MAX_CHARS, min_letters: int = MIN_LETTERS):
        self.max_chars, self.min_letters = max_chars, min_letters
        self.text = ""
        self._open_tag: str | None = None

    def push(self, piece: str) -> list[str]:
        self.text += piece
        return self._take(final=False)

    def flush(self, piece: str = "") -> list[str]:
        """Add the last piece and return everything that is left."""
        self.text += piece
        return self._take(final=True)

    def _take(self, final: bool) -> list[str]:
        pieces, self.text = split_sentences(self.text, final, self.max_chars, self.min_letters)
        out = []
        for piece in pieces:
            if self._open_tag:
                piece = self._open_tag + piece
            self._open_tag = open_language_tag(piece)
            out.append(piece)
        return out
