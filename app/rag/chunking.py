"""Splits documents into overlapping, heading-aware chunks sized for embedding."""

import re
from dataclasses import dataclass

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Chunk:
    text: str
    source: str
    heading: str
    index: int


def chunk_document(text: str, source: str, chunk_size: int = 800, overlap: int = 120) -> list[Chunk]:
    chunks: list[Chunk] = []
    for heading, body in _split_sections(text):
        for piece in split_text(body, chunk_size, overlap):
            # Prefixing the heading gives the embedding the context the paragraph lacks.
            text_with_ctx = f"{heading}\n{piece}" if heading else piece
            chunks.append(Chunk(text=text_with_ctx, source=source, heading=heading, index=len(chunks)))
    return chunks


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Markdown -> [(heading path, body)]. Heading path looks like 'Title > Section'."""
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    lines: list[str] = []

    def flush():
        body = "\n".join(lines).strip()
        if body:
            sections.append((" > ".join(h for _, h in stack), body))
        lines.clear()

    for line in text.splitlines():
        m = _HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            stack[:] = [(lvl, h) for lvl, h in stack if lvl < level]
            stack.append((level, m.group(2).strip()))
        else:
            lines.append(line)
    flush()
    return sections


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Greedy packing of paragraphs (then sentences) up to chunk_size chars, with overlap."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces: list[str] = []
    for p in paragraphs:
        pieces.extend(_split_long(p, chunk_size) if len(p) > chunk_size else [p])

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) + 2 > chunk_size:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            tail = tail.split(" ", 1)[-1] if " " in tail else tail  # start on a word boundary
            current = f"{tail}\n\n{piece}" if tail else piece
        else:
            current = f"{current}\n\n{piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks


def _split_long(paragraph: str, chunk_size: int) -> list[str]:
    out: list[str] = []
    current = ""
    for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
        while len(sentence) > chunk_size:  # pathological: no punctuation at all
            out.append(sentence[:chunk_size])
            sentence = sentence[chunk_size:]
        if current and len(current) + len(sentence) + 1 > chunk_size:
            out.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}" if current else sentence
    if current:
        out.append(current)
    return out
