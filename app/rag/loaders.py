"""Loads .md / .txt / .pdf files into plain text."""

import io
from pathlib import Path

SUPPORTED = {".md", ".txt", ".pdf"}


def load_bytes(name: str, data: bytes) -> str:
    suffix = Path(name).suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix in {".md", ".txt"}:
        return data.decode("utf-8", errors="replace")
    raise ValueError(f"Unsupported file type: {suffix} (supported: {sorted(SUPPORTED)})")


def iter_files(root: str | Path):
    root = Path(root)
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            yield path.relative_to(root).as_posix(), path.read_bytes()
