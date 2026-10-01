from app.rag.chunking import chunk_document, split_text


def test_sections_carry_heading_path():
    text = "# Title\n\nintro\n\n## Sub\n\nbody text\n\n### Deep\n\ndeeper"
    chunks = chunk_document(text, "doc.md", chunk_size=500, overlap=0)
    assert [c.heading for c in chunks] == ["Title", "Title > Sub", "Title > Sub > Deep"]
    assert chunks[1].text.startswith("Title > Sub\n")
    assert [c.index for c in chunks] == [0, 1, 2]


def test_split_respects_size_and_overlaps():
    paragraphs = [f"paragraph {i} " + "word " * 30 for i in range(10)]
    pieces = split_text("\n\n".join(paragraphs), chunk_size=400, overlap=50)
    assert len(pieces) > 1
    assert all(len(p) <= 400 + 60 for p in pieces)
    # the tail of chunk N re-appears at the start of chunk N+1
    assert pieces[0][-20:].strip() in pieces[1]


def test_long_paragraph_without_breaks_is_split():
    pieces = split_text("x" * 2000, chunk_size=500, overlap=0)
    assert len(pieces) == 4


def test_empty_document_gives_no_chunks():
    assert chunk_document("", "empty.md") == []
