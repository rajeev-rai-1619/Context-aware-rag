"""
Tests for DatasetLoader -- paragraph chunking on blank lines.
"""

from app.dataset import DatasetLoader


def test_load_text_file_splits_on_blank_lines(tmp_path):
    file_path = tmp_path / "docs.txt"
    file_path.write_text(
        "First paragraph.\n\n"
        "Second paragraph spans\n"
        "two lines.\n\n"
        "Third paragraph.\n",
        encoding="utf-8",
    )

    chunks = DatasetLoader.load_text_file(str(file_path))

    assert len(chunks) == 3
    assert chunks[0] == "First paragraph."
    assert chunks[1] == "Second paragraph spans\ntwo lines."
    assert chunks[2] == "Third paragraph."


def test_load_text_file_ignores_empty_chunks(tmp_path):
    file_path = tmp_path / "docs.txt"
    file_path.write_text("\n\n\nOnly one.\n\n\n\n", encoding="utf-8")

    chunks = DatasetLoader.load_text_file(str(file_path))

    assert chunks == ["Only one."]
