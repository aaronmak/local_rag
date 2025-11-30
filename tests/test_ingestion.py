"""Tests for document ingestion utilities."""

import tempfile
from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter

from scripts.ingest_documents import load_documents, load_pdf_file, load_text_file


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_load_text_file(temp_dir):
    """Test loading a text file."""
    # Create a test text file
    test_file = temp_dir / "test.txt"
    test_content = "This is a test document.\nWith multiple lines."
    test_file.write_text(test_content, encoding="utf-8")

    # Load the file
    content, metadata = load_text_file(test_file)

    assert content == test_content
    assert metadata["filename"] == "test.txt"
    assert metadata["file_type"] == "txt"
    assert "source" in metadata


def test_load_pdf_file(temp_dir):
    """Test loading a PDF file."""
    # Create a simple test PDF
    test_file = temp_dir / "test.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)

    with open(test_file, "wb") as f:
        writer.write(f)

    # Load the file
    content, metadata = load_pdf_file(test_file)

    assert isinstance(content, str)
    assert metadata["filename"] == "test.pdf"
    assert metadata["file_type"] == "pdf"
    # Verify num_pages is an integer (may be 0 for blank pages with no extractable text)
    assert isinstance(metadata["num_pages"], int)
    assert metadata["num_pages"] >= 0
    assert "source" in metadata


def test_load_documents(temp_dir):
    """Test loading multiple documents from a directory."""
    # Create test files
    (temp_dir / "doc1.txt").write_text("Document one", encoding="utf-8")
    (temp_dir / "doc2.txt").write_text("Document two", encoding="utf-8")

    # Create a PDF
    pdf_file = temp_dir / "doc3.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with open(pdf_file, "wb") as f:
        writer.write(f)

    # Create a file with unsupported extension
    (temp_dir / "ignored.md").write_text("Should be ignored", encoding="utf-8")

    # Load documents
    documents = load_documents(temp_dir)

    # Should load 3 files (2 txt + 1 pdf), ignore .md
    assert len(documents) == 3

    # Verify structure
    for content, metadata in documents:
        assert isinstance(content, str)
        assert "filename" in metadata
        assert "source" in metadata
        assert "file_type" in metadata


def test_load_documents_recursive(temp_dir):
    """Test loading documents from nested directories."""
    # Create nested structure
    subdir = temp_dir / "subdir"
    subdir.mkdir()

    (temp_dir / "root.txt").write_text("Root document", encoding="utf-8")
    (subdir / "nested.txt").write_text("Nested document", encoding="utf-8")

    # Load documents
    documents = load_documents(temp_dir)

    # Should find both files
    assert len(documents) == 2


def test_load_documents_empty_directory(temp_dir):
    """Test loading from an empty directory."""
    documents = load_documents(temp_dir)
    assert len(documents) == 0


def test_load_documents_with_errors(temp_dir, capsys):
    """Test handling of file loading errors."""
    # Create a file with invalid encoding
    bad_file = temp_dir / "bad.txt"
    bad_file.write_bytes(b"\x80\x81\x82")  # Invalid UTF-8

    # Load documents (should handle error gracefully)
    documents = load_documents(temp_dir)

    # Should return empty list and print error
    assert len(documents) == 0
    captured = capsys.readouterr()
    assert "Error loading" in captured.out
