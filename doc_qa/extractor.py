"""
Extract text from PDF, DOCX, TXT, and MD files with page/section metadata.

Returns a list of (text, page_num, section) tuples where:
  - text       : raw text of this segment
  - page_num   : int for PDFs, None for DOCX/TXT
  - section    : heading label (e.g. "Page 3", "Introduction", "Section 2.1")
"""

from pathlib import Path


def extract(filepath: str) -> list:
    """
    Dispatch to the correct extractor based on file extension.
    Raises ValueError for unsupported types.
    If extraction fails, raises the underlying exception so the caller
    can surface a clear error message to the user (AC: clear error on failure).
    """
    ext = Path(filepath).suffix.lower()
    if ext == ".pdf":
        return _extract_pdf(filepath)
    elif ext == ".docx":
        return _extract_docx(filepath)
    elif ext in (".txt", ".md"):
        return _extract_text(filepath)
    else:
        raise ValueError(
            f"Unsupported file type: '{ext}'. Supported types: PDF, DOCX, TXT, MD."
        )


def _extract_pdf(filepath: str) -> list:
    from pypdf import PdfReader

    reader = PdfReader(filepath)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((text, i, f"Page {i}"))

    if not pages:
        raise RuntimeError("PDF appears to contain no extractable text (may be scanned/image-only).")
    return pages


def _extract_docx(filepath: str) -> list:
    from docx import Document

    doc = Document(filepath)
    segments = []
    current_heading = "Introduction"
    buffer = []

    for para in doc.paragraphs:
        if para.style.name.startswith("Heading"):
            if buffer:
                segments.append(("\n".join(buffer), None, current_heading))
                buffer = []
            current_heading = para.text.strip() or current_heading
        elif para.text.strip():
            buffer.append(para.text.strip())

    if buffer:
        segments.append(("\n".join(buffer), None, current_heading))

    if not segments:
        raise RuntimeError("DOCX appears to contain no extractable text.")
    return segments


def _extract_text(filepath: str) -> list:
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    segments = []
    current_section = "Document"
    buffer = []

    for line in lines:
        stripped = line.rstrip()
        if stripped.startswith("#"):
            if buffer:
                segments.append(("\n".join(buffer), None, current_section))
                buffer = []
            current_section = stripped.lstrip("#").strip() or current_section
        elif stripped:
            buffer.append(stripped)

    if buffer:
        segments.append(("\n".join(buffer), None, current_section))

    if not segments:
        # Plain text with no headings — treat the whole file as one segment
        full = "".join(lines).strip()
        if full:
            return [(full, None, "Document")]
        raise RuntimeError("File appears to be empty.")

    return segments
