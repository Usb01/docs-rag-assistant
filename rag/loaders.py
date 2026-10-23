"""Turn files into (heading, text, page) sections.

Keeping the heading next to the text matters more than people expect: a chunk
that says "Up to 5 unused days can be carried over" is much easier to retrieve
and cite when it also carries "Annual leave".
"""
import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTED = {".md", ".txt", ".pdf", ".docx"}


@dataclass
class Section:
    heading: str
    text: str
    page: int | None = None


def load(path: Path) -> tuple[str, list[Section]]:
    ext = path.suffix.lower()
    if ext in (".md", ".txt"):
        return _markdown(path.read_text(encoding="utf-8"), path.stem)
    if ext == ".pdf":
        return _pdf(path)
    if ext == ".docx":
        return _docx(path)
    raise ValueError(f"unsupported file type: {ext}")


def _markdown(text: str, fallback_title: str) -> tuple[str, list[Section]]:
    title = fallback_title
    sections: list[Section] = []
    heading, buf = "", []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            sections.append(Section(heading or title, body))

    for line in text.splitlines():
        m = re.match(r"^(#{1,4})\s+(.*)", line)
        if m:
            if len(m.group(1)) == 1 and title == fallback_title:
                title = m.group(2).strip()
                continue
            flush()
            heading, buf = re.sub(r"^\d+\.\s*", "", m.group(2).strip()), []
        else:
            buf.append(line)
    flush()
    return title, sections


def _pdf(path: Path) -> tuple[str, list[Section]]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    title = (reader.metadata.title if reader.metadata else None) or path.stem.replace("_", " ").title()
    sections = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        # Headings in our PDFs look like "3. On-time delivery commitment" on their own line
        parts = re.split(r"\n(?=\d+\.\s+[A-Z][^\n]{2,60}\n)", "\n" + text)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            first, _, rest = part.partition("\n")
            if re.match(r"\d+\.\s+", first) and rest:
                sections.append(Section(re.sub(r"^\d+\.\s*", "", first).strip(), _unwrap(rest), page_no))
            else:
                sections.append(Section(title, _unwrap(part), page_no))
    return title, sections


def _docx(path: Path) -> tuple[str, list[Section]]:
    from docx import Document

    doc = Document(str(path))
    title = doc.core_properties.title or path.stem.replace("_", " ").title()
    sections, heading, buf = [], "", []
    for para in doc.paragraphs:
        style = (para.style.name or "").lower()
        text = para.text.strip()
        if not text:
            continue
        if style == "title":
            title = text
        elif style.startswith("heading"):
            if buf:
                sections.append(Section(heading or title, "\n".join(buf)))
            heading, buf = text, []
        else:
            buf.append(text)
    if buf:
        sections.append(Section(heading or title, "\n".join(buf)))
    return title, sections


def _unwrap(text: str) -> str:
    # PDF extraction breaks lines mid-sentence; join them back
    return re.sub(r"(?<![.\n])\n(?!\n)", " ", text).strip()
