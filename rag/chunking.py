import re
from dataclasses import dataclass

from .loaders import Section


@dataclass
class Chunk:
    doc_title: str
    heading: str
    text: str
    page: int | None
    position: int

    @property
    def indexed_text(self) -> str:
        # heading is prepended for retrieval only; we show `text` to users
        return f"{self.doc_title} – {self.heading}\n{self.text}"


SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(•-])")


def split_sentences(text: str) -> list[str]:
    out = []
    for block in re.split(r"\n\s*\n|\n(?=[-*•]\s)", text):
        block = block.strip()
        if block:
            out.extend(s.strip() for s in SENTENCE_END.split(block) if s.strip())
    return out


def chunk_sections(doc_title: str, sections: list[Section], max_chars: int = 700, overlap: int = 1) -> list[Chunk]:
    """Pack sentences into chunks of roughly `max_chars`, never crossing a heading.

    `overlap` sentences are repeated at the start of the next chunk so an
    answer that straddles a boundary still lands in one chunk.
    """
    chunks: list[Chunk] = []
    for section in sections:
        sentences = split_sentences(section.text)
        current: list[str] = []
        for sentence in sentences:
            if current and sum(len(s) + 1 for s in current) + len(sentence) > max_chars:
                chunks.append(Chunk(doc_title, section.heading, " ".join(current), section.page, len(chunks)))
                current = current[-overlap:] if overlap else []
            current.append(sentence)
        if current:
            chunks.append(Chunk(doc_title, section.heading, " ".join(current), section.page, len(chunks)))
    return chunks
