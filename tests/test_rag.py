from pathlib import Path

import pytest

from rag.answer import NOT_FOUND, Answerer
from rag.bm25 import BM25, tokenize
from rag.chunking import chunk_sections, split_sentences
from rag.loaders import Section, load
from rag.retriever import Retriever
from rag.store import Store

DOCS = Path(__file__).resolve().parent.parent / "sample_docs"


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    s = Store(tmp_path_factory.mktemp("db") / "t.db")
    s.ingest_folder(DOCS)
    return s


def test_every_format_loads_with_headings():
    for name in ["employee_handbook.md", "service_level_agreement.pdf", "it_security_policy.docx"]:
        title, sections = load(DOCS / name)
        assert title and len(sections) >= 4
        assert all(s.text for s in sections)


def test_pdf_sections_keep_page_numbers():
    _, sections = load(DOCS / "service_level_agreement.pdf")
    credits = next(s for s in sections if s.heading == "Service credits")
    assert credits.page == 1 and "25%" in credits.text


def test_chunks_never_cross_headings_and_respect_size():
    long = " ".join(f"Sentence number {i} is here." for i in range(80))
    chunks = chunk_sections("Doc", [Section("A", long), Section("B", "Short one.")], max_chars=300)
    assert {c.heading for c in chunks} == {"A", "B"}
    assert all(len(c.text) <= 330 for c in chunks)
    a = [c for c in chunks if c.heading == "A"]
    # one sentence of overlap between consecutive chunks
    assert a[0].text.split(". ")[-1].rstrip(".") in a[1].text


def test_sentence_split_keeps_decimals_and_amounts():
    assert len(split_sentences("Rate is SAR 0.75 per km. Next sentence.")) == 2


def test_tokenizer_matches_word_forms():
    assert tokenize("deliveries")[0] == tokenize("deliver")[0]
    assert tokenize("carried")[0] == tokenize("carry")[0]


def test_bm25_scores_short_focused_docs_higher():
    idx = BM25(["the cat sat", "the dog sat", "a cat with a big brown dog nearby"])
    s = idx.scores("cat")
    assert s[1] == 0 and s[0] > s[2] > 0


def test_reingest_is_skipped_when_unchanged(store):
    assert store.ingest(DOCS / "customer_faq.md") == "unchanged"


def test_answer_is_cited(store):
    a = Answerer(Retriever(store), provider="none").ask("What is the maximum service credit?")
    assert a.found and "25%" in a.text and "[1]" in a.text
    assert a.sources[0]["heading"] == "Service credits"


def test_out_of_scope_question(store):
    a = Answerer(Retriever(store), provider="none").ask("Who won the 2022 world cup?")
    assert not a.found and a.text == NOT_FOUND


def test_llm_answer_with_bad_citation_falls_back(store, monkeypatch):
    ans = Answerer(Retriever(store), provider="openai")
    monkeypatch.setattr(ans, "_call", lambda s, u: "Passwords need 8 characters [7].")
    a = ans.ask("What is the minimum password length?")
    assert a.mode == "extractive" and "12" in a.text


def test_llm_answer_with_valid_citation_is_used(store, monkeypatch):
    ans = Answerer(Retriever(store), provider="anthropic")
    monkeypatch.setattr(ans, "_call", lambda s, u: "At least 12 characters, with MFA on email [1].")
    a = ans.ask("What is the minimum password length?")
    assert a.mode == "anthropic" and len(a.sources) == 1


def test_llm_not_found(store, monkeypatch):
    ans = Answerer(Retriever(store), provider="openai")
    monkeypatch.setattr(ans, "_call", lambda s, u: "NOT_FOUND")
    assert not ans.ask("What is the password length for the VPN?").found


def test_question_with_only_generic_overlap_is_not_answered(store):
    a = Answerer(Retriever(store), provider="none").ask("What is the dress code for drivers?")
    assert not a.found
