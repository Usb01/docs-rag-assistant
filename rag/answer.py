import os
import re
from dataclasses import dataclass, field

import httpx

from .bm25 import tokenize
from .chunking import split_sentences
from .retriever import Hit, Retriever

NOT_FOUND = "I couldn't find this in the documents. Try rephrasing, or ask the team that owns this policy."

SYSTEM = """You answer questions for employees and customers using ONLY the numbered sources provided.
Rules:
- Cite every fact with its source number in square brackets, e.g. [2]. Use only numbers that exist.
- If the sources don't contain the answer, reply exactly: NOT_FOUND
- Keep it short: 1-4 sentences or a short list. Quote numbers, amounts and deadlines exactly.
- Don't mention "the sources" or "the context" - just answer."""


@dataclass
class Answer:
    text: str
    sources: list[dict] = field(default_factory=list)
    mode: str = "extractive"
    found: bool = True


def _sources(hits: list[Hit]) -> list[dict]:
    return [{"n": i + 1, "citation": h.chunk.citation(), "doc": h.chunk.doc_title, "heading": h.chunk.heading,
             "page": h.chunk.page, "text": h.chunk.text, "score": round(h.score, 4)} for i, h in enumerate(hits)]


class Answerer:
    def __init__(self, retriever: Retriever, provider: str | None = None, min_keyword: float = 1.5,
                 min_semantic: float = 0.35, min_coverage: float = 0.15):
        self.retriever = retriever
        self.provider = (provider or os.getenv("LLM_PROVIDER", "none")).lower()
        self.min_keyword = min_keyword
        self.min_semantic = min_semantic
        self.min_coverage = min_coverage

    def ask(self, question: str, k: int = 4) -> Answer:
        hits = self.retriever.search(question, k=k)
        if not self._relevant(hits):
            return Answer(NOT_FOUND, [], "none", found=False)

        if self.provider in ("openai", "anthropic"):
            try:
                answer = self._generate(question, hits)
                if answer:
                    return answer
            except Exception:
                pass  # fall through to the extractive answer
        return self._extractive(question, hits)

    # -----------------------------------------------------------------------

    def _relevant(self, hits: list[Hit]) -> bool:
        if not hits:
            return False
        top = hits[0]
        if top.semantic is not None and top.semantic >= self.min_semantic:
            return True
        return top.keyword >= self.min_keyword and top.coverage >= self.min_coverage

    def _extractive(self, question: str, hits: list[Hit]) -> Answer:
        """No-LLM mode: return the sentences that best match the question, with citations."""
        q = set(tokenize(question))
        scored = []
        for n, hit in enumerate(hits[:3], start=1):
            for pos, sentence in enumerate(split_sentences(hit.chunk.text)):
                overlap = len(q & set(tokenize(sentence)))
                if overlap:
                    # small bonus for the best-ranked chunk so ties go to it
                    scored.append((overlap + (0.5 if n == 1 else 0), n, pos, sentence))
        if not scored:
            return Answer(NOT_FOUND, [], "extractive", found=False)

        best = sorted(scored, key=lambda s: -s[0])[:3]
        threshold = best[0][0] * 0.75
        picked = sorted([s for s in best if s[0] >= threshold], key=lambda s: (s[1], s[2]))
        text = " ".join(f"{_clean(s[3])} [{s[1]}]" for s in picked)
        used = {s[1] for s in picked}
        return Answer(text, [s for s in _sources(hits) if s["n"] in used], "extractive")

    def _generate(self, question: str, hits: list[Hit]) -> Answer | None:
        sources = _sources(hits)
        context = "\n\n".join(f"[{s['n']}] {s['citation']}\n{s['text']}" for s in sources)
        user = f"Sources:\n{context}\n\nQuestion: {question}"
        text = self._call(SYSTEM, user).strip()

        if text.startswith("NOT_FOUND"):
            return Answer(NOT_FOUND, [], self.provider, found=False)

        cited = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
        valid = {s["n"] for s in sources}
        if not cited or not cited <= valid:
            return None  # ungrounded or hallucinated citation -> use extractive answer
        return Answer(text, [s for s in sources if s["n"] in cited], self.provider)

    def _call(self, system: str, user: str) -> str:
        if self.provider == "openai":
            r = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                json={"model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"), "temperature": 0,
                      "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
                timeout=60,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        r = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"},
            json={"model": os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest"), "max_tokens": 500,
                  "system": system, "messages": [{"role": "user", "content": user}]},
            timeout=60,
        )
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json()["content"])


def _clean(sentence: str) -> str:
    return re.sub(r"\*\*|^[-*•]\s*", "", sentence).strip()
