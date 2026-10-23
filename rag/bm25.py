"""Small BM25 implementation – keyword search that works with no API key and
catches exact terms (SAR amounts, plan names, codes) that embeddings sometimes miss."""
import math
import re
from collections import Counter

STOPWORDS = set("""a an the and or of to in on for is are was were be been it its this that with as at by from
can do does did i you we our your my me us they them their what which who when where how if not no than then
there these those so any all about into over after before up out will would should could may might has have had""".split())


def stem(word: str) -> str:
    """Deliberately crude: strip common suffixes, then cut to 6 characters so
    deliver / delivery / deliveries and carry / carried all meet."""
    for suffix in ("ing", "ies", "ied", "es", "ed", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)] + ("y" if suffix in ("ies", "ied") else "")
            break
    return word[:6]


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.4, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(d)) for d in docs]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.docs else 0
        df = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        out = []
        for doc, length in zip(self.docs, self.lengths):
            s = 0.0
            for t in terms:
                tf = doc.get(t)
                if not tf:
                    continue
                s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / self.avg_len))
            out.append(s)
        return out

    def coverage(self, query: str, doc_index: int) -> float:
        """Share of the query's information (idf-weighted) found in one doc.

        Terms that appear nowhere in the corpus get the highest idf, so a
        question about "parental leave in Germany" scores low against a chunk
        that only matches "leave".
        """
        terms = set(tokenize(query))
        if not terms:
            return 0.0
        unseen = max(self.idf.values(), default=1.0) * 1.2
        doc = self.docs[doc_index]
        total = sum(self.idf.get(t, unseen) for t in terms)
        found = sum(self.idf[t] for t in terms if t in doc)
        return found / total
