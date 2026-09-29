"""Small, dependency-free text utilities (tokenising, TF-IDF cosine, Jaccard)."""

from __future__ import annotations

import math
import re
from collections import Counter

_STOP = set(
    """a an and are as at be been by for from has have in is it its of on or that the this to was were
    we with our which these those their can not no into than then also using used use based via such
    both each more most other some over only same so very""".split()
)
_TOKEN = re.compile(r"[a-z][a-z0-9\-]+")


def tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 2]


def jaccard(a: str, b: str) -> float:
    sa, sb = set(tokens(a)), set(tokens(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


class TfIdfIndex:
    """Tiny TF-IDF index; good enough for a local literature/duplicate corpus."""

    def __init__(self, docs: dict[str, str]):
        self.ids = list(docs)
        self._tf = {i: Counter(tokens(t)) for i, t in docs.items()}
        df: Counter[str] = Counter()
        for tf in self._tf.values():
            df.update(tf.keys())
        n = max(len(docs), 1)
        self._idf = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}
        self._vec = {i: self._weigh(tf) for i, tf in self._tf.items()}

    def _weigh(self, tf: Counter[str]) -> dict[str, float]:
        vec = {t: c * self._idf.get(t, 1.0) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def query(self, text: str, k: int = 5) -> list[tuple[str, float]]:
        q = self._weigh(Counter(tokens(text)))
        scores = [(i, sum(w * self._vec[i].get(t, 0.0) for t, w in q.items())) for i in self.ids]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text.strip())
    return [p for p in parts if p]
