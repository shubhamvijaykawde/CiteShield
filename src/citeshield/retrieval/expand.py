"""Structure-aware expansion of the reranked hits.

Legal text is dense with internal references that embeddings cannot follow:
  * definitions: "Is an emotion recognition system high-risk?" needs AI Act Art. 3(39);
  * cross-references: "the conditions referred to in paragraph 1" needs paragraph 1.
Expansions are appended after the ranked hits and labelled with their ``source`` so the
agent and the evaluation can tell them apart from search results.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from citeshield.contracts.chunk import ChunkKind
from citeshield.retrieval.config import RetrieverConfig
from citeshield.retrieval.corpus import Corpus
from citeshield.retrieval.hybrid import Hit, HitSource

# ‘term’ means ... (OJ uses typographic quotes; tolerate ASCII ones)
_DEFINED_TERM = re.compile(r"[‘'\"]([^’'\"]{3,80})[’'\"]\s+means\b")
_PARAGRAPH_REF = re.compile(r"\b(?:referred to in|pursuant to|under) paragraphs? (\d+)\b")
_GENERIC_TERMS = frozenset({"person", "processing", "service", "data", "user", "product"})


@dataclass(frozen=True)
class _Definition:
    term: str
    pattern: re.Pattern[str]
    chunk_id: str
    regulation: str


class Expander:
    def __init__(self, corpus: Corpus) -> None:
        self.corpus = corpus
        self.definitions: list[_Definition] = []
        for chunk in corpus:
            if chunk.kind is not ChunkKind.ARTICLE or not chunk.heading_path[-1].endswith(
                "Definitions"
            ):
                continue
            for m in _DEFINED_TERM.finditer(chunk.text):
                term = m.group(1).strip()
                if term.lower() in _GENERIC_TERMS:
                    continue
                pattern = re.compile(rf"\b{re.escape(term)}s?\b", re.IGNORECASE)
                self.definitions.append(
                    _Definition(term, pattern, chunk.chunk_id, chunk.regulation)
                )
        # Prefer the most specific term: "remote biometric identification system" over "AI system".
        self.definitions.sort(key=lambda d: len(d.term), reverse=True)

    def expand(self, query: str, hits: list[Hit], cfg: RetrieverConfig) -> list[Hit]:
        if cfg.max_expansions == 0 or not hits:
            return []
        seen = {h.chunk.chunk_id for h in hits}
        top = hits[:3]
        regulations = {h.chunk.regulation for h in top}
        out: list[Hit] = []

        def add(chunk_id: str, source: HitSource) -> None:
            if chunk_id not in seen and len(out) < cfg.max_expansions:
                seen.add(chunk_id)
                chunk = self.corpus[chunk_id]
                out.append(Hit(chunk=chunk, retrieval_rank=0, retrieval_score=0.0, source=source))

        if cfg.expand_definitions:
            covered: list[str] = []
            for d in self.definitions:
                if d.regulation not in regulations or not d.pattern.search(query):
                    continue
                # Skip "AI system" if "high-risk AI system" already matched.
                if any(d.term.lower() in longer.lower() for longer in covered):
                    continue
                covered.append(d.term)
                add(d.chunk_id, "definition")

        if cfg.expand_cross_refs:
            for h in top:
                c = h.chunk
                if c.kind is not ChunkKind.ARTICLE or c.article is None:
                    continue
                for m in _PARAGRAPH_REF.finditer(c.text):
                    target = int(m.group(1))
                    if target == c.paragraph:
                        continue
                    siblings = self.corpus.article_chunks(c.regulation, c.article, target)
                    for s in siblings[:2]:
                        add(s.chunk_id, "cross_ref")
        return out
