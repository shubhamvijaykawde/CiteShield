"""Claim-level verification of a drafted ``AgentAnswer``.

For every claim and checklist item:
  1. each citation is checked against the canonical corpus and this run's evidence
     (fabricated / wrong version / non-citable / not retrieved);
  2. its quote must match the canonical text after deterministic normalisation;
  3. only citations that pass 1 and 2 go to NLI: the claim must be *entailed* by the cited
     text (a matching quote proves the passage exists, not that the claim follows from it).

Summary and caveat sentences are checked too: each must be entailed by the verified claims
or their supporting sources. Nothing the model writes reaches the user unverified.

Everything here is computed from the corpus, string comparison and NLI probabilities. No
field written by the drafting model can set a verdict.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from citeshield.agent.tools import EvidenceItem
from citeshield.contracts.answer import AgentAnswer, AnswerStatus, Citation
from citeshield.contracts.verification import (
    CitationCheck,
    CitationStatus,
    ClaimCheck,
    ClaimVerdict,
    Entailment,
    QuoteMatch,
    SentenceCheck,
    VerificationReport,
)
from citeshield.retrieval.corpus import Corpus
from citeshield.verify.citations import CitationChecker
from citeshield.verify.config import VerifierConfig
from citeshield.verify.nli import NLIModel, premise_windows
from citeshield.verify.quotes import match_quote

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(‘“])")
_MIN_SENTENCE_WORDS = 3


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]


def retrieval_top_score(evidence: Mapping[str, EvidenceItem]) -> float | None:
    """Best rerank score among binding sources the tools returned (None: no scored search)."""
    scores = [e.score for e in evidence.values() if e.score is not None and e.kind != "recital"]
    return max(scores) if scores else None


@dataclass
class _Item:
    kind: Literal["claim", "checklist"]
    index: int
    text: str
    citations: list[Citation]
    checks: list[CitationCheck] = field(default_factory=list)
    premises: dict[int, str] = field(default_factory=dict)  # usable citation index -> premise


class Verifier:
    def __init__(self, corpus: Corpus, nli: NLIModel, cfg: VerifierConfig | None = None) -> None:
        self.corpus = corpus
        self.nli = nli
        self.cfg = cfg or VerifierConfig()
        self.checker = CitationChecker(corpus)

    @property
    def thresholds(self) -> dict[str, float]:
        return {
            "entail": self.cfg.entail_threshold,
            "contradiction": self.cfg.contradiction_threshold,
            "sentence": self.cfg.sentence_threshold,
            "min_retrieval_score": self.cfg.min_retrieval_score,
        }

    def verify(
        self, answer: AgentAnswer, evidence: Mapping[str, EvidenceItem]
    ) -> VerificationReport:
        report = VerificationReport(
            nli_model=self.nli.name,
            thresholds=self.thresholds,
            retrieval_top_score=retrieval_top_score(evidence),
        )
        if answer.status is AnswerStatus.REFUSED:
            return report

        items = [_Item("claim", i, c.text, c.citations) for i, c in enumerate(answer.claims)]
        items += [
            _Item("checklist", i, f"{c.obligation} (applies to: {c.applies_to})", c.citations)
            for i, c in enumerate(answer.checklist or [])
        ]
        for item in items:
            self._check_citations(item, evidence)

        # One batched NLI pass over every (window, claim) pair.
        jobs: list[tuple[tuple[int, int], str, str]] = []  # ((item, citation|-1), premise, hyp)
        for n, item in enumerate(items):
            for j, premise in item.premises.items():
                jobs += [
                    ((n, j), w, item.text) for w in premise_windows(premise, item.text, self.nli)
                ]
            if len(item.premises) > 1:
                combined = "\n".join(item.premises.values())
                jobs += [
                    ((n, -1), w, item.text) for w in premise_windows(combined, item.text, self.nli)
                ]
        best = _best_by_key(jobs, self.nli.score([(p, h) for _, p, h in jobs]) if jobs else [])

        for n, item in enumerate(items):
            for j, check in enumerate(item.checks):
                if (n, j) in best:
                    check.entailment = best[(n, j)]
            report.claims.append(self._verdict(item, best.get((n, -1))))

        report.sentences = self._check_sentences(answer, report)
        return report

    # ------------------------------------------------------------------ steps

    def _check_citations(self, item: _Item, evidence: Mapping[str, EvidenceItem]) -> None:
        for j, cit in enumerate(item.citations):
            res = self.checker.check(cit.chunk_id, evidence)
            label = res.chunk.citation_label if res.chunk else None
            if res.status is not CitationStatus.VALID or res.chunk is None:
                item.checks.append(
                    CitationCheck(
                        chunk_id=cit.chunk_id,
                        label=label,
                        status=res.status,
                        quote_match=QuoteMatch.NOT_CHECKED,
                        reason=res.reason,
                    )
                )
                continue
            # Always compare with the canonical corpus text, never the evidence copy.
            quote = match_quote(
                cit.quote, res.chunk.text, min_segment_chars=self.cfg.min_quote_segment_chars
            )
            item.checks.append(
                CitationCheck(
                    chunk_id=cit.chunk_id,
                    label=label,
                    status=CitationStatus.VALID,
                    quote_match=QuoteMatch.MATCHED if quote.matched else QuoteMatch.NOT_FOUND,
                    quote_similarity=quote.similarity,
                    reason=quote.reason,
                )
            )
            if quote.matched:
                item.premises[j] = f"{label}: {res.chunk.text}"

    def _verdict(self, item: _Item, combined: Entailment | None) -> ClaimCheck:
        usable = [(j, c) for j, c in enumerate(item.checks) if c.usable]
        reasons = [f"{c.chunk_id}: {c.reason}" for c in item.checks if not c.usable and c.reason]
        check = ClaimCheck(
            kind=item.kind,
            index=item.index,
            text=item.text,
            verdict=ClaimVerdict.CITATION_INVALID,
            citations=item.checks,
            combined_entailment=combined,
            reasons=reasons,
        )
        if not usable:
            check.reasons.append("no citation passed the corpus, retrieval and quote checks")
            return check

        t_e, t_c = self.cfg.entail_threshold, self.cfg.contradiction_threshold
        entailing = [
            c.chunk_id for _, c in usable if c.entailment and c.entailment.entailment >= t_e
        ]
        contradicting = [
            c.chunk_id for _, c in usable if c.entailment and c.entailment.contradiction >= t_c
        ]
        combined_ok = combined is not None and combined.entailment >= t_e
        supported = bool(entailing) or combined_ok

        if supported and contradicting:
            check.verdict = ClaimVerdict.CONFLICTING
            check.reasons.append(f"entailed by {entailing or 'the combined sources'} "
                                 f"but contradicted by {contradicting}")  # fmt: skip
        elif supported:
            check.verdict = ClaimVerdict.SUPPORTED
            check.supporting_chunk_ids = entailing or [c.chunk_id for _, c in usable]
        elif contradicting:
            check.verdict = ClaimVerdict.CONTRADICTED
            check.reasons.append(f"contradicted by {contradicting}")
        else:
            best = max(
                [c.entailment.entailment for _, c in usable if c.entailment]
                + ([combined.entailment] if combined else [0.0])
            )
            check.verdict = ClaimVerdict.UNSUPPORTED
            check.reasons.append(
                f"cited text does not entail the claim (best P(entailment)={best:.2f} < {t_e})"
            )
        return check

    def _check_sentences(
        self, answer: AgentAnswer, report: VerificationReport
    ) -> list[SentenceCheck]:
        sentences: list[tuple[Literal["summary", "caveat"], str]] = [
            ("summary", s) for s in split_sentences(answer.summary)
        ]
        for caveat in answer.caveats:
            sentences += [("caveat", s) for s in split_sentences(caveat)]
        if not sentences:
            return []

        supported = report.supported
        if not supported:
            return [
                SentenceCheck(
                    source=src, text=s, kept=False, reason="no verified claim supports it"
                )
                for src, s in sentences
            ]
        premises = [" ".join(c.text for c in supported)]
        for claim in supported:
            for cit in claim.citations:
                if cit.chunk_id in claim.supporting_chunk_ids:
                    chunk = self.corpus[cit.chunk_id]
                    premises.append(f"{chunk.citation_label}: {chunk.text}")

        jobs: list[tuple[int, str, str]] = []
        for k, (_, sentence) in enumerate(sentences):
            if len(sentence.split()) < _MIN_SENTENCE_WORDS:
                continue
            for premise in dict.fromkeys(premises):
                jobs += [(k, w, sentence) for w in premise_windows(premise, sentence, self.nli)]
        scores = self.nli.score([(p, h) for _, p, h in jobs]) if jobs else []
        best: dict[int, float] = {}
        for (k, _, _), e in zip(jobs, scores, strict=True):
            best[k] = max(best.get(k, 0.0), e.entailment)

        out = []
        for k, (src, sentence) in enumerate(sentences):
            p = best.get(k)
            kept = p is not None and p >= self.cfg.sentence_threshold
            reason = None
            if p is None:
                reason = "too short to verify"
            elif not kept:
                reason = f"not entailed by the verified claims or their sources (P={p:.2f})"
            out.append(
                SentenceCheck(source=src, text=sentence, kept=kept, entailment=p, reason=reason)
            )
        return out


def _best_by_key(
    jobs: Sequence[tuple[tuple[int, int], str, str]], scores: Sequence[Entailment]
) -> dict[tuple[int, int], Entailment]:
    """Per key, the window with the highest entailment probability."""
    best: dict[tuple[int, int], Entailment] = {}
    for (key, _, _), e in zip(jobs, scores, strict=True):
        if key not in best or e.entailment > best[key].entailment:
            best[key] = e
    return best
