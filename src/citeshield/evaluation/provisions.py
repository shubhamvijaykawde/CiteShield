"""Explicit legal-provision references in an answer, resolved against the canonical corpus.

The text patterns only *find* candidate references ("AI Act Article 16(2)", "Art. 33(1)
GDPR"). Whether a provision exists is decided by the corpus: the same ``golden.Ref``
matching the citation metrics use, over ``Corpus.article_chunks``, and, for articles whose
sub-units were not split into chunks, the canonical article text.

Each reference gets one status:

* ``exists``: the corpus has the article and the paragraph / point.
* ``nonexistent``: confirmed absent. Either the regulation has no such article, or the
  article is split into numbered paragraphs (or numbered points) and none matches, or an
  unsplit article's canonical text has no marker for that sub-unit.
* ``disputed``: would be ``nonexistent``, but the sentence itself says the reference is
  invalid ("there is no Article 16(2)"). Discussing a bad reference is not asserting it.
* ``ambiguous``: cannot be decided. No regulation named and the answer covers several
  regulations, two regulations in the same sentence, or a point the corpus does not split
  out and whose marker the article text does not show.
* ``out_of_corpus``: the reference names another act (a Directive, another Regulation).
* ``unparseable``: a citation-like string in a non-standard form ("Article 16.2").

Only ``nonexistent`` counts as a confirmed integrity failure. The regulation of a reference
comes from the act named right next to it, else the only regulation named in its sentence,
else the only regulation in the whole answer (its text and its citations). That basis is
recorded with every finding.

Scanned text: everything the model wrote that a reader sees (summary, claims, checklist
items, caveats). Citation quotes are left out: they are verbatim official text and were
checked against the corpus already.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from citeshield.contracts.chunk import ChunkRecord, corpus_hash
from citeshield.evaluation.golden import parse_ref
from citeshield.evaluation.records import AnswerRecord
from citeshield.retrieval.corpus import Corpus

CHECKER_VERSION = "provrefs-1"

# Official short titles beyond the corpus names. The "YYYY/N" numbers are derived from each
# regulation's CELEX number in the corpus, so they need no table.
TITLE_ALIASES: dict[str, tuple[str, ...]] = {
    "GDPR": ("General Data Protection Regulation",),
    "AI Act": ("Artificial Intelligence Act",),
    "DSA": ("Digital Services Act",),
    "NIS2": ("NIS 2 Directive", "NIS 2"),
    "Data Act": (),
}

_ART = re.compile(
    r"\b(?:Art(?:icle)?s?\.?)\s*(?P<art>\d{1,3})"
    r"(?P<subs>(?:\s?\((?:\d{1,3}|[a-z]{1,3})\))*)"
    r"(?:,?\s*(?:paragraph|para\.)\s*(?P<para_word>\d{1,3}))?",
    re.IGNORECASE,
)
_SUB = re.compile(r"\(([0-9]{1,3}|[a-z]{1,3})\)")
# Something citation-like right after a match that the grammar does not cover: "16.2",
# "16-2", or a bracket the sub-unit grammar did not accept ("16(zz9)").
_TRAILING_JUNK = re.compile(r"^(?:\.\d|-\d|\()")
_EXTERNAL_ACT = re.compile(
    r"\b(?:Directive|Regulation|Decision)\s*(?:\((?:EU|EC|EEC)\)\s*)?(?:No\s*)?\d{2,4}/\d+",
    re.IGNORECASE,
)
_NEGATION = re.compile(
    r"\b(?:does not exist|doesn't exist|do not exist|non-?existent|no such (?:article|paragraph|"
    r"provision|point)|has no (?:paragraph|point)|there is no|not a valid|invalid reference)\b",
    re.IGNORECASE,
)
_SENTENCE = re.compile(r"(?<!\bArt\.)(?<!\bArts\.)(?<!\bNo\.)(?<!\bpara\.)(?<=[.!?;])\s+|\n+")


class RefStatus(StrEnum):
    EXISTS = "exists"
    NONEXISTENT = "nonexistent"
    DISPUTED = "disputed"
    AMBIGUOUS = "ambiguous"
    OUT_OF_CORPUS = "out_of_corpus"
    UNPARSEABLE = "unparseable"


@dataclass(frozen=True)
class Finding:
    text: str  # the reference as written
    status: RefStatus
    regulation: str | None = None
    basis: str | None = None  # adjacent | sentence | answer
    reason: str = ""


class ProvisionChecker:
    def __init__(self, corpus: Corpus) -> None:
        self.corpus = corpus
        self.regulations = corpus.regulations
        celex: dict[str, str] = {}
        for c in corpus:
            celex.setdefault(c.regulation, c.celex)
        aliases: list[tuple[str, str]] = []
        for reg in self.regulations:
            names = [reg, *TITLE_ALIASES.get(reg, ())]
            cx = celex[reg]  # e.g. 32016R0679 -> 2016/679
            names.append(f"{cx[1:5]}/{int(cx[6:])}")
            aliases += [(n, reg) for n in names]
        # Longest first, so "NIS 2 Directive" wins over "NIS 2".
        aliases.sort(key=lambda a: -len(a[0]))
        self._aliases = aliases
        self._alias_re = re.compile(
            "|".join(rf"(?<![\w/]){re.escape(n)}(?![\w/])" for n, _ in aliases)
        )
        self._by_alias = {n.lower(): reg for n, reg in aliases}

    # ------------------------------------------------------------------ identity

    def meta(self) -> dict[str, str]:
        """Checker identity for run metadata: version, rules/aliases hash, corpus."""
        config = json.dumps(
            {"version": CHECKER_VERSION, "aliases": self._aliases, "art": _ART.pattern},
            sort_keys=True,
        )
        return {
            "version": CHECKER_VERSION,
            "config_sha": hashlib.sha256(config.encode()).hexdigest()[:12],
            "corpus": corpus_hash(self.corpus),
        }

    # ------------------------------------------------------------------ scanning

    def _regs_in(self, text: str) -> set[str]:
        return {self._by_alias[m.group(0).lower()] for m in self._alias_re.finditer(text)}

    def scan(self, rec: AnswerRecord) -> list[Finding]:
        """Findings for every explicit article reference the model wrote; [] without answer."""
        a = rec.answer
        if a is None:
            return []
        units = [a.summary, *a.caveats]
        units += [c.text for c in a.claims]
        units += [f"{i.obligation} {i.applies_to}" for i in a.checklist or []]
        answer_regs = self._regs_in("\n".join(units))
        answer_regs |= {
            c.regulation for cid in rec.cited_ids if (c := self.corpus.get(cid)) is not None
        }
        findings = []
        for unit in units:
            for sentence in _SENTENCE.split(unit):
                findings += self.scan_sentence(sentence, answer_regs)
        return findings

    def scan_sentence(self, sentence: str, answer_regs: Iterable[str] = ()) -> list[Finding]:
        out = []
        sentence_regs = self._regs_in(sentence)
        for m in _ART.finditer(sentence):
            raw = m.group(0).strip()
            after = sentence[m.end() :]
            if _TRAILING_JUNK.match(after):
                out.append(Finding(raw, RefStatus.UNPARSEABLE, reason="non-standard form"))
                continue
            before = sentence[: m.start()]
            reg, basis = self._regulation(before, after, sentence_regs, set(answer_regs))
            if reg is None:
                out.append(Finding(raw, RefStatus.AMBIGUOUS, reason=basis))
                continue
            if reg == "":
                out.append(Finding(raw, RefStatus.OUT_OF_CORPUS, reason="names another act"))
                continue
            subs = _SUB.findall(m.group("subs") or "")
            if m.group("para_word"):
                subs = [m.group("para_word"), *subs]
            status, reason = self.resolve(reg, int(m.group("art")), subs)
            if status is RefStatus.NONEXISTENT and _NEGATION.search(sentence):
                status, reason = RefStatus.DISPUTED, "the sentence calls it invalid"
            out.append(Finding(raw, status, reg, basis, reason))
        return out

    def _regulation(
        self, before: str, after: str, sentence_regs: set[str], answer_regs: set[str]
    ) -> tuple[str | None, str]:
        """(regulation, basis); ("", ...) for another act; (None, why) when ambiguous."""
        rest = re.sub(r"^\s*(?:of\s+(?:the\s+)?)?", "", after, count=1)
        if (ext := _EXTERNAL_ACT.match(rest)) is not None:
            named = self._regs_in(ext.group(0))  # "Regulation (EU) 2016/679" is the GDPR
            return (named.pop(), "adjacent") if len(named) == 1 else ("", "external")
        if (alias := self._alias_re.match(rest)) is not None:
            return self._by_alias[alias.group(0).lower()], "adjacent"
        # Only an act directly before the reference ("AI Act Article 16"); across a comma
        # ("the GDPR and NIS2, Article 33") it may belong to the clause, not the reference.
        stem = before.rstrip(" ")
        ends = [m for m in self._alias_re.finditer(stem) if m.end() == len(stem)]
        if ends:
            return self._by_alias[ends[-1].group(0).lower()], "adjacent"
        if len(sentence_regs) == 1:
            return next(iter(sentence_regs)), "sentence"
        if len(sentence_regs) > 1:
            return None, "several regulations in the sentence"
        if len(answer_regs) == 1:
            return next(iter(answer_regs)), "answer"
        if not answer_regs:
            return None, "no regulation named"
        return None, "several regulations in the answer"

    # ------------------------------------------------------------------ resolution

    def resolve(self, regulation: str, article: int, subs: Sequence[str]) -> tuple[RefStatus, str]:
        """Does ``regulation`` ``article`` (sub-units ``subs``, outermost first) exist?"""
        if regulation not in self.regulations:
            return RefStatus.OUT_OF_CORPUS, f"{regulation} is not in the corpus"
        chunks = self.corpus.article_chunks(regulation, article)
        if not chunks:
            return RefStatus.NONEXISTENT, f"{regulation} has no Article {article}"
        if not subs:
            return RefStatus.EXISTS, "article"
        label = f"{regulation} {article}" + "".join(f"({s})" for s in subs[:2])
        try:
            ref = parse_ref(label)
        except ValueError:
            return RefStatus.UNPARSEABLE, f"cannot parse {label!r}"
        if any(ref.matches(c) for c in chunks):
            return RefStatus.EXISTS, "corpus metadata"
        return self._absent(regulation, article, subs, chunks)

    def _absent(
        self, regulation: str, article: int, subs: Sequence[str], chunks: Sequence[ChunkRecord]
    ) -> tuple[RefStatus, str]:
        first = subs[0]
        where = f"{regulation} Article {article}"
        if first.isdigit():
            n = int(first)
            paragraphs = {c.paragraph for c in chunks if c.paragraph is not None}
            numbered_points = {c.point for c in chunks if c.point and c.point.isdigit()}
            has_n = n in paragraphs or str(n) in numbered_points
            if has_n:
                # The paragraph exists; the deeper point did not match the metadata.
                return self._point_in_text(
                    where, chunks, [c for c in chunks if c.paragraph == n], subs[1:]
                )
            if paragraphs or numbered_points:
                return RefStatus.NONEXISTENT, f"{where} has no paragraph or point {n}"
            # Unsplit article: look for the sub-unit's marker in the canonical text.
            if _marker(n, "\n".join(c.text for c in chunks)):
                return RefStatus.AMBIGUOUS, f"{where} text has a ({n}) / {n}. marker"
            return RefStatus.NONEXISTENT, f"{where} has no paragraph {n}"
        return self._point_in_text(where, chunks, chunks, subs)

    def _point_in_text(
        self,
        where: str,
        chunks: Sequence[ChunkRecord],
        scope: Sequence[ChunkRecord],
        points: Sequence[str],
    ) -> tuple[RefStatus, str]:
        if not points:
            return RefStatus.EXISTS, "corpus metadata"
        p = points[0]
        if any(c.point is not None for c in scope):
            return RefStatus.NONEXISTENT, f"{where} has no point ({p}) in the corpus"
        if _marker(p, "\n".join(c.text for c in scope)):
            return RefStatus.EXISTS, "canonical text marker"
        return RefStatus.AMBIGUOUS, f"point ({p}) not split out and no marker in {where}"


def _marker(unit: str | int, text: str) -> bool:
    """A sub-unit marker at a line start: "(2)", "2." or "(a)"."""
    u = re.escape(str(unit))
    return re.search(rf"(?:^|\n)\s*(?:\({u}\)|{u}\.)\s", text) is not None


def confirmed_nonexistent(findings: Iterable[Finding]) -> list[str]:
    return [f.text for f in findings if f.status is RefStatus.NONEXISTENT]
