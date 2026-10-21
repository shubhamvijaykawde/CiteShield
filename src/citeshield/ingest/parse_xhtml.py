"""Parse Official Journal XHTML (Cellar "CONVEX" output) into raw chunk records.

The output is deliberately *unvalidated*: this module only recovers structure as faithfully
as it can. ``validate.py`` applies the data contract and decides what gets indexed.

Markup relied on (stable across OJ acts since ~2014):
  * ``div#art_N``                articles, with ``div.eli-title`` for the article title
  * ``div#NNN.MMM``              numbered paragraph MMM of article NNN
  * ``table`` inside a paragraph points (a), (b), ... as label/text rows
  * ``div#rct_N`` / ``div#anx_X`` recitals / annexes
  * ``div#cpt_X[.sct_N]``        chapter / section wrappers (used for breadcrumbs)
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from lxml import etree

from citeshield.contracts.chunk import (
    ChunkKind,
    TrustTier,
    estimate_tokens,
    make_chunk_id,
    sha256_text,
)
from citeshield.ingest.manifest import SourceSpec

RawRecord = dict[str, object]
Element = etree._Element

# Any whitespace except newline (str patterns are Unicode-aware, so NBSP/en-space match too).
_WS = re.compile(r"[^\S\n]+")
_ARTICLE_ID = re.compile(r"^art_(\d+)$")
_PARA_ID = re.compile(r"^(\d{3})\.(\d{3})$")
_RECITAL_ID = re.compile(r"^rct_(\d+)$")
_ANNEX_ID = re.compile(r"^anx_([IVXLC]+)$")
_CHAPTER_ID = re.compile(r"^cpt_[^.]+(\.sct_\d+)?$")
_POINT_LABEL = re.compile(r"^\(?([a-z]{1,3}|\d{1,3})\)?\.?$")
_SENTENCE_END = re.compile(r"(?<=[.;:])\s+")
_ANNEX_ITEM = re.compile(r"^(\d{1,2}\.\s|(Section|SECTION|Part|PART) [A-Z0-9]+\b)")
_BLOCK_TAGS = frozenset({"p", "div", "table"})
_TITLE_CLASSES = frozenset({"oj-ti-art", "eli-title"})


@dataclass
class DocumentToc:
    """What the document says it contains; validation checks records against it."""

    articles: list[int] = field(default_factory=list)
    paragraphs: dict[int, list[int]] = field(default_factory=dict)
    recitals: list[int] = field(default_factory=list)
    annexes: list[str] = field(default_factory=list)


@dataclass
class ParseResult:
    records: list[RawRecord]
    toc: DocumentToc
    warnings: list[str]


def parse_document(
    data: bytes,
    src: SourceSpec,
    *,
    max_tokens: int = 512,
    trust_tier: TrustTier = TrustTier.OFFICIAL,
) -> ParseResult:
    root = _parse_tree(data)
    toc = DocumentToc()
    warnings: list[str] = []
    build = _RecordBuilder(src, trust_tier)

    records = [
        *_recitals(root, build, max_tokens, toc, warnings),
        *_articles(root, build, max_tokens, toc, warnings),
        *_annexes(root, build, max_tokens, toc, warnings),
    ]
    return ParseResult(records, toc, warnings)


# --------------------------------------------------------------------------- tree helpers


def _parse_tree(data: bytes) -> Element:
    # Untrusted input: no DTD loading, no entity expansion, no network access (XXE-safe).
    parser = etree.XMLParser(
        resolve_entities=False, no_network=True, load_dtd=False, remove_comments=True
    )
    root = etree.fromstring(data, parser)
    for el in root.iter():
        if isinstance(el.tag, str):
            el.tag = etree.QName(el).localname
    return root


def _elements(el: Element, expr: str) -> list[Element]:
    result = el.xpath(expr)
    if not isinstance(result, list):
        return []
    return [e for e in result if isinstance(e, etree._Element)]


def _id(el: Element) -> str:
    return el.get("id") or ""


def _classes(el: Element) -> set[str]:
    return set((el.get("class") or "").split())


def _normalise(text: str) -> str:
    lines = (_WS.sub(" ", line).strip() for line in text.split("\n"))
    return "\n".join(line for line in lines if line)


def _render(el: Element) -> str:
    """Flatten an element to text: blocks become lines, table rows become 'label text'."""
    if "oj-note-tag" in _classes(el):  # footnote markers
        return ""
    if el.tag == "table":
        rows = []
        for tr in _elements(el, "./tbody/tr | ./tr"):
            cells = [_render(td) for td in tr.findall("td")]
            rows.append(" ".join(c for c in cells if c))
        return "\n".join(r for r in rows if r)

    # Source newlines inside inline content are pretty-printing, not structure.
    pieces = [(el.text or "").replace("\n", " ")]
    for child in el:
        if not isinstance(child.tag, str):
            pieces.append((child.tail or "").replace("\n", " "))
            continue
        rendered = _render(child)
        pieces.append(f"\n{rendered}\n" if child.tag in _BLOCK_TAGS else rendered)
        pieces.append((child.tail or "").replace("\n", " "))
    return _normalise("".join(pieces))


def _pack(lines: list[str], max_tokens: int) -> list[str]:
    """Greedily pack lines (sentence-split if a line alone is too long) into chunks."""
    units: list[str] = []
    for line in lines:
        units.extend(_SENTENCE_END.split(line) if estimate_tokens(line) > max_tokens else [line])
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for unit in units:
        tokens = estimate_tokens(unit)
        if current and current_tokens + tokens > max_tokens:
            chunks.append("\n".join(current))
            current, current_tokens = [], 0
        current.append(unit)
        current_tokens += tokens
    if current:
        chunks.append("\n".join(current))
    return chunks


def _point_label(table: Element) -> str:
    first_row = _elements(table, "(./tbody/tr | ./tr)[1]")
    if not first_row:
        return ""
    cells = [_render(td) for td in first_row[0].findall("td")]
    label = next((c for c in cells if c), "")
    m = _POINT_LABEL.match(label)
    # An unparseable label is passed through as-is so the contract quarantines it visibly.
    return m.group(1) if m else label


def _chapter_headings(el: Element) -> list[str]:
    heads = []
    for anc in el.iterancestors("div"):
        aid = anc.get("id", "")
        if not _CHAPTER_ID.match(aid):
            continue
        label = next((_render(c) for c in anc if c.tag == "p"), "")
        title_el = anc.find(f'div[@id="{aid}.tit_1"]')
        title = _render(title_el) if title_el is not None else ""
        heads.append(" — ".join(x.replace("\n", " ") for x in (label, title) if x))
    return heads[::-1]


# --------------------------------------------------------------------------- records


class _RecordBuilder:
    def __init__(self, src: SourceSpec, trust_tier: TrustTier) -> None:
        self.src = src
        self.trust_tier = trust_tier

    def __call__(
        self,
        *,
        kind: ChunkKind,
        text: str,
        heading_path: list[str],
        anchor: str,
        article: int | None = None,
        paragraph: int | None = None,
        subparagraph: int | None = None,
        point: str | None = None,
        recital: int | None = None,
        annex: str | None = None,
        part: int | None = None,
    ) -> RawRecord:
        loc: dict[str, object] = {
            "article": article,
            "paragraph": paragraph,
            "subparagraph": subparagraph,
            "point": point,
            "recital": recital,
            "annex": annex,
            "part": part,
        }
        chunk_id = make_chunk_id(
            celex=self.src.celex,
            kind=kind,
            version=self.src.version,
            article=article,
            paragraph=paragraph,
            subparagraph=subparagraph,
            point=point,
            recital=recital,
            annex=annex,
            part=part,
        )
        return {
            "chunk_id": chunk_id,
            "regulation": self.src.regulation,
            "celex": self.src.celex,
            "kind": kind.value,
            **loc,
            "heading_path": heading_path,
            "text": text,
            "source_url": f"{self.src.eurlex_url}#{anchor}",
            "version": self.src.version,
            "version_date": self.src.version_date.isoformat(),
            "language": "en",
            "content_sha256": sha256_text(text),
            "n_tokens": estimate_tokens(text),
            "trust_tier": self.trust_tier.value,
        }


def _recitals(
    root: Element, build: _RecordBuilder, max_tokens: int, toc: DocumentToc, warnings: list[str]
) -> Iterator[RawRecord]:
    for div in _elements(root, '//div[starts-with(@id, "rct_")]'):
        m = _RECITAL_ID.match(_id(div))
        if not m:
            warnings.append(f"unrecognised recital id {_id(div)!r}")
            continue
        n = int(m.group(1))
        toc.recitals.append(n)
        text = re.sub(rf"^\({n}\)\s*", "", _render(div))
        pieces = (
            _pack(text.split("\n"), max_tokens) if estimate_tokens(text) > max_tokens else [text]
        )
        for i, piece in enumerate(pieces, 1):
            yield build(
                kind=ChunkKind.RECITAL,
                text=piece,
                heading_path=[build.src.regulation, "Recitals", f"Recital {n}"],
                anchor=_id(div),
                recital=n,
                part=i if len(pieces) > 1 else None,
            )


def _articles(
    root: Element, build: _RecordBuilder, max_tokens: int, toc: DocumentToc, warnings: list[str]
) -> Iterator[RawRecord]:
    for art in _elements(root, '//div[starts-with(@id, "art_") and not(contains(@id, "."))]'):
        art_id = _id(art)
        m = _ARTICLE_ID.match(art_id)
        if not m:
            warnings.append(f"unrecognised article id {art_id!r}")
            continue
        n = int(m.group(1))
        toc.articles.append(n)

        title = " ".join(_render(t) for t in art if "eli-title" in _classes(t)).replace("\n", " ")
        heading = [
            build.src.regulation,
            *_chapter_headings(art),
            f"Article {n}" + (f" — {title}" if title else ""),
        ]

        # Group the article body by numbered paragraph. Loose blocks before the first
        # paragraph form an unnumbered unit; loose blocks after one belong to it.
        groups: list[tuple[int | None, list[Element]]] = []
        for child in art:
            if not isinstance(child.tag, str) or _classes(child) & _TITLE_CLASSES:
                continue
            pm = _PARA_ID.match(child.get("id", "")) if child.tag == "div" else None
            if pm:
                if int(pm.group(1)) != n:
                    warnings.append(f"Article {n}: paragraph div {_id(child)} out of place")
                groups.append((int(pm.group(2)), [c for c in child if isinstance(c.tag, str)]))
            elif groups:
                groups[-1][1].append(child)
            else:
                groups.append((None, [child]))

        toc.paragraphs[n] = [p for p, _ in groups if p is not None]
        for paragraph, elements in groups:
            yield from _unit_records(
                build,
                elements,
                max_tokens,
                article=n,
                paragraph=paragraph,
                heading=heading,
                anchor=art_id,
            )


def _unit_records(
    build: _RecordBuilder,
    elements: list[Element],
    max_tokens: int,
    *,
    article: int,
    paragraph: int | None,
    heading: list[str],
    anchor: str,
) -> Iterator[RawRecord]:
    rendered = [(el, text) for el in elements if (text := _render(el))]
    if not rendered:
        return

    def rec(
        text: str,
        subparagraph: int | None = None,
        point: str | None = None,
        part: int | None = None,
    ) -> RawRecord:
        return build(
            kind=ChunkKind.ARTICLE,
            text=text,
            heading_path=heading,
            anchor=anchor,
            article=article,
            paragraph=paragraph,
            subparagraph=subparagraph,
            point=point,
            part=part,
        )

    full = "\n".join(text for _, text in rendered)
    if estimate_tokens(full) <= max_tokens:
        yield rec(full)
        return

    first_table = next((i for i, (el, _) in enumerate(rendered) if el.tag == "table"), None)
    if first_table is None:
        for i, piece in enumerate(_pack(full.split("\n"), max_tokens), 1):
            yield rec(piece, part=i)
        return

    # Too long, with points: one chunk per point. The lead-in sentence is repeated on each
    # point chunk so every chunk reads on its own ("1. Personal data shall be: (a) ...").
    # Text after a list opens a new subparagraph, which may carry its own (a), (b) list
    # (e.g. DSA Art. 51(3)); the subparagraph number keeps those points distinct.
    lead = "\n".join(text for _, text in rendered[:first_table])
    units: list[tuple[int, str | None, str]] = []  # (subparagraph, point label, text)
    subparagraph = 1
    loose: list[str] = []
    for el, text in rendered[first_table:]:
        if el.tag != "table":
            loose.append(text)
            continue
        if loose:
            subparagraph += 1
            units.append((subparagraph, None, "\n".join(loose)))
            loose = []
        units.append((subparagraph, _point_label(el), text))
    if loose:
        units.append((subparagraph + 1, None, "\n".join(loose)))

    for sp, label, text in units:
        body = f"{lead}\n{text}" if lead else text
        pieces = (
            _pack(body.split("\n"), max_tokens) if estimate_tokens(body) > max_tokens else [body]
        )
        for i, piece in enumerate(pieces, 1):
            yield rec(
                piece,
                subparagraph=sp if sp > 1 else None,
                point=label,
                part=i if len(pieces) > 1 else None,
            )


def _annexes(
    root: Element, build: _RecordBuilder, max_tokens: int, toc: DocumentToc, warnings: list[str]
) -> Iterator[RawRecord]:
    for anx in _elements(root, '//div[starts-with(@id, "anx_") and not(contains(@id, "."))]'):
        m = _ANNEX_ID.match(_id(anx))
        if not m:
            warnings.append(f"unrecognised annex id {_id(anx)!r}")
            continue
        roman = m.group(1)
        toc.annexes.append(roman)
        titles = [_render(c) for c in anx if "oj-doc-ti" in _classes(c)]
        heading = [build.src.regulation, " — ".join(titles) or f"Annex {roman}"]
        body = [c for c in anx if isinstance(c.tag, str) and "oj-doc-ti" not in _classes(c)]
        lines = [ln for ln in "\n".join(_render(c) for c in body).split("\n") if ln]
        # Never pack across a numbered item or section header ("4. Employment, ...") so that
        # "Annex III point 4" stays one citable chunk.
        sections: list[list[str]] = []
        for line in lines:
            if not sections or _ANNEX_ITEM.match(line):
                sections.append([])
            sections[-1].append(line)
        pieces = [piece for section in sections for piece in _pack(section, max_tokens)]
        for i, piece in enumerate(pieces, 1):
            yield build(
                kind=ChunkKind.ANNEX,
                text=piece,
                heading_path=heading,
                anchor=_id(anx),
                annex=roman,
                part=i,
            )
