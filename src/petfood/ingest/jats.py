"""Parse JATS full-text XML (Europe PMC / PMC format) into ordered sections."""
from __future__ import annotations

import re

from lxml import etree

from petfood.models import Section

_SECTION_TYPES = [
    ("introduction", r"intro|background"),
    ("methods", r"method|material|design|procedure|experimental|animals"),
    ("results", r"result|finding"),
    ("discussion", r"discussion"),
    ("conclusion", r"conclusion|summary|implication|clinical relevance"),
]
_SKIP_HEADINGS = re.compile(
    r"acknowledg|funding|conflict|competing|author contribution|ethic|data availab|"
    r"supplementary|abbreviation|disclosure|declaration", re.I)
_SKIP_TAGS = {"table-wrap", "fig", "disp-formula", "supplementary-material", "xref-group"}


def _classify(heading: str | None, sec_type_attr: str | None) -> str:
    probe = f"{sec_type_attr or ''} {heading or ''}".lower()
    for name, pat in _SECTION_TYPES:
        if re.search(pat, probe):
            return name
    return "other"


def _text(el) -> str:
    """Text of an element, skipping tables/figures/formulas, whitespace-normalized."""
    parts: list[str] = []

    def walk(node):
        if node.tag in _SKIP_TAGS:
            if node.tail:
                parts.append(node.tail)
            return
        if node.text:
            parts.append(node.text)
        for child in node:
            walk(child)
        if node is not el and node.tail:
            parts.append(node.tail)

    walk(el)
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def _cell_text(cell) -> str:
    return re.sub(r"\s+", " ", "".join(cell.itertext())).strip().replace("|", "/")


def table_to_text(tw) -> str | None:
    """Render a <table-wrap> as caption + markdown-style rows + footnotes.

    Markdown rows keep the header/cell structure readable for both the LLM and BM25
    (numbers like "32.5" and terms like "crude protein" stay on the same line as their label).
    Row/col spans are flattened (cell text repeated once) — good enough for retrieval.
    """
    label = tw.find("./label")
    caption = tw.find("./caption")
    head = " ".join(x for x in [
        _text(label) if label is not None else "",
        _text(caption) if caption is not None else "",
    ] if x).strip()
    rows: list[str] = []
    for tr in tw.iter("tr"):
        cells = [_cell_text(c) for c in tr if c.tag in ("td", "th")]
        if any(cells):
            rows.append("| " + " | ".join(cells) + " |")
            if tr.getparent() is not None and tr.getparent().tag == "thead" and len(rows) == 1:
                rows.append("|" + "---|" * len(cells))
    foot = tw.find("./table-wrap-foot")
    foot_txt = _text(foot) if foot is not None else ""
    if not rows and not head:
        return None
    return "\n".join(x for x in [head, *rows, foot_txt] if x)


def _add_floats(root, sections: list[Section]) -> None:
    """Tables and figure captions (anywhere in body or floats-group) as their own sections."""
    for tw in root.iter("table-wrap"):
        t = table_to_text(tw)
        if t:
            first = t.split("\n", 1)[0]
            sections.append(Section(ord=len(sections), section_type="table",
                                    heading=first[:200] if not first.startswith("|") else "Table", text=t))
    for fig in root.iter("fig"):
        cap = fig.find("./caption")
        if cap is None:
            continue
        label = fig.find("./label")
        txt = " ".join(x for x in [_text(label) if label is not None else "", _text(cap)] if x)
        if len(txt.split()) >= 5:
            sections.append(Section(ord=len(sections), section_type="figure", heading=txt[:200], text=txt))


def _parse(xml: bytes):
    parser = etree.XMLParser(recover=True, resolve_entities=False, no_network=True, huge_tree=False)
    return etree.fromstring(xml, parser=parser)


def parse_front_meta(xml: bytes) -> dict:
    """Article type and the year the manuscript was received (from <history>).

    A study's data must exist before the paper is submitted, so received_year is a hard upper
    bound on the study period, tighter than the publication year.
    """
    root = _parse(xml)
    if root is None:
        return {"article_type": None, "received_year": None}
    year_el = root.find(".//front//history/date[@date-type='received']/year")
    year = (year_el.text or "").strip() if year_el is not None else ""
    return {
        "article_type": root.get("article-type"),
        "received_year": int(year) if year.isdigit() else None,
    }


def parse_jats(xml: bytes) -> tuple[str | None, list[Section]]:
    """Return (abstract_text, sections). Nested <sec> are flattened under the top-level type."""
    root = _parse(xml)
    if root is None:
        return None, []

    sections: list[Section] = []
    abstract = None
    abs_el = root.find(".//front//abstract")
    if abs_el is not None:
        abstract = _text(abs_el) or None
        if abstract:
            sections.append(Section(ord=0, section_type="abstract", heading="Abstract", text=abstract))

    body = root.find(".//body")
    if body is None:
        return abstract, sections

    top_secs = body.findall("./sec")
    if not top_secs:  # some articles have a flat body of <p>
        txt = " ".join(_text(p) for p in body.findall(".//p"))
        if txt:
            sections.append(Section(ord=len(sections), section_type="other", heading=None, text=txt))
        _add_floats(root, sections)
        return abstract, sections

    for sec in top_secs:
        title_el = sec.find("./title")
        heading = _text(title_el) if title_el is not None else None
        if heading and _SKIP_HEADINGS.search(heading):
            continue
        stype = _classify(heading, sec.get("sec-type"))
        # gather paragraphs (including nested subsections), keeping sub-headings inline
        chunks: list[str] = []
        for node in sec.iter("p", "title"):
            if node is title_el:
                continue
            t = _text(node)
            if not t:
                continue
            chunks.append(f"{t}:" if node.tag == "title" else t)
        text = " ".join(chunks).strip()
        if text:
            sections.append(Section(ord=len(sections), section_type=stype, heading=heading, text=text))

    _add_floats(root, sections)
    return abstract, sections
