"""Splits documents into chunks. Each chunk keeps its own text and a copy of the document's metadata."""
from dataclasses import dataclass

from rag.loader import Doc


@dataclass
class Chunk:
    id: str            # "<doc id>#<number>", e.g. "of-210-city#2"
    doc_id: str
    text: str          # the original text, exactly as it will be shown to the model
    meta: dict


def chunk_fixed(doc: Doc, size: int = 350, overlap: int = 50) -> list[Chunk]:
    """Baseline: cut every `size` characters, with `overlap` characters repeated between neighbours."""
    chunks = []
    step = size - overlap
    for start in range(0, max(len(doc.text), 1), step):
        piece = doc.text[start:start + size]
        if piece.strip():
            chunks.append(Chunk(f"{doc.id}#{len(chunks)}", doc.id, piece, dict(doc.meta)))
        if start + size >= len(doc.text):
            break
    return chunks


def chunk_by_headings(doc: Doc) -> list[Chunk]:
    """TODO (step 2): one chunk per "## " section.

    - Every chunk starts with a heading path line: "<document title> › <section heading>".
      The text before the first "## " goes in a chunk whose path line is just the title.
    - Skip the "# " title line itself (the path line already carries the title).
    - Keep each section whole, so a table stays together with its header row.
    - Copy the document metadata and add meta["section"] = the section heading.
    - Ids: f"{doc.id}#{n}", numbered from 0.
    """
    title = doc.title  # front-matter title, e.g. "אופק סיטי (OF-210)"; includes the model code
    sections: list[tuple[str | None, list[str]]] = [(None, [])]  # (heading, body lines); None = intro
    for line in doc.text.splitlines():
        if line.startswith("## "):  # "### " and deeper do not match, so they stay in their parent section
            sections.append((line[3:].strip(), []))
        elif line.startswith("# ") and sections[-1][0] is None:
            continue  # the "# " title line: the path line already carries the title
        else:
            sections[-1][1].append(line)

    chunks = []
    for heading, lines in sections:
        body = "\n".join(lines).strip()
        if not body:
            continue  # heading (or intro) with no content
        path = f"{title} › {heading}" if heading else title
        meta = dict(doc.meta, section=heading or "")
        chunks.append(Chunk(f"{doc.id}#{len(chunks)}", doc.id, f"{path}\n{body}", meta))
    if not chunks:  # e.g. a document whose body is only its title: keep it findable
        chunks.append(Chunk(f"{doc.id}#0", doc.id, title, dict(doc.meta, section="")))
    return chunks
