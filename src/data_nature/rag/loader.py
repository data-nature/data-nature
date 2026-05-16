from __future__ import annotations

import re
from pathlib import Path

PAPER_INFO: dict[str, dict[str, str]] = {
    "Errors in Time-Series Remote Sensing and an Open Access Application for Detecting and Visualizing Spatial Data Outlier Using Google Earth Engine.pdf": {
        "short_title": "Time-Series Remote Sensing Errors & Outlier Detection",
        "topic": "Data quality, outlier detection, Google Earth Engine",
        "emoji": "📡",
    },
    "Examination of the Relationship Between Surface Temperature and Spectral Land Cover.pdf": {
        "short_title": "Surface Temperature and Land Cover Relationship",
        "topic": "LST, NDVI, land cover analysis",
        "emoji": "🌡️",
    },
    "Google Earth Engine Planetary-scale geospatial analysis for everyone.pdf": {
        "short_title": "Google Earth Engine: Planetary-Scale Geospatial Analysis",
        "topic": "Cloud computing, satellite imagery, GEE platform",
        "emoji": "🌍",
    },
    "Machine Learning Prediction of future LST.pdf": {
        "short_title": "Machine Learning Prediction of Future LST",
        "topic": "ML forecasting, Land Surface Temperature, time series",
        "emoji": "🤖",
    },
    "Spatial-RAG Spatial Retrieval Augmented Generation.pdf": {
        "short_title": "Spatial-RAG: Spatial Retrieval Augmented Generation",
        "topic": "RAG systems, spatial data, geospatial AI",
        "emoji": "🗺️",
    },
}

# ── Noise patterns ────────────────────────────────────────────────────────────

_RE_EMAIL  = re.compile(r"\S+@\S+\.\S+")
_RE_URL    = re.compile(r"https?://\S+|www\.\S+")
_RE_DOI    = re.compile(r"\bdoi\.org/\S+|\bDOI:\s*\S+", re.IGNORECASE)

# Lines that are almost certainly noise (short, numeric, affiliation-like)
_JUNK_LINE = re.compile(
    r"""
      ^\d+$                                    |  # bare page number
      ^\s*fig(?:ure)?\.?\s*\d+                 |  # "Fig. 3" captions
      ^\s*table\s*\d+                          |  # "Table 2" captions
      university|department|institute|ORCID       # affiliation noise
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _alpha_ratio(s: str) -> float:
    return sum(c.isalpha() for c in s) / len(s) if s else 0.0


def _clean_lines(raw: str) -> str:
    """Keep only lines that look like real prose; join into a single string."""
    out: list[str] = []
    for line in raw.splitlines():
        line = _RE_EMAIL.sub("", line)
        line = _RE_URL.sub("", line)
        line = _RE_DOI.sub("", line)
        line = line.strip()
        if len(line) < 35:              # too short → header / footer / lone number
            continue
        if _alpha_ratio(line) < 0.55:  # mostly numbers or symbols
            continue
        if _JUNK_LINE.search(line):
            continue
        out.append(line)
    return " ".join(out)


def _truncate_at_references(words: list[str]) -> list[str]:
    """Drop everything from the References / Bibliography section onward.

    We only search the last 40 % of the word list to avoid matching
    the word 'references' in the body text.
    """
    cutoff = int(len(words) * 0.60)
    tail = " ".join(words[cutoff:])
    m = re.search(r"\bReferences\b|\bBibliography\b", tail, re.IGNORECASE)
    if m:
        tail_words_before = len(tail[: m.start()].split())
        return words[: cutoff + tail_words_before]
    return words


# ── Main extraction ───────────────────────────────────────────────────────────

def _extract_chunks(
    pdf_path: Path,
    chunk_words: int = 250,
    overlap_words: int = 40,
    min_words: int = 80,
) -> list[dict]:
    """Return overlapping word-window chunks from *pdf_path*.

    Pipeline
    --------
    1. Extract text per page (preserves page breaks).
    2. Line-level noise filtering (removes author blocks, captions, lone numbers).
    3. Skip first ~150 words (title / author / abstract metadata).
    4. Truncate at References section.
    5. Sliding-window chunking with overlap.
    6. Final alpha-ratio filter on each chunk.
    """
    try:
        import pypdf  # noqa: PLC0415
        reader = pypdf.PdfReader(str(pdf_path))
        pages_text = [p.extract_text() or "" for p in reader.pages]
    except Exception:
        return []

    clean = _clean_lines("\n".join(pages_text))
    words = clean.split()

    # Skip the title / author / keywords block at the very start
    skip = min(150, max(0, len(words) // 10))
    words = words[skip:]

    words = _truncate_at_references(words)

    step = chunk_words - overlap_words
    chunks: list[dict] = []
    for i in range(0, len(words), step):
        piece = words[i : i + chunk_words]
        if len(piece) < min_words:
            break
        text = " ".join(piece)
        if _alpha_ratio(text) < 0.60:   # final sanity filter
            continue
        chunks.append(
            {
                "text": text,
                "source": pdf_path.name,
                "chunk_id": len(chunks),
            }
        )

    return chunks


def load_all_chunks(papers_dir: Path) -> list[dict]:
    """Return all chunks from every PDF in *papers_dir*."""
    all_chunks: list[dict] = []
    for pdf in sorted(papers_dir.glob("*.pdf")):
        all_chunks.extend(_extract_chunks(pdf))
    return all_chunks
