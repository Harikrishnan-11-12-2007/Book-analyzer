"""
analyzer/nlp_engine.py

Core NLP logic for Book Analyzer.

Loads the spaCy model once at import time and exposes two functions:
    - analyze_text(text)      -> overall book-level analysis
    - analyze_chapters(text)  -> list of per-chapter analyses
"""

import re
from collections import Counter

import nltk
import spacy
from textstat import flesch_reading_ease

# ---------------------------------------------------------------------------
# One-time setup (runs once, at import time)
# ---------------------------------------------------------------------------

for _resource in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{_resource}")
    except LookupError:
        nltk.download(_resource, quiet=True)

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 3_000_000  # allow full-length books, not just short passages

GENRE_KEYWORDS = {
    "science": {"experiment", "research", "physics", "biology", "chemistry"},
    "history": {"war", "king", "empire", "revolution", "ancient"},
    "fiction": {"love", "magic", "adventure", "hero", "fantasy"},
    "philosophy": {"truth", "mind", "ethic", "existence", "reason"},
}

# Matches "Chapter 1", "chapter 12", etc. as a whole word + number,
# so it won't false-match inside words like "subchapter".
CHAPTER_PATTERN = re.compile(r"\bchapter\s+\d+\b", re.IGNORECASE)


def _lemmas(doc):
    """Lowercase lemmas for alphabetic, non-stopword tokens."""
    return [t.lemma_.lower() for t in doc if t.is_alpha and not t.is_stop]


def _detect_genres(lemmas):
    lemma_set = set(lemmas)
    genres = [g for g, words in GENRE_KEYWORDS.items() if lemma_set & words]
    return genres or ["General"]


def _short_description(top_keywords):
    if len(top_keywords) >= 2:
        return f"A gripping story about {top_keywords[0][0]} and {top_keywords[1][0]}."
    if top_keywords:
        return f"A unique book woven around {top_keywords[0][0]}."
    return "A unique book with intriguing themes."


def analyze_text(text):
    """Run full NLP analysis on a block of text and return a results dict.

    Always returns the same keys, even for empty/blank input, so callers
    and templates never have to special-case a missing field.
    """
    text = (text or "").strip()

    if not text:
        return {
            "keywords": [],
            "summary": "No summary available.",
            "quotes": ["No quotes found."],
            "readability": 0,
            "short_description": "No description available.",
            "genres": ["General"],
        }

    doc = nlp(text)
    lemmas = _lemmas(doc)
    top_keywords = Counter(lemmas).most_common(10)

    # Reuse spaCy's own sentence boundaries instead of re-tokenizing with
    # NLTK — one parse pass instead of two, and spaCy's boundaries are
    # generally more reliable on messy PDF-extracted text.
    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
    if not sentences:
        sentences = nltk.sent_tokenize(text)

    quotes = [s for s in sentences if '"' in s or "'" in s][:5]

    return {
        "keywords": top_keywords,
        "summary": " ".join(sentences[:3]) if sentences else "No summary available.",
        "quotes": quotes or ["No quotes found."],
        "readability": round(flesch_reading_ease(text), 1),
        "short_description": _short_description(top_keywords),
        "genres": _detect_genres(lemmas),
    }


def split_into_chapters(text):
    """Split book text on 'Chapter <n>' markers.

    Falls back to treating the whole text as one chapter if no markers
    are found, so short stories / single-chapter PDFs still work.
    """
    matches = list(CHAPTER_PATTERN.finditer(text))
    if not matches:
        return [text]

    chunks = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
    return chunks or [text]


def analyze_chapters(text):
    """Analyze each chapter and return a list of per-chapter result dicts."""
    chunks = split_into_chapters(text)
    results = []
    for i, chunk in enumerate(chunks, start=1):
        analysis = analyze_text(chunk)
        results.append({
            "chapter": i,
            "summary": analysis["summary"],
            "keywords": analysis["keywords"],
            "readability": analysis["readability"],
        })
    return results