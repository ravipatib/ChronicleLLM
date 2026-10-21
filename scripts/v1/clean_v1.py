"""
clean_v1.py — Clean and quality-gate the v1 raw texts.

Reads   data/australia_1850_1950_v1/raw/{gutenberg,pga,ia,hansard}/*.txt
Writes  data/australia_1850_1950_v1/cleaned/<source>_<id>.txt
        data/australia_1850_1950_v1/clean_report.csv

On top of the v0 clean() in clean_texts.py this adds:
  - PGA header/footer stripping (HTML-derived and plain-text layouts)
  - an English check (wrong-language books slipped into v0)
  - a modern-contamination scrub: paragraphs with URLs, "ebook", Gutenberg or
    proofreader credits are removed, and editor/transcriber notes in {} or
    [Transcriber's note ...] are cut
  - corpus-wide paragraph de-duplication, since PGA "Collected" editions
    repeat stories already taken from gutenberg.org
  - for Internet Archive OCR: scanner watermark removal, re-joining words
    hyphenated across lines, running-head removal, and a word-validity
    filter: a vocabulary is built from the proofread Gutenberg/PGA text in the
    same run; OCR paragraphs under 90% known words are dropped, and a book is
    kept only if it is still >=95% known words and keeps half its text
  - Hansard is already clean text from XML, so it only gets whitespace
    normalisation (v0 clean() would delete its debate headings)

Usage:
    python scripts/v1/clean_v1.py
"""

import collections
import csv
import hashlib
import os
import re
import sys

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_DIR)
from clean_texts import clean  # noqa: E402  (v0 cleaner, reused)

DATA_DIR    = os.path.join(REPO_DIR, "data", "australia_1850_1950_v1")
RAW_DIR     = os.path.join(DATA_DIR, "raw")
CLEANED_DIR = os.path.join(DATA_DIR, "cleaned")
REPORT      = os.path.join(DATA_DIR, "clean_report.csv")
SOURCES     = ["gutenberg", "pga", "ia", "hansard"]   # proofread sources first win de-duplication

MIN_CHARS      = 5000
MIN_ENGLISH    = 0.25   # share of words that are common English function words
MIN_DEDUP_LEN  = 200    # only de-duplicate paragraphs at least this long
OCR_MIN_KEPT   = 0.5    # an OCR book must keep at least half its text after the paragraph filter
OCR_PARA_VOCAB = 0.90   # OCR paragraph: minimum share of words found in the proofread vocabulary
OCR_BOOK_VOCAB = 0.95   # OCR book: minimum share after filtering (proofread text scores ~0.999)
VOCAB_MIN_FREQ = 3

STOPWORDS = set("the and of to a in that it was he i his is with for as had you not be her on at by "
                "but this have which she they from all were my we one so said there would or an been "
                "their what no if when him".split())
MODERN = re.compile(r"https?://|www\.|\be-?book\b|project gutenberg|gutenberg\.|"
                    r"distributed proofread|transcriber'?s note|\bpgdp\b", re.I)


def strip_pga(text):
    """Remove the PGA header (up to the licence line or dashed rule) and footer."""
    # Header: cut after the last licence/boilerplate marker in the opening section.
    head_markers = [
        r"View our\s+licence\s+and\s+header[^\n]*\n",
        r"Google Site Search[^\n]*\n",
        r"gutenberg\.net\.au/licence\.html[^\n]*\n",
        r"To contact Project Gutenberg (of )?Australia go to[^\n]*\n",
        r"Licence which may be viewed online[^\n]*\n",
        r"Australia\s+HOME PAGE[^\n]*\n",
        r"\n-{20,}\s*\n",
    ]
    end = 0
    for pat in head_markers:
        for m in re.finditer(pat, text[:10000]):
            end = max(end, m.end())
    text = text[end:]
    # Footer: older pages end with a site banner, newer ones with the project name.
    for pat in [r"This site is full of FREE ebooks", r"Project Gutenberg (of )?Australia"]:
        hits = [m.start() for m in re.finditer(pat, text)]
        if hits and hits[-1] > len(text) * 0.8:
            text = text[:hits[-1]]
    return re.sub(r"\s*THE END\s*$", "", text.rstrip())


def scrub_modern(text):
    """Drop editorial notes and any paragraph carrying a modern marker; return (text, n_removed)."""
    text = re.sub(r"\{[^{}]{0,400}\}", "", text)
    text = re.sub(r"\[(Transcriber|Editor)'?s? [Nn]ote[^\]]{0,2000}\]", "", text)
    text = re.sub(r"^E-text prepared by.*$", "", text, flags=re.MULTILINE)
    paras = text.split("\n\n")
    kept  = [p for p in paras if not MODERN.search(p)]
    return "\n\n".join(kept), len(paras) - len(kept)


OCR_JUNK = re.compile(r"^.*(Digitized by|digitized by the Internet Archive|archive\.org|Google|"
                      r"University of .{0,40} Librar|Original from|Generated on|Public Domain, Google).*$",
                      re.MULTILINE)


def vocab_share(text, vocab):
    words = re.findall(r"[a-z]+", text.lower())
    return sum(w in vocab for w in words) / max(len(words), 1)


def clean_ocr(text, vocab):
    """Repair and filter Internet Archive OCR text; return (text, share of text kept)."""
    text = OCR_JUNK.sub("", text)
    text = re.sub(r"(\w)[-¬]\s*\n\s*(\w)", r"\1\2", text)       # re-join hyphenated words
    lines = text.split("\n")
    counts = {}
    for l in lines:
        key = re.sub(r"[\d\W]+", "", l.lower())
        if 0 < len(l.strip()) <= 60:
            counts[key] = counts.get(key, 0) + 1
    # Running heads (book title / chapter title repeated on every page) and page numbers
    lines = [l for l in lines
             if not (0 < len(l.strip()) <= 60 and counts.get(re.sub(r"[\d\W]+", "", l.lower()), 0) >= 5)
             and not re.match(r"^\s*[\divxlcIVXLC.\-—*]{1,8}\s*$", l)]
    # Re-wrap: OCR keeps one printed line per line; join lines inside a paragraph.
    paras = re.split(r"\n\s*\n", "\n".join(lines))
    paras = [" ".join(p.split()) for p in paras]
    before = sum(len(p) for p in paras) or 1
    good = []
    for p in paras:
        words = re.findall(r"[A-Za-z]+", p)
        if not words:
            continue
        letters = sum(c.isalpha() for c in p) / len(p)
        stops   = sum(w.lower() in STOPWORDS for w in words) / len(words)
        if len(p) < 80 or (letters > 0.7 and stops > 0.18 and vocab_share(p, vocab) >= OCR_PARA_VOCAB):
            good.append(p)
    kept = "\n\n".join(good)
    # Watermarks split across OCR lines survive the line filter; catch them after re-wrapping.
    kept = re.sub(r"Digitized\s+by\s+(the|tine)?\s*Internet\s+Ar\S*(\s+in\s+\d{4})?"
                  r"(\s+with\s+funding\s+from[^.\n]{0,150})?", "", kept)
    kept = re.sub(r"Digitized\s+by\s*\S{0,6}(OOQ|oog|OOg|ooQ)\S{0,6}(\s+[A-Za-z]{1,3}\b)?!?", "", kept)
    kept = re.sub(r"Digitized\s+by\b", "", kept)
    return kept, len(kept) / before


def english_ratio(text):
    words = re.findall(r"[a-z]+", text[:200000].lower())
    return sum(w in STOPWORDS for w in words) / max(len(words), 1)


def main():
    os.makedirs(CLEANED_DIR, exist_ok=True)
    seen_paras = set()
    report     = []
    kept_chars = 0
    word_count = collections.Counter()   # from proofread sources, for the OCR filter
    vocab      = set()

    for source in SOURCES:
        if source == "ia":
            vocab = {w for w, n in word_count.items() if n >= VOCAB_MIN_FREQ}
        src_dir = os.path.join(RAW_DIR, source)
        if not os.path.isdir(src_dir):
            continue
        for fname in sorted(os.listdir(src_dir)):
            if not fname.endswith(".txt"):
                continue
            raw = open(os.path.join(src_dir, fname), encoding="utf-8", errors="ignore").read()
            ocr_kept = 1.0
            if source == "hansard":
                text = re.sub(r"[ \t]+", " ", raw).strip()
                text = re.sub(r"\. — [.\s-]*-\s*", ". — ", text)   # "Mr X. — . - I move" -> "Mr X. — I move"
            elif source == "ia":
                text, ocr_kept = clean_ocr(raw, vocab)
                text = clean(text)
            else:
                text = clean(strip_pga(raw) if source == "pga" else raw)
            text, modern = scrub_modern(text)

            # Paragraph de-duplication across the whole corpus
            paras, dropped = [], 0
            for p in text.split("\n\n"):
                key = hashlib.md5(re.sub(r"\W+", "", p.lower()).encode()).hexdigest()
                if len(p) >= MIN_DEDUP_LEN and key in seen_paras:
                    dropped += 1
                    continue
                seen_paras.add(key)
                paras.append(p)
            text = "\n\n".join(paras).strip()

            ratio = english_ratio(text)
            if len(text) < MIN_CHARS:
                status = "skip_short"
            elif ratio < MIN_ENGLISH:
                status = "skip_not_english"
            elif source == "ia" and (ocr_kept < OCR_MIN_KEPT or vocab_share(text, vocab) < OCR_BOOK_VOCAB):
                status = "skip_poor_ocr"
            else:
                status = "ok"
                out_name = f"{source}_{fname}"
                with open(os.path.join(CLEANED_DIR, out_name), "w", encoding="utf-8") as f:
                    f.write(text)
                kept_chars += len(text)
                if source in ("gutenberg", "pga"):
                    word_count.update(re.findall(r"[a-z]+", text.lower()))

            report.append({"source": source, "file": fname, "status": status, "chars": len(text),
                           "english_ratio": f"{ratio:.2f}", "ocr_kept": f"{ocr_kept:.2f}",
                           "modern_paras_removed": modern,
                           "dup_paras_dropped": dropped})

    with open(REPORT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        writer.writeheader()
        writer.writerows(report)

    counts = {}
    for r in report:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"Files   : {len(report)}  {counts}")
    print(f"Cleaned : {kept_chars/1e6:.1f}MB kept")
    print(f"Report  : {REPORT}")


if __name__ == "__main__":
    main()
