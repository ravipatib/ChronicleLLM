"""
clean_texts.py — Preprocess raw Project Gutenberg texts for Chronicle LLM.

Reads all .txt files from data/australia_1850_1950/raw/, strips Gutenberg
headers/footers, OCR artifacts, illustration tags, and excessive whitespace,
then writes cleaned files to data/australia_1850_1950/cleaned/.

Usage:
    python clean_texts.py
"""

import os
import re

RAW_DIR     = "data/australia_1850_1950/raw"
CLEANED_DIR = "data/australia_1850_1950/cleaned"


def clean(text):
    """Remove boilerplate and normalise whitespace from a Gutenberg text."""

    # 1. Remove BOM if present
    text = text.lstrip('\ufeff')

    # 2. Strip Gutenberg.org header (everything before *** START OF ***)
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG",
        "*** START OF THIS PROJECT GUTENBERG",
        "*END*THE SMALL PRINT",
    ]
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            text = text[text.find('\n', idx) + 1:]
            break

    # 3. Strip Gutenberg.org footer
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG",
        "*** END OF THIS PROJECT GUTENBERG",
        "End of the Project Gutenberg",
        "End of Project Gutenberg",
        "THE END OF THE PROJECT GUTENBERG",
    ]
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
            break

    # 4. Strip gutenberg.net.au header (ends with a line of dashes)
    lines = text.split('\n')
    dash_line_idx = None
    for i, line in enumerate(lines[:50]):
        stripped = line.strip()
        if len(stripped) > 10 and re.match(r'^[-=*]{10,}$', stripped):
            dash_line_idx = i
            break
    if dash_line_idx is not None:
        header_text = '\n'.join(lines[:dash_line_idx]).lower()
        if any(word in header_text for word in
               ['gutenberg', 'ebook', 'copyright', 'transcribed', 'produced']):
            lines = lines[dash_line_idx + 1:]
            text = '\n'.join(lines)

    # 5. Remove "Produced by..." lines
    text = re.sub(
        r'^(Produced by|Transcribed by|Prepared by|HTML version by).*$',
        '', text, flags=re.MULTILINE | re.IGNORECASE
    )

    # 6. Remove illustration/photo tags
    text = re.sub(r'\[Illustration[^\]]*\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[Frontispiece[^\]]*\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[Photo[^\]]*\]', '', text, flags=re.IGNORECASE)

    # 7. Remove page markers
    text = re.sub(r'\[Page \d+\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

    # 8. Remove lines that are just dashes/equals/stars
    text = re.sub(r'^\s*[-=*_]{4,}\s*$', '', text, flags=re.MULTILINE)

    # 9. Normalise whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\t', ' ', text)

    return text.strip()


if __name__ == "__main__":
    os.makedirs(CLEANED_DIR, exist_ok=True)

    files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith('.txt')])
    print(f"Found {len(files)} files to clean\n")

    total_raw     = 0
    total_cleaned = 0
    skipped       = 0

    for fname in files:
        raw_path     = os.path.join(RAW_DIR, fname)
        cleaned_path = os.path.join(CLEANED_DIR, fname)

        try:
            with open(raw_path, 'r', encoding='utf-8', errors='ignore') as f:
                raw = f.read()

            cleaned = clean(raw)

            if len(cleaned) < 1000:
                print(f"  SKIP {fname}: too small after cleaning ({len(cleaned)} chars)")
                skipped += 1
                continue

            with open(cleaned_path, 'w', encoding='utf-8') as f:
                f.write(cleaned)

            raw_kb      = len(raw) / 1024
            cleaned_kb  = len(cleaned) / 1024
            removed_pct = (1 - len(cleaned) / len(raw)) * 100

            total_raw     += len(raw)
            total_cleaned += len(cleaned)

            print(f"  OK  {fname:<50} {raw_kb:6.0f}KB -> {cleaned_kb:6.0f}KB  (-{removed_pct:.0f}%)")

        except Exception as e:
            print(f"  ERR {fname}: {e}")

    print(f"\n{'='*60}")
    print(f"Files cleaned  : {len(files) - skipped}")
    print(f"Files skipped  : {skipped}")
    print(f"Raw total      : {total_raw/1e6:.1f}MB")
    print(f"Cleaned total  : {total_cleaned/1e6:.1f}MB")
    print(f"Boilerplate    : {(1 - total_cleaned/total_raw)*100:.1f}% of data removed")
