"""
fetch_pga.py — Select and download Project Gutenberg Australia texts for v1.

PGA hosts works that are public domain in AUSTRALIA only (its licence says
"Persons outside of Australia should check their laws"), including many authors
still in copyright in the US. So nothing is taken in bulk: only authors listed
in data/australia_1850_1950_v1/pga_authors.csv are considered.

Licensing rule:
  - Author died 1945 or earlier -> include. Their works were already out of
    copyright in Australia on 1 January 1996, so the US URAA did not restore
    them, and they are public domain in both countries.
  - Author died 1946-1954 -> "review": public domain in Australia, but in the US
    only if first published 1930 or earlier, which must be checked per title.
  - Authors whose note starts "scope:" (prolific, mostly non-Australian
    settings) -> "review", so they cannot swamp the corpus.
  - Titles already included from gutenberg.org, or repeated within PGA, are skipped.

Inputs : PGA index https://gutenberg.net.au/gutindex_aus.txt
Outputs: data/australia_1850_1950_v1/pga_manifest.csv
         data/australia_1850_1950_v1/raw/pga/<ebook_no>.txt

Usage:
    curl -o gutindex_aus.txt https://gutenberg.net.au/gutindex_aus.txt
    python scripts/v1/fetch_pga.py gutindex_aus.txt [--download]
"""

import csv
import html
import os
import re
import sys
import time
import urllib.request

REPO_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR   = os.path.join(REPO_DIR, "data", "australia_1850_1950_v1")
AUTHORS    = os.path.join(DATA_DIR, "pga_authors.csv")
GUT_MANIF  = os.path.join(DATA_DIR, "gutenberg_manifest.csv")
OUT_PATH   = os.path.join(DATA_DIR, "pga_manifest.csv")
RAW_DIR    = os.path.join(DATA_DIR, "raw", "pga")
USER_AGENT = "ChronicleLLM-dataset-builder (+https://github.com/ravipatib/ChronicleLLM)"
DELAY_SEC  = 1.0
SAFE_DEATH = 1945

FIELDS = ["ebook_no", "decision", "reason", "title", "author", "death_year", "url", "status"]


def norm(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())


def parse_index(path):
    """Yield (title, author, url) from the PGA plain-text index."""
    lines = open(path, encoding="utf-8", errors="ignore").read().split("\n")
    for i, line in enumerate(lines[:-1]):
        m = re.match(r"^\w{3} \d{4} (.+?)\s+\[\w+\.?\w*\]", line)
        url = lines[i + 1].strip()
        if not m or "gutenberg.net.au/ebooks" not in url:
            continue
        if not url.endswith((".html", ".txt")):
            continue  # PDF-only entries
        entry = m.group(1).strip()
        title, _, author = entry.rpartition(", ") if ", " in entry else (entry, "", "")
        author = re.sub(r"^by\s+", "", author.strip())
        yield title.strip(), " ".join(author.split()), url.replace("http://", "https://")


def html_to_text(raw):
    """Strip a PGA HTML ebook down to plain text, keeping paragraph breaks."""
    raw = re.sub(r"(?is)<(script|style|head).*?</\1>", "", raw)
    raw = re.sub(r"(?i)<br\s*/?>|</p>|</h\d>|</div>", "\n", raw)
    text = html.unescape(re.sub(r"<[^>]+>", "", raw))
    text = re.sub(r"[ \t\xa0]+", " ", text)
    return re.sub(r"\n\s*\n\s*", "\n\n", text).strip()


def main():
    index_path = sys.argv[1]
    download   = "--download" in sys.argv

    authors = list(csv.DictReader(open(AUTHORS, encoding="utf-8")))
    deaths  = {r["pga_name"]: int(r["death_year"]) for r in authors}
    scoped  = {r["pga_name"] for r in authors if r["note"].startswith("scope:")}
    gut_titles = {norm(r["title"])[:30] for r in csv.DictReader(open(GUT_MANIF, encoding="utf-8"))
                  if r["decision"] == "include"}

    rows, seen, seen_titles = [], set(), set()
    for title, author, url in parse_index(index_path):
        if author not in deaths:
            continue
        ebook_no = re.search(r"/(\d+)h?\.(html|txt)$", url).group(1)
        key = norm(title)[:30]
        if ebook_no in seen:
            continue
        seen.add(ebook_no)
        death = deaths[author]
        if key in gut_titles:
            decision, reason = "exclude", "already included from gutenberg.org"
        elif (key, author) in seen_titles:
            decision, reason = "exclude", "duplicate title within PGA"
        elif author in scoped:
            decision, reason = "review", "prolific author, mostly non-Australian settings: pick by hand"
        elif death <= SAFE_DEATH:
            decision, reason = "include", "author died <=1945: PD in AU; not URAA-restored, so PD in US"
        else:
            decision, reason = "review", "author died 1946-1954: confirm first publication <=1930"
        seen_titles.add((key, author))
        rows.append({"ebook_no": ebook_no, "decision": decision, "reason": reason, "title": title,
                     "author": author, "death_year": death, "url": url, "status": ""})

    if download:
        os.makedirs(RAW_DIR, exist_ok=True)
        todo = [r for r in rows if r["decision"] == "include"]
        for i, r in enumerate(todo, 1):
            dest = os.path.join(RAW_DIR, f"{r['ebook_no']}.txt")
            if os.path.exists(dest):
                r["status"] = "ok"
                continue
            try:
                req = urllib.request.Request(r["url"], headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    raw = resp.read().decode("utf-8", errors="ignore")
                text = html_to_text(raw) if r["url"].endswith(".html") else raw
                # Verify the downloaded header names the expected title (v0 lesson).
                head = norm(text[:3000])
                if norm(r["title"])[:20] not in head:
                    r["status"] = "title_mismatch"
                else:
                    with open(dest, "w", encoding="utf-8") as f:
                        f.write(text)
                    r["status"] = "ok"
            except Exception as e:
                r["status"] = f"failed: {e}"
            print(f"  [{i}/{len(todo)}] {r['ebook_no']} {r['status']:<14} {r['title'][:50]}", flush=True)
            time.sleep(DELAY_SEC)

    with open(OUT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    counts = {}
    for r in rows:
        counts[r["decision"]] = counts.get(r["decision"], 0) + 1
    print(f"\nPGA candidates : {len(rows)}  {counts}")
    if download:
        ok = sum(1 for r in rows if r["status"] == "ok")
        print(f"Downloaded ok  : {ok}")
    print(f"Manifest       : {OUT_PATH}")


if __name__ == "__main__":
    main()
