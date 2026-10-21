"""
fetch_hansard.py — Sample Commonwealth Hansard 1901-1945 and convert it to text.

Source: Tim Sherratt's harvest of the Parliament's Hansard XML
(https://github.com/wragge/hansard-xml, via the GLAM Workbench), House of
Representatives and Senate.

Licensing position (see data/australia_1850_1950_v1/README.md):
  - Australia : Commonwealth Crown copyright lasts 50 years (Copyright Act s 180),
                so debates up to 1945 were out of copyright by 1996 at the latest.
  - US        : pre-1931 debates are PD; 1931-1945 were already PD in Australia on
                1 Jan 1996, so the URAA did not restore them.
  - Parliament publishes the XML under CC BY-NC-ND. That licence cannot extend to
    expired material, but we attribute the source and only use 1901-1945.

Sampling: about TARGET_MB of text spread evenly over the years, with each year's
sitting days chosen at even intervals (not just the first N), both houses in
proportion to their size. Output: one text file per sitting day.

Usage:
    python scripts/v1/fetch_hansard.py
"""

import json
import os
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

REPO_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR   = os.path.join(REPO_DIR, "data", "australia_1850_1950_v1")
RAW_DIR    = os.path.join(DATA_DIR, "raw", "hansard")
MANIFEST   = os.path.join(DATA_DIR, "hansard_manifest.csv")
TREE_URL   = "https://api.github.com/repos/wragge/hansard-xml/git/trees/master?recursive=1"
RAW_URL    = "https://raw.githubusercontent.com/wragge/hansard-xml/master/{path}"
USER_AGENT = "ChronicleLLM-dataset-builder (+https://github.com/ravipatib/ChronicleLLM)"

FIRST_YEAR, LAST_YEAR = 1901, 1945
TARGET_MB  = 150
TEXT_RATIO = 0.43   # measured over the full sample: text is ~43% of the XML size
WORKERS    = 4


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.read()
        except (TimeoutError, OSError):
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))


def pick_files():
    """Choose sitting-day files evenly across years and houses up to the size budget."""
    tree = json.loads(get(TREE_URL))["tree"]
    by_year = defaultdict(list)
    for node in tree:
        m = re.match(r"(hofreps|senate)/(\d{4})/.+\.xml$", node["path"])
        if node["type"] == "blob" and m and FIRST_YEAR <= int(m.group(2)) <= LAST_YEAR:
            by_year[int(m.group(2))].append((node["path"], node["size"]))

    budget_per_year = TARGET_MB * 1e6 / TEXT_RATIO / len(by_year)
    chosen = []
    for year in sorted(by_year):
        files = sorted(by_year[year])
        total = sum(size for _, size in files)
        n_keep = max(1, min(len(files), round(len(files) * budget_per_year / total)))
        step   = len(files) / n_keep
        chosen.extend(files[int(i * step)] for i in range(n_keep))
    return chosen


def text_of(elem):
    return " ".join("".join(elem.itertext()).split())


def hansard_to_text(xml_bytes):
    """Flatten a sitting day into 'Debate title' headings and 'Speaker. — words' paragraphs."""
    root = ET.fromstring(xml_bytes)
    out = []
    for debate in root.iter("debate"):
        title = debate.find("debateinfo/title")
        if title is not None and text_of(title):
            out.append(text_of(title).upper())
        for block in debate.iter():
            if block.tag not in ("speech", "question", "answer", "interjection", "continue"):
                continue
            talker = block.find("talk.start/talker/name[@role='display']")
            speaker = text_of(talker) if talker is not None else ""
            paras = [text_of(p) for p in block.iter("para")]
            paras = [p for p in paras if p]
            if not paras:
                continue
            # Paragraphs are split at page breaks; rejoin those that end mid-sentence.
            merged = [paras[0]]
            for p in paras[1:]:
                ends_sentence = re.search(r"[.!?:;\"')]$", merged[-1]) and \
                    not re.search(r"\b(Mr|Mrs|Dr|Sir|St|No|Hon|Messrs)\.$", merged[-1])
                if not ends_sentence:
                    merged[-1] += " " + p
                else:
                    merged.append(p)
            if speaker:
                merged[0] = f"{speaker}. — {merged[0].lstrip('- ').strip()}"
            out.extend(merged)
            # Nested blocks are visited again by iter(); stop children duplicating.
            for child in list(block):
                if child.tag in ("interjection", "continue"):
                    block.remove(child)
    return "\n\n".join(out)


def fetch(item):
    path, _ = item
    dest = os.path.join(RAW_DIR, os.path.basename(path).replace(".xml", ".txt"))
    if os.path.exists(dest):
        return path, "ok", os.path.getsize(dest)
    try:
        text = hansard_to_text(get(RAW_URL.format(path=path)))
    except Exception as e:
        return path, f"failed: {e}", 0
    with open(dest, "w", encoding="utf-8") as f:
        f.write(text)
    return path, "ok", len(text)


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    chosen = pick_files()
    print(f"Sampling {len(chosen)} sitting days, {FIRST_YEAR}-{LAST_YEAR}\n")
    # Drop days from an earlier run that are no longer in the sample.
    wanted = {os.path.basename(p).replace(".xml", ".txt") for p, _ in chosen}
    for name in os.listdir(RAW_DIR):
        if name.endswith(".txt") and name not in wanted:
            os.remove(os.path.join(RAW_DIR, name))
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(fetch, chosen))

    with open(MANIFEST, "w", encoding="utf-8") as f:
        f.write("path,status,text_bytes\n")
        for path, status, size in results:
            f.write(f"{path},{status},{size}\n")
    ok = [r for r in results if r[1] == "ok"]
    print(f"Converted ok : {len(ok)}  ({sum(r[2] for r in ok)/1e6:.1f}MB text)")
    print(f"Failed       : {len(results) - len(ok)}")
    print(f"Manifest     : {MANIFEST}")


if __name__ == "__main__":
    main()
