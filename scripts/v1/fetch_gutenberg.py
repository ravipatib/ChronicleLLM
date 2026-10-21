"""
fetch_gutenberg.py — Download the "include" rows of the v1 Gutenberg manifest.

Downloads plain-text files one at a time over a single reused curl connection
(rate-limited) from an official mirror (Old Dominion University, listed in
https://www.gutenberg.org/MIRRORS.ALL) rather than www.gutenberg.org,
whose robot policy forbids automated access to the main site
(https://www.gutenberg.org/policy/robot_access.html). Already-downloaded files
are skipped, so the script can be re-run safely.

After download every file is verified: the "Title:" line in its Gutenberg header
must match the manifest title. v0 shipped ~60 wrong books because IDs were typed
by hand, so a mismatch is reported and the file is moved to raw_rejected/.

Usage:
    python scripts/v1/build_gutenberg_manifest.py pg_catalog.csv
    python scripts/v1/fetch_gutenberg.py
"""

import csv
import os
import re
import shutil
import subprocess
import tempfile

REPO_DIR     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR     = os.path.join(REPO_DIR, "data", "australia_1850_1950_v1")
MANIFEST     = os.path.join(DATA_DIR, "gutenberg_manifest.csv")
RAW_DIR      = os.path.join(DATA_DIR, "raw", "gutenberg")
REJECTED_DIR = os.path.join(DATA_DIR, "raw_rejected", "gutenberg")
REPORT       = os.path.join(DATA_DIR, "gutenberg_fetch_report.csv")
MIRROR       = "https://mirror.cs.odu.edu/gutenberg-epub/{id}/pg{id}.txt"
USER_AGENT   = "ChronicleLLM-dataset-builder (+https://github.com/ravipatib/ChronicleLLM)"
RATE_LIMIT   = "30/m"   # curl --rate: at most one request every 2 seconds


def norm(text):
    """Lowercase alphanumerics only, for loose title comparison."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def header_title(path):
    """Return the Title: value from a Gutenberg header (may span continuation lines)."""
    with open(path, encoding="utf-8", errors="ignore") as f:
        head = f.read(6000)
    m = re.search(r"^Title:\s*(.+(?:\n {2,}.+)*)", head, re.MULTILINE)
    return " ".join(m.group(1).split()) if m else ""


def main():
    rows = [r for r in csv.DictReader(open(MANIFEST, encoding="utf-8")) if r["decision"] == "include"]
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(REJECTED_DIR, exist_ok=True)

    wanted = [r["pg_id"] for r in rows
              if not os.path.exists(os.path.join(RAW_DIR, f"pg{r['pg_id']}.txt"))]
    print(f"Manifest includes {len(rows)} texts, {len(wanted)} to download\n")

    if wanted:
        # One curl process keeps the HTTPS connection alive across files;
        # --fail stops HTML error pages being saved as books.
        with tempfile.NamedTemporaryFile("w", suffix=".cfg", delete=False) as cfg:
            for pg_id in wanted:
                cfg.write(f'url = "{MIRROR.format(id=pg_id)}"\n')
                cfg.write(f'output = "{os.path.join(RAW_DIR, f"pg{pg_id}.txt")}"\n')
        subprocess.run(["curl", "--fail", "--silent", "--show-error", "--location",
                        "--rate", RATE_LIMIT, "--retry", "2", "--user-agent", USER_AGENT,
                        "--config", cfg.name], check=False)
        os.unlink(cfg.name)

    report = []
    ok = missing = mismatched = 0
    for r in rows:
        path = os.path.join(RAW_DIR, f"pg{r['pg_id']}.txt")
        if not os.path.exists(path):
            status, got = "missing", ""
            missing += 1
        else:
            got = header_title(path)
            # Catalogue titles append subtitles; the header title must be a prefix match.
            a, b = norm(got)[:40], norm(r["title"])[:40]
            if a and (a.startswith(b[:len(a)]) or b.startswith(a)):
                status = "ok"
                ok += 1
            else:
                status = "title_mismatch"
                mismatched += 1
                shutil.move(path, os.path.join(REJECTED_DIR, os.path.basename(path)))
        report.append({"pg_id": r["pg_id"], "status": status, "manifest_title": r["title"],
                       "header_title": got, "bytes": os.path.getsize(path) if status == "ok" else 0})

    with open(REPORT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pg_id", "status", "manifest_title", "header_title", "bytes"])
        writer.writeheader()
        writer.writerows(report)

    total_mb = sum(r["bytes"] for r in report) / 1e6
    print(f"\nVerified ok    : {ok}  ({total_mb:.1f}MB)")
    print(f"Missing        : {missing}")
    print(f"Title mismatch : {mismatched}  (moved to {REJECTED_DIR})")
    print(f"Report         : {REPORT}")


if __name__ == "__main__":
    main()
