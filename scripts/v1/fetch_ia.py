"""
fetch_ia.py — Download OCR text for the "include" rows of the v1 IA manifest.

For each identifier, fetches <identifier>_djvu.txt from archive.org/download.
If that name doesn't exist, the item's file list is read from the metadata API
and the first *_djvu.txt is used. 4 concurrent requests, resumable.

Usage:
    python scripts/v1/build_ia_manifest.py pg_catalog.csv
    python scripts/v1/fetch_ia.py
"""

import csv
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR   = os.path.join(REPO_DIR, "data", "australia_1850_1950_v1")
MANIFEST   = os.path.join(DATA_DIR, "ia_manifest.csv")
RAW_DIR    = os.path.join(DATA_DIR, "raw", "ia")
REPORT     = os.path.join(DATA_DIR, "ia_fetch_report.csv")
USER_AGENT = "ChronicleLLM-dataset-builder (+https://github.com/ravipatib/ChronicleLLM)"
WORKERS    = 4


def get(url, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                raise
            time.sleep(5 * (attempt + 1))
        except (TimeoutError, OSError):
            time.sleep(5 * (attempt + 1))
    raise OSError(f"failed after retries: {url}")


def fetch(identifier):
    dest = os.path.join(RAW_DIR, f"{identifier}.txt")
    if os.path.exists(dest):
        return identifier, "ok", os.path.getsize(dest)
    try:
        try:
            data = get(f"https://archive.org/download/{identifier}/{identifier}_djvu.txt")
        except urllib.error.HTTPError:
            meta  = json.loads(get(f"https://archive.org/metadata/{identifier}/files"))
            names = [f["name"] for f in meta.get("result", []) if f["name"].endswith("_djvu.txt")]
            if not names:
                return identifier, "no_ocr_text", 0
            data = get(f"https://archive.org/download/{identifier}/{urllib.request.quote(names[0])}")
    except Exception as e:
        return identifier, f"failed: {e}", 0
    with open(dest + ".part", "wb") as f:
        f.write(data)
    os.replace(dest + ".part", dest)
    return identifier, "ok", len(data)


def main():
    rows = [r for r in csv.DictReader(open(MANIFEST, encoding="utf-8")) if r["decision"] == "include"]
    os.makedirs(RAW_DIR, exist_ok=True)
    print(f"Manifest includes {len(rows)} items\n")

    results = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(fetch, r["identifier"]) for r in rows]
        for i, fut in enumerate(as_completed(futures), 1):
            identifier, status, size = fut.result()
            results.append({"identifier": identifier, "status": status, "bytes": size})
            if status != "ok" or i % 50 == 0:
                print(f"  [{i}/{len(rows)}] {identifier} {status}", flush=True)

    with open(REPORT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["identifier", "status", "bytes"])
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda r: r["identifier"]))

    ok = [r for r in results if r["status"] == "ok"]
    print(f"\nDownloaded ok : {len(ok)}  ({sum(r['bytes'] for r in ok)/1e6:.1f}MB)")
    print(f"Not ok        : {len(results) - len(ok)}")
    print(f"Report        : {REPORT}")


if __name__ == "__main__":
    main()
