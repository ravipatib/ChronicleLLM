"""
build_ia_manifest.py — Select Internet Archive scanned books for Chronicle LLM v1.

Queries the Internet Archive scrape API for English texts about Australia dated
1788-1930 and writes data/australia_1850_1950_v1/ia_manifest.csv with an
include/exclude/review decision per item.

Licensing rule (same standard as the Gutenberg and PGA manifests):
  - United States : first published 1930 or earlier (the IA "date" field).
  - Australia     : every named author has a recorded death year before 1955,
                    OR the item is anonymous / a government (Crown) publication,
                    OR it was published 1870 or earlier (an author would have to
                    live past ~103 to still be in copyright).
  - Undated author names are looked up in the Gutenberg catalogue and the PGA
    author list by exact surname + first given name; names that map to more
    than one death year are treated as unknown.
  - Anything else -> "review" (author dates unknown), never auto-included.

Scope rules:
  - Journals, periodicals, article reprints and newspapers are excluded: they
    have many contributors, some of whom lived past 1955, and the IA newspaper
    scans (e.g. the Sydney Morning Herald) have unusable OCR.
  - Statistical/government-table, seed-catalogue, sheet-music and biodiversity
    collections are excluded (little running prose).
  - Repeat scans of the same book keep only the first; titles already taken
    from Gutenberg or PGA are excluded.

Usage:
    python scripts/v1/build_ia_manifest.py [pg_catalog.csv]
"""

import csv
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

REPO_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR   = os.path.join(REPO_DIR, "data", "australia_1850_1950_v1")
OUT_PATH   = os.path.join(DATA_DIR, "ia_manifest.csv")
GUT_MANIF  = os.path.join(DATA_DIR, "gutenberg_manifest.csv")
PGA_MANIF  = os.path.join(DATA_DIR, "pga_manifest.csv")
PGA_AUTH   = os.path.join(DATA_DIR, "pga_authors.csv")
SCRAPE_URL = "https://archive.org/services/search/v1/scrape"
USER_AGENT = "ChronicleLLM-dataset-builder (+https://github.com/ravipatib/ChronicleLLM)"

US_LAST_YEAR    = 1930
AU_DEATH_CUTOFF = 1955
SAFE_PUB_YEAR   = 1870

QUERY = (
    'mediatype:texts AND date:[1788-01-01 TO 1930-12-31] AND (language:eng OR language:English) AND '
    '(subject:(australia OR australian OR australians OR "new south wales" OR queensland OR tasmania OR '
    '"van diemen" OR bushrangers OR "new holland" OR "victoria (australia)" OR "victoria (vic.)" OR '
    '"sydney (n.s.w.)" OR "sydney (australia)" OR "melbourne (vic.)" OR "adelaide (s. aust.)") OR '
    'title:(australia OR australian OR australasia OR bushranger OR bushrangers OR "new south wales" OR '
    'queensland OR tasmania OR "van diemen" OR gippsland OR riverina OR anzac OR "botany bay" OR '
    '"port jackson" OR "swan river"))'
)

EXCLUDED_COLLECTIONS = {
    "newspapers", "The_Sydney_Morning_Herald", "journals", "periodicals", "biostor", "sim_microfilm",
    "pub_medical-journal-of-australia", "jstor_ejc", "paper_doi_mirrored_texts", "taxonomyarchive",
    "government-documents", "australian-statistics", "biodiversity", "usdanationalagriculturallibrary",
    "usda-nurseryandseedcatalog", "slm-sheet-music-and-manuscripts", "slmtradecatalogues", "catalogs",
    "medicalofficerofhealthreports", "fedlink", "statelibraryofqueensland", "csl-lawson-catalogues",
    "swontariopostcards", "sydneylivingmuseums", "magazine_rack", "magazine_rack_misc", "cabi",
}
GOVERNMENT = re.compile(r"^(Australia|New South Wales|Victoria|Queensland|South Australia|Western Australia|"
                        r"Tasmania|Great Britain|Australasian Federa|Federal Council)\b")
ANONYMOUS  = re.compile(r"^(anonymous|unknown|not available|n/a)$", re.I)
NON_TEXT_TITLE = re.compile(r"\b(catalogue|catalog|directory|almanac|gazette|statistic|report of the|"
                            r"returns?|census|register|tables?|index to|list of|proceedings|transactions|"
                            r"journal of the|bulletin no|annual report|log ?book|account book|"
                            r"review|magazine|spectator|chambers's journal)\b|^reports?\b|\(ship\)|\(bark\)|"
                            r"\(brig\)|\(vessel\)", re.I)
FIELDS = ["identifier", "decision", "reason", "title", "creator", "year", "max_death_year",
          "collections", "imagecount"]


def scrape(query, fields):
    """Yield every item for a query, following the scrape API cursor."""
    cursor = None
    while True:
        params = {"q": query, "fields": ",".join(fields), "count": "10000"}
        if cursor:
            params["cursor"] = cursor
        req = urllib.request.Request(SCRAPE_URL + "?" + urllib.parse.urlencode(params),
                                     headers={"User-Agent": USER_AGENT})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    data = json.load(resp)
                break
            except (TimeoutError, OSError):
                if attempt == 3:
                    raise
                time.sleep(10 * (attempt + 1))
        yield from data.get("items", [])
        cursor = data.get("cursor")
        if not cursor:
            return
        time.sleep(1)


def as_list(value):
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def norm(text):
    """First four words of the main title, lowercased — subtitles vary between editions."""
    text = re.split(r"\s*[:;(.]\s*|\s+/\s+", text.lower())[0]
    text = re.sub(r"[^a-z ]", "", text)
    text = re.sub(r"^(the|a|an) ", "", text).strip()
    return " ".join(text.split()[:4])


def name_key(name):
    """(surname, first given name) from 'Surname, Given ...' or 'Given ... Surname'."""
    name = re.sub(r"\(.*?\)|\[.*?\]|\d{4}.*$|,?\s*(Sir|Mrs?|Rev|Dr|Lady|author|editor)\.?\b", "", name, flags=re.I)
    name = name.strip(" ,.")
    if "," in name:
        surname, given = [p.strip() for p in name.split(",", 1)]
    else:
        parts = name.split()
        if len(parts) < 2:
            return None
        surname, given = parts[-1], " ".join(parts[:-1])
    given = given.split()[0].strip(".") if given.split() else ""
    if len(given) < 2 or not surname:
        return None  # initials alone are too ambiguous
    return surname.lower(), given.lower()


def load_death_lookup(catalog_path):
    """Map name_key -> death year from the Gutenberg catalogue and pga_authors.csv."""
    found = {}
    def add(name, death):
        key = name_key(name)
        if key:
            found.setdefault(key, set()).add(death)
    if catalog_path and os.path.exists(catalog_path):
        for row in csv.DictReader(open(catalog_path, encoding="utf-8")):
            for part in row["Authors"].split(";"):
                m = re.search(r"(\d{4})\??-(\d{4})", part)
                if m:
                    add(part, int(m.group(2)))
    for row in csv.DictReader(open(PGA_AUTH, encoding="utf-8")):
        add(row["pga_name"], int(row["death_year"]))
    return {k: next(iter(v)) for k, v in found.items() if len(v) == 1}


def is_taken(title_key, taken_titles):
    return any(title_key.startswith(t) or t.startswith(title_key) for t in taken_titles if t and title_key)


def classify(item, taken_titles, seen_titles, deaths_by_name):
    collections = set(as_list(item.get("collection")))
    title    = " ".join(str(item.get("title", "")).split())
    creators = [c.strip() for c in as_list(item.get("creator")) for c in str(c).split(";") if c.strip()]
    year_m   = re.match(r"(\d{4})", str(item.get("date", "")))
    year     = int(year_m.group(1)) if year_m else None

    if collections & EXCLUDED_COLLECTIONS:
        return "exclude", "collection: periodical/newspaper/tables/catalogue", year, ""
    if year is None or year > US_LAST_YEAR:
        return "exclude", "US: no publication year <=1930", year, ""
    if NON_TEXT_TITLE.search(title):
        return "exclude", "scope: catalogue/report/table-like title", year, ""
    if int(item.get("imagecount") or 0) and int(item["imagecount"]) < 40:
        return "exclude", "scope: under 40 pages (pamphlet/offprint)", year, ""
    key = (norm(title), (creators[0].split(",")[0].lower() if creators else ""))
    if len(norm(title).split()) >= 2 and is_taken(norm(title), taken_titles):
        return "exclude", "duplicate: already from Gutenberg/PGA", year, ""
    if key in seen_titles:
        return "exclude", "duplicate: repeat scan of the same book", year, ""
    seen_titles.add(key)

    deaths, unknown = [], []
    for c in creators:
        if ANONYMOUS.match(c) or GOVERNMENT.match(c):
            continue
        m = re.search(r"\b(\d{4})\??\s*-\s*(\d{4})\b", c) or re.search(r"\bd\.\s*(\d{4})()", c)
        if m:
            deaths.append(int(m.group(2) or m.group(1)))
        elif name_key(c) in deaths_by_name:
            deaths.append(deaths_by_name[name_key(c)])
        else:
            unknown.append(c)
    max_death = max(deaths) if deaths else ""

    if deaths and max_death >= AU_DEATH_CUTOFF:
        return "exclude", "AU copyright: an author died >=1955", year, max_death
    if not unknown:
        return "include", "pub <=1930 (US PD); authors died <1955 or anonymous/Crown (AU PD)", year, max_death
    if year <= SAFE_PUB_YEAR:
        return "include", "pub <=1870: US PD and author cannot have died >=1955", year, max_death
    return "review", "author death year unknown: " + "; ".join(unknown)[:120], year, max_death


def main():
    deaths_by_name = load_death_lookup(sys.argv[1] if len(sys.argv) > 1 else None)
    taken = set()
    for path, title_field, ok in [(GUT_MANIF, "title", "include"), (PGA_MANIF, "title", "include")]:
        for r in csv.DictReader(open(path, encoding="utf-8")):
            if r["decision"] == ok:
                taken.add(norm(r[title_field]))

    fields = ["identifier", "title", "creator", "date", "collection", "imagecount"]
    items  = sorted(scrape(QUERY, fields), key=lambda i: (str(i.get("date", "")), i["identifier"]))
    seen, rows = set(), []
    for item in items:
        decision, reason, year, max_death = classify(item, taken, seen, deaths_by_name)
        rows.append({
            "identifier": item["identifier"], "decision": decision, "reason": reason,
            "title": " ".join(str(item.get("title", "")).split())[:300],
            "creator": "; ".join(str(c) for c in as_list(item.get("creator")))[:200],
            "year": year or "", "max_death_year": max_death,
            "collections": ";".join(as_list(item.get("collection")))[:200],
            "imagecount": item.get("imagecount", ""),
        })

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    counts = {}
    for r in rows:
        counts[r["decision"]] = counts.get(r["decision"], 0) + 1
    print(f"IA items : {len(rows)}  {counts}")
    print(f"Output   : {OUT_PATH}")


if __name__ == "__main__":
    main()
