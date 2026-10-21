"""
build_ia_manifest.py — Select Internet Archive scanned books for Chronicle LLM v1.

Queries the Internet Archive scrape API for English texts dated 1788-1930 that
are about Australia (subject or title) or were published in Australia, and
writes data/australia_1850_1950_v1/ia_manifest.csv with an include/exclude/review
decision per item and how it was found (found_by: topic or publisher).

Licensing rule (same standard as the Gutenberg and PGA manifests):
  - United States : first published 1930 or earlier (the IA "date" field).
  - Australia     : every named author has a recorded death year before 1955,
                    OR the item is anonymous / a government (Crown) publication,
                    OR it was published 1870 or earlier (an author would have to
                    live past ~103 to still be in copyright).
  - Undated author names are looked up in the Gutenberg catalogue and the PGA
    author list by exact surname + first given name; names that map to more
    than one death year are treated as unknown.
  - Then in author_deaths.csv (written by resolve_ia_authors.py from
    Wikidata, checked against Library of Congress): only rows with
    status=resolved count. Authors born 1850 or earlier count as safe.
  - Societies, museums, surveys and other bodies with no named author are
    treated like anonymous works (published before 1955 -> AU public domain).
  - Anything else -> "review" (author dates unknown), never auto-included.

Scope rules:
  - Items found only through the publisher field must name an Australian city
    (or Angus & Robertson / George Robertson) as the FIRST place of
    publication: "London, Melbourne" imprints, Sydney (Nova Scotia),
    Hampden-Sydney College and Melbourne (Florida) are excluded, as are
    journals, proceedings and medical-library scans found that way.
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

from resolve_ia_authors import ORGANISATION, SAFE_BIRTH_YEAR

REPO_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR   = os.path.join(REPO_DIR, "data", "australia_1850_1950_v1")
OUT_PATH   = os.path.join(DATA_DIR, "ia_manifest.csv")
GUT_MANIF  = os.path.join(DATA_DIR, "gutenberg_manifest.csv")
PGA_MANIF  = os.path.join(DATA_DIR, "pga_manifest.csv")
PGA_AUTH   = os.path.join(DATA_DIR, "pga_authors.csv")
RESOLVED   = os.path.join(DATA_DIR, "author_deaths.csv")
SCRAPE_URL = "https://archive.org/services/search/v1/scrape"
USER_AGENT = "ChronicleLLM-dataset-builder (+https://github.com/ravipatib/ChronicleLLM)"

US_LAST_YEAR    = 1930
AU_DEATH_CUTOFF = 1955
SAFE_PUB_YEAR   = 1870

BASE_QUERY = 'mediatype:texts AND date:[1788-01-01 TO 1930-12-31] AND (language:eng OR language:English)'
TOPIC_QUERY = (
    '(subject:(australia OR australian OR australians OR "new south wales" OR queensland OR tasmania OR '
    '"van diemen" OR bushrangers OR "new holland" OR "victoria (australia)" OR "victoria (vic.)" OR '
    '"sydney (n.s.w.)" OR "sydney (australia)" OR "melbourne (vic.)" OR "adelaide (s. aust.)") OR '
    'title:(australia OR australian OR australasia OR bushranger OR bushrangers OR "new south wales" OR '
    'queensland OR tasmania OR "van diemen" OR gippsland OR riverina OR anzac OR "botany bay" OR '
    '"port jackson" OR "swan river"))'
)
# Books printed in Australia that carry no Australian subject or title term
PUBLISHER_QUERY = (
    'publisher:(melbourne OR sydney OR adelaide OR brisbane OR hobart OR ballarat OR bendigo OR launceston OR '
    'geelong OR "angus & robertson" OR "angus and robertson" OR "george robertson") AND NOT ' + TOPIC_QUERY
)
PLACE = re.compile(r"(?<![-\w.])(Melbourne|Sydney|Sidney|Adelaide|Brisbane|Hobart|Ballarat|Bendigo|Launceston|"
                   r"Geelong|Angus (?:&|and) Robertson|George Robertson|London|New York|Toronto|Edinburgh|Glasgow|"
                   r"Paris|Boston|Chicago|Philadelphia|Calcutta|Bombay|Wellington|Auckland|Dunedin|Christchurch|"
                   r"Cape Town|Los Angeles)(?!\s*&)", re.I)
AU_PLACE = re.compile(r"Melbourne|Sydney|Sidney|Adelaide|Brisbane|Hobart|Ballarat|Bendigo|Launceston|Geelong|"
                      r"Angus|George Robertson", re.I)
# Overseas places that share a name with an Australian city, and British houses with Australian branches
NOT_AU_IMPRINT = re.compile(r"Hampden|\bFla\b|Florida|Nova Scotia|Cape Breton|C\[?ape\]? ?B|\bN\.\s?S\.(?!\s?W)|"
                            r"Mich\b|Ohio|Heinemann|Cassell|Macmillan|Longmans|Ward,? Lock|Collins|Hodder|"
                            r"Oxford University|Blackie|Nelson|Methuen|Hutchinson|Theosophical", re.I)
PERIODICAL_TITLE = re.compile(r"\b(journal|proceedings|transactions|naturalist|gazette|bulletin|year ?book|"
                              r"calendar|vol\.?\s*[ivxl\d]|no\.\s*\d|statutes|acts? of)\b", re.I)
MEDICAL_COLLECTIONS = {"medicallibrary", "medicalheritagelibrary", "ukmhl", "gerstein"}

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
                            r"mineral statistics|statistical register|vital statistics|^statistics\b|"
                            r"returns?|census|register|tables?|index to|list of|proceedings|transactions|"
                            r"journal of the|bulletin no|annual report|log ?book|account book|"
                            r"review|magazine|spectator|chambers's journal|bulletin,? (vol|no)|reading lists)\b|^reports?\b|\(ship\)|\(bark\)|"
                            r"\(brig\)|\(vessel\)", re.I)
FIELDS = ["identifier", "decision", "reason", "found_by", "title", "creator", "year", "max_death_year",
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


def load_resolved():
    """creator string -> death year (or None for 'born <=1850') from author_deaths.csv."""
    if not os.path.exists(RESOLVED):
        return {}
    out = {}
    for r in csv.DictReader(open(RESOLVED, encoding="utf-8")):
        if r["status"] == "resolved":
            out[r["creator"]] = int(r["death_year"]) if r["death_year"] else None
        elif r["status"] == "died_1955_or_later":
            out[r["creator"]] = int(r["death_year"])
    return out


def creator_deaths(c, deaths_by_name, resolved):
    """Death years for one creator entry, or None if any named person is unknown."""
    m = re.search(r"\b(\d{4})\??\s*-\s*(\d{4})\b", c) or re.search(r"\bd\.\s*(\d{4})()", c)
    if m:
        return [int(m.group(2) or m.group(1))]
    b = re.search(r"\b(\d{4})\??\s*-(?!\s*\d)|\bb\.\s*(\d{4})", c)
    if b and int(b.group(1) or b.group(2)) <= SAFE_BIRTH_YEAR:
        return []
    if name_key(c) in deaths_by_name:
        return [deaths_by_name[name_key(c)]]
    parts = [c] if re.search(r"\d{4}", c) else re.split(r"\s+(?:&|and)\s+", c)
    deaths = []
    for part in parts:
        if part not in resolved:
            return None
        if resolved[part] is not None:
            deaths.append(resolved[part])
    return deaths


def classify(item, taken_titles, seen_titles, deaths_by_name, resolved):
    collections = set(as_list(item.get("collection")))
    title    = " ".join(str(item.get("title", "")).split())
    creators = [c.strip() for c in as_list(item.get("creator")) for c in str(c).split(";") if c.strip()]
    year_m   = re.match(r"(\d{4})", str(item.get("date", "")))
    year     = int(year_m.group(1)) if year_m else None
    publisher = " ".join(str(x) for x in as_list(item.get("publisher")))

    if item.get("_via") == "publisher":
        # Found only through the publisher field: must be an Australian imprint.
        first = PLACE.search(publisher)
        if not first or not AU_PLACE.match(first.group(1)) or NOT_AU_IMPRINT.search(publisher) \
                or PLACE.fullmatch(publisher.strip(" []")):   # a bare city name ("Hobart") is not an imprint
            return "exclude", "publisher route: first place of publication not Australian", year, ""
        if PERIODICAL_TITLE.search(title) or collections & MEDICAL_COLLECTIONS:
            return "exclude", "publisher route: periodical, statutes or medical-library scan", year, ""
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
        if ANONYMOUS.match(c) or GOVERNMENT.match(c) or ORGANISATION.search(c):
            continue
        found = creator_deaths(c, deaths_by_name, resolved)
        if found is None:
            unknown.append(c)
        else:
            deaths.extend(found)
    max_death = max(deaths) if deaths else ""

    if deaths and max_death >= AU_DEATH_CUTOFF:
        return "exclude", "AU copyright: an author died >=1955", year, max_death
    if not unknown:
        return "include", "pub <=1930 (US PD); authors died <1955, born <=1850, or anonymous/Crown/corporate (AU PD)", year, max_death
    if year <= SAFE_PUB_YEAR:
        return "include", "pub <=1870: US PD and author cannot have died >=1955", year, max_death
    return "review", "author death year unknown: " + "; ".join(unknown)[:120], year, max_death


def main():
    deaths_by_name = load_death_lookup(sys.argv[1] if len(sys.argv) > 1 else None)
    resolved = load_resolved()
    taken = set()
    for path, title_field, ok in [(GUT_MANIF, "title", "include"), (PGA_MANIF, "title", "include")]:
        for r in csv.DictReader(open(path, encoding="utf-8")):
            if r["decision"] == ok:
                taken.add(norm(r[title_field]))

    fields = ["identifier", "title", "creator", "date", "collection", "imagecount", "publisher"]
    items = []
    for via, query in [("topic", TOPIC_QUERY), ("publisher", PUBLISHER_QUERY)]:
        for item in scrape(BASE_QUERY + " AND " + query, fields):
            item["_via"] = via
            items.append(item)
    items.sort(key=lambda i: (str(i.get("date", "")), i["identifier"]))
    seen, rows = set(), []
    for item in items:
        decision, reason, year, max_death = classify(item, taken, seen, deaths_by_name, resolved)
        rows.append({
            "identifier": item["identifier"], "decision": decision, "reason": reason, "found_by": item["_via"],
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
