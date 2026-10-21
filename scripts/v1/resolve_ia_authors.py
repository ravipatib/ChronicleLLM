"""
resolve_ia_authors.py — Find death years for authors in the IA and Gutenberg review queues.

Reads   data/australia_1850_1950_v1/ia_manifest.csv         (rows with decision=review)
        data/australia_1850_1950_v1/gutenberg_manifest.csv  (rows with decision=review)
Writes  data/australia_1850_1950_v1/author_deaths.csv
        creator, name, birth_year, death_year, status, evidence

build_ia_manifest.py and build_gutenberg_manifest.py read author_deaths.csv and
treat a name as known only when status=resolved (or died_1955_or_later, which
excludes it). Re-run both builders afterwards.

How a name is resolved (conservative: anything unclear stays "unresolved"):
  - Dates already in the IA creator string ("1836-1907", "d. 1900") are used as is;
    born 1850 or earlier also counts as safe (would have to live past ~104).
  - Otherwise every Wikidata human whose English label or alias is exactly the
    name is collected. A candidate is "plausible" if it could have written a book
    published that year: born within 100 years before it and at least 15 before
    (candidates with no birth date are plausible unless they died before the
    book could have been written). A birth year in the IA string must match, and
    when some candidates match it exactly only those count.
  - "Surname, Husband's name, Mrs" entries are never looked up: the name is the
    husband's, so a match would be the wrong person.
  - Religious names ("Agnes, Sister") are never looked up either.
  - The name is resolved only if there is at least one plausible candidate and
    every plausible candidate died before 1955. One plausible candidate with no
    death date, or one who died 1955 or later, leaves it unresolved.
  - Library of Congress name authorities are a cross-check: an LC heading with
    a birth year for the same name must agree with the Wikidata candidate.

Usage:
    python scripts/v1/resolve_ia_authors.py
"""

import csv
import http.client
import json
import os
import re
import time
import urllib.parse
import urllib.request
from collections import defaultdict

REPO_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR   = os.path.join(REPO_DIR, "data", "australia_1850_1950_v1")
MANIFEST   = os.path.join(DATA_DIR, "ia_manifest.csv")
GUT_MANIF  = os.path.join(DATA_DIR, "gutenberg_manifest.csv")
OUT_PATH   = os.path.join(DATA_DIR, "author_deaths.csv")
WDQS_URL   = "https://query.wikidata.org/sparql"
LOC_URL    = "https://id.loc.gov/authorities/names/suggest2"
USER_AGENT = "ChronicleLLM-dataset-builder/1.0 (+https://github.com/ravipatib/ChronicleLLM)"

AU_DEATH_CUTOFF = 1955
SAFE_BIRTH_YEAR = 1850
BATCH           = 40
FIELDS = ["creator", "name", "birth_year", "death_year", "status", "evidence"]

ORGANISATION = re.compile(r"\b(survey|museum|observatory|department|society|college|government|office|board|"
                          r"commission|university|council|library|company|club|association|institute|"
                          r"institution|church|mission|parliament|bureau|trustees|committee|bank|league|"
                          r"union|exhibition|gardens|school|agency|ltd|limited|& co|and co)\b", re.I)
ROLE = re.compile(r"\b(editor|ed|tr|trans|translator|compiler|comp|illustrator|ill|joint author|author|"
                  r"pseud|sir|mrs?|miss|rev|dr|lady|hon|capt|captain|col|major|lieut|prof|bart|k\.?c\.?m\.?g)\b\.?",
                  re.I)


def get(url, data=None, accept="application/json"):
    req = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT, "Accept": accept})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.load(resp)
        except (TimeoutError, OSError, http.client.HTTPException) as e:
            if attempt == 3:
                raise
            wait = 60 if "429" in str(e) else 10 * (attempt + 1)
            time.sleep(wait)


def parse_person(creator):
    """Return (display name 'Given Surname', birth, death) from an IA creator string, or None."""
    text  = creator.strip()
    birth = death = None
    if re.search(r"\b(Mrs|Lady)\b", text):
        return None   # "Millett, Edward, Mrs": the name is the husband's, not the author's
    if re.search(r"\b(Sister|Brother|Mother|Father)\b", text):
        return None   # religious names ("Agnes, Sister") are shared by many people
    if text == text.lower():
        text = text.title()
    if m := re.search(r"\b(\d{4})\??\s*-\s*(\d{4})?", text):
        birth = int(m.group(1))
        death = int(m.group(2)) if m.group(2) else None
    elif m := re.search(r"\bb\.\s*(\d{4})", text):
        birth = int(m.group(1))
    elif m := re.search(r"\bd\.\s*(\d{4})", text):
        death = int(m.group(1))
    fuller = re.search(r"\(([A-Z][a-z]+(?: [A-Z][a-z]+)*)\)", text)   # "Hunt, H. A. (Henry Ambrose)"
    text = re.sub(r"\(.*?\)|\[.*?\]|\b(b|d|fl|ca?)\.\s*\d{4}|\d{4}\??\s*-\s*(\d{4})?", "", text)
    text = ROLE.sub("", text)
    text = " ".join(text.replace(" ,", ",").split()).strip(" ,.;")
    if "," in text:
        surname, given = [p.strip(" ,.") for p in text.split(",", 1)]
        given = given.split(",")[0].strip(" ,.")
    else:
        parts = text.split()
        if len(parts) < 2:
            return None
        # "Boyd A. J." (surname first, initials after)
        if all(re.fullmatch(r"[A-Z]\.?", p) for p in parts[1:]):
            surname, given = parts[0], " ".join(parts[1:])
        else:
            surname, given = parts[-1], " ".join(parts[:-1])
    if fuller:
        given = fuller.group(1)
    if not surname or not given or not re.search(r"[A-Za-z]", given):
        return None
    given = re.sub(r"\b([A-Z])\.?(?=\s|$)", r"\1.", given)            # "A J" -> "A. J."
    return f"{given} {surname}".strip(), birth, death


def label_variants(name):
    """Exact-match labels to try: 'A. J. Boyd', 'A.J. Boyd'."""
    out = {name}
    out.add(re.sub(r"\.\s+(?=[A-Z]\.)", ".", name))
    return out


def wikidata_candidates(names):
    """label -> list of (qid, birth, death, description) for humans with that exact label or alias."""
    values = " ".join(json.dumps(v) + "@en" for n in names for v in label_variants(n))
    query = f"""
    SELECT ?label ?p ?b ?d ?desc WHERE {{
      VALUES ?label {{ {values} }}
      {{ ?p rdfs:label ?label }} UNION {{ ?p skos:altLabel ?label }}
      ?p wdt:P31 wd:Q5 .
      OPTIONAL {{ ?p wdt:P569 ?b }}
      OPTIONAL {{ ?p wdt:P570 ?d }}
      OPTIONAL {{ ?p schema:description ?desc FILTER(LANG(?desc) = "en") }}
    }}"""
    data = get(WDQS_URL, urllib.parse.urlencode({"query": query, "format": "json"}).encode(),
               accept="application/sparql-results+json")
    found = defaultdict(dict)
    for row in data["results"]["bindings"]:
        label = row["label"]["value"]
        qid   = row["p"]["value"].rsplit("/", 1)[-1]
        # "unknown value" dates come back as a URL, so only accept a leading year
        year  = lambda k: int(m.group(1)) if k in row and (m := re.match(r"\+?(\d{4})-", row[k]["value"])) else None
        cand  = found[label].setdefault(qid, {"birth": None, "death": None, "desc": ""})
        # Several dates can be recorded; keep the latest death and earliest birth (the cautious choice).
        if (b := year("b")) and (cand["birth"] is None or b < cand["birth"]):
            cand["birth"] = b
        if (d := year("d")) and (cand["death"] is None or d > cand["death"]):
            cand["death"] = d
        cand["desc"] = row.get("desc", {}).get("value", cand["desc"])
    return found


def loc_birth_years(name):
    """Birth years on LC name headings that match the name exactly (surname, given)."""
    given, surname = name.rsplit(" ", 1)
    q = f"{surname}, {given}"
    try:
        data = get(LOC_URL + "?" + urllib.parse.urlencode({"q": q, "searchtype": "leftanchored", "count": 20}))
    except OSError:
        return set()
    years = set()
    for hit in data.get("hits", []):
        head = re.sub(r",?\s*\d{4}.*$", "", hit["aLabel"]).strip(" ,.")
        if head.lower() == q.lower():
            years.update(int(y[:4]) for y in hit["more"].get("birthdates", []) if y[:4].isdigit())
    return years


def judge(name, birth, pub_year, candidates, loc_births):
    """Return (status, death_year, evidence)."""
    plausible = []
    for qid, c in candidates.items():
        b, d = c["birth"], c["death"]
        if birth and b and abs(b - birth) > 1:
            continue
        if pub_year is None:          # Gutenberg: no publication year in the catalogue
            if b and not 1780 <= b <= 1915:
                continue
        elif b and not (pub_year - 100 <= b <= pub_year - 15):
            continue
        elif not b and d and d < pub_year - 80:
            continue
        plausible.append((qid, c))
    # A birth year in the IA record that a candidate matches identifies the person;
    # undated namesakes no longer count.
    if birth and any(c["birth"] for _, c in plausible):
        plausible = [(q, c) for q, c in plausible if c["birth"]]
    if not plausible:
        return "unresolved", "", "no plausible Wikidata match"
    if loc_births and not any(c["birth"] and any(abs(c["birth"] - y) <= 1 for y in loc_births)
                              for _, c in plausible):
        return "unresolved", "", f"LC birth {sorted(loc_births)} disagrees with Wikidata"
    evidence = "; ".join(f"{q} {c['birth'] or '?'}-{c['death'] or '?'} {c['desc'][:60]}" for q, c in plausible)
    deaths = [c["death"] for _, c in plausible]
    if any(d is None for d in deaths):
        return "unresolved", "", "a plausible match has no death date: " + evidence
    if max(deaths) >= AU_DEATH_CUTOFF:
        return "died_1955_or_later", max(deaths), evidence
    return "resolved", max(deaths), evidence


def main():
    rows = [r for r in csv.DictReader(open(MANIFEST, encoding="utf-8")) if r["decision"] == "review"]
    people = {}   # creator string -> (name, birth, death, earliest pub year)
    for r in rows:
        year = int(r["year"]) if r["year"] else 1930
        for c in r["creator"].split(";"):
            c = c.strip()
            if not c or ORGANISATION.search(c):
                continue
            for part in re.split(r"\s+(?:&|and)\s+", c) if not re.search(r"\d{4}", c) else [c]:
                p = parse_person(part)
                if p:
                    prev = people.get(part)
                    people[part] = (p[0], p[1], p[2], min(year, prev[3]) if prev else year)
    for r in csv.DictReader(open(GUT_MANIF, encoding="utf-8")):
        if r["decision"] != "review" or not r["reason"].startswith("death year unknown: "):
            continue
        for c in r["reason"][len("death year unknown: "):].split("; "):
            c = c.strip()
            if c and c != "Various" and c not in people and not ORGANISATION.search(c):
                p = parse_person(c)
                if p:
                    people[c] = (p[0], p[1], p[2], None)

    out = {}
    if os.path.exists(OUT_PATH):   # earlier runs: keep their answers, look up only new creators
        for r in csv.DictReader(open(OUT_PATH, encoding="utf-8")):
            if r["evidence"] == "dates in IA record" or parse_person(r["creator"]):   # drop names now ruled out
                out[r["creator"]] = [r[k] for k in FIELDS[1:]]
    to_lookup = []
    for creator, (name, birth, death, year) in people.items():
        if creator in out:
            continue
        if death:
            out[creator] = [name, birth or "", death,
                            "resolved" if death < AU_DEATH_CUTOFF else "died_1955_or_later", "dates in IA record"]
        elif birth and birth <= SAFE_BIRTH_YEAR:
            out[creator] = [name, birth, "", "resolved", f"born {birth}: would be over 104 in 1955"]
        else:
            to_lookup.append(creator)

    names = sorted({people[c][0] for c in to_lookup})
    print(f"Creators: {len(people)}  already known: {len(out)}  to look up: {len(names)} names", flush=True)
    cands = {}
    for i in range(0, len(names), BATCH):
        batch = names[i:i + BATCH]
        found = wikidata_candidates(batch)
        for n in batch:
            merged = {}
            for v in label_variants(n):
                merged.update(found.get(v, {}))
            cands[n] = merged
        print(f"  Wikidata {min(i + BATCH, len(names))}/{len(names)}")
        time.sleep(1)

    for k, creator in enumerate(to_lookup):
        name, birth, _, year = people[creator]
        loc = loc_birth_years(name) if cands[name] else set()
        status, death, evidence = judge(name, birth, year, cands[name], loc)
        out[creator] = [name, birth or "", death, status, evidence]
        if k % 100 == 0:
            print(f"  judged {k}/{len(to_lookup)}")
        time.sleep(0.3)

    with open(OUT_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        for creator in sorted(out):
            w.writerow([creator] + out[creator])
    counts = defaultdict(int)
    for v in out.values():
        counts[v[3]] += 1
    print(f"Authors : {dict(counts)}")
    print(f"Output  : {OUT_PATH}")


if __name__ == "__main__":
    main()
