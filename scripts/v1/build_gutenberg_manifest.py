"""
build_gutenberg_manifest.py — Select Project Gutenberg texts for Chronicle LLM v1.

Reads the Project Gutenberg offline catalogue (pg_catalog.csv, public domain,
https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv) and writes
data/australia_1850_1950_v1/gutenberg_manifest.csv with one row per candidate
and an include/exclude/review decision plus the reason.

Licensing rule (must hold in BOTH jurisdictions):
  - United States : satisfied by being on gutenberg.org (US public domain only).
  - Australia     : every text contributor died before 1 January 1955
                    (Copyright Act 1968; NLA guidance). Illustrators are ignored
                    because illustrations are not used.
  - Anonymous works and government (Crown) publications are included: on
    gutenberg.org they are US public domain, and in Australia anonymous works
    published before 1955 and Crown works older than 50 years are out of copyright.
  - Unknown death year but born 1850 or earlier -> included (would have to have
    lived past 105 to still be in copyright).
  - Any other unknown death year -> "review", never auto-included.

Selection rule:
  - English text, and either an Australian subject/bookshelf/LoCC, or an author
    on the CORE_AUTHORS list (writers whose work is overwhelmingly Australian).

Usage:
    curl -o pg_catalog.csv https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv
    python scripts/v1/build_gutenberg_manifest.py pg_catalog.csv
"""

import csv
import os
import re
import sys

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_PATH = os.path.join(REPO_DIR, "data", "australia_1850_1950_v1", "gutenberg_manifest.csv")

AU_DEATH_CUTOFF = 1955   # contributor must have died before this year
EARLIEST_BIRTH  = 1700   # keeps out classical/medieval texts that mention Australia in subjects
SAFE_BIRTH      = 1850   # born this year or earlier => certainly died before 1955

ANONYMOUS  = re.compile(r"^(Anonymous|Unknown)$")
GOVERNMENT = re.compile(r"^(Australia|Queensland|New South Wales|Victoria|South Australia|"
                        r"Western Australia|Tasmania|Great Britain)\b")
NOT_TEXT   = re.compile(r"Linked Index|Index for Works|Index of the Project Gutenberg", re.I)

AU_SUBJECT = re.compile(
    r"Australia|Austral(ian|asia)|New South Wales|Victoria \(Australia\)|Queensland|"
    r"Tasmania|Van Diemen|Bushrangers|Port Jackson|Botany Bay|Gallipoli|Anzac"
)

# Surname, given-name prefix as they appear in the catalogue "Authors" field.
CORE_AUTHORS = [
    "Lawson, Henry", "Paterson, A. B.", "Paterson, Andrew Barton", "Dennis, C. J.",
    "Boldrewood, Rolf", "Clarke, Marcus", "Dyson, Edward", "Cambridge, Ada",
    "Praed, Campbell", "Gaunt, Mary", "Warung, Price", "Furphy, Joseph",
    "Franklin, Miles", "Baynton, Barbara", "Rudd, Steele", "Favenc, Ernest",
    "Kendall, Henry", "Gordon, Adam Lindsay", "O'Dowd, Bernard", "Brennan, Christopher",
    "Neilson, John Shaw", "Daley, Victor", "Quinn, Roderic", "Tasma",
    "Spence, Catherine Helen", "Mack, Louise", "Pedley, Ethel", "Becke, Louis",
    "Stone, Louis", "Bedford, Randolph", "Pratt, Ambrose", "Abbott, J. H. M.",
    "Atkinson, Louisa", "Harris, Alexander", "Rowcroft, Charles", "Nisbet, Hume",
    "Carboni, Raffaello", "Richardson, Henry Handel", "Lane, William",
    "Stephens, A. G.", "Evans, George Essex", "Stephens, James Brunton",
    "Sladen, Douglas", "Parker, K. Langloh", "Stevens, Bertram", "Esson, Louis",
    "Wright, David McKee", "Cuthbertson, James Lister", "Gay, William",
]

FIELDS = [
    "pg_id", "decision", "reason", "title", "authors", "max_death_year",
    "min_birth_year", "match", "subjects", "bookshelves",
]


def contributors(authors_field):
    """Yield (name, birth, death, role) for each contributor in the Authors field."""
    for part in authors_field.split(";"):
        part = part.strip()
        if not part:
            continue
        role = re.search(r"\[([^\]]+)\]", part)
        role = role.group(1) if role else "Author"
        name = re.sub(r"\s*\[[^\]]+\]", "", part)
        m = re.search(r"(\d{3,4})\??\s*(?:BCE)?-(\d{3,4})\??", name)
        birth = death = None
        if m:
            birth, death = int(m.group(1)), int(m.group(2))
        else:
            d = re.search(r"-(\d{4})", name)
            b = re.search(r"(\d{4})\??-\s*$", name)
            death = int(d.group(1)) if d else None
            birth = int(b.group(1)) if b else None
        yield name, birth, death, role


def classify(row):
    """Return (decision, reason, max_death, min_birth, match) for a catalogue row."""
    blob = " ".join([row["Subjects"], row["LoCC"], row["Bookshelves"]])
    by_subject = bool(AU_SUBJECT.search(blob))
    by_author  = any(re.search(re.escape(a) + r"(,|\s\(|\s\d|\s*$)", row["Authors"]) for a in CORE_AUTHORS)
    if not (by_subject or by_author):
        return None
    match = "subject+author" if by_subject and by_author else ("subject" if by_subject else "author")

    if NOT_TEXT.search(row["Title"]):
        return "exclude", "not a text: Gutenberg index page", "", "", match

    people = [c for c in contributors(row["Authors"]) if "Illustrator" not in c[3]]
    if not people:
        return "review", "no named text contributor", "", "", match

    births = [c[1] for c in people if c[1]]
    min_birth = min(births) if births else ""
    unknown = []
    for name, birth, death, _ in people:
        if death is not None:
            continue
        if ANONYMOUS.match(name.strip()) or GOVERNMENT.match(name.strip()):
            continue
        if birth is not None and birth <= SAFE_BIRTH:
            continue
        unknown.append(name)
    if unknown:
        return "review", "death year unknown: " + "; ".join(unknown), "", min_birth, match
    known = [c[2] for c in people if c[2] is not None]
    max_death = max(known) if known else ""

    if max_death and max_death >= AU_DEATH_CUTOFF:
        late = [c[0] for c in people if c[2] and c[2] >= AU_DEATH_CUTOFF]
        return "exclude", "AU copyright: contributor died >=1955: " + "; ".join(late), max_death, min_birth, match
    if min_birth and min_birth < EARLIEST_BIRTH:
        return "exclude", "out of era: contributor born before 1700", max_death, min_birth, match
    return "include", "PD in US (gutenberg.org) and AU (all contributors died <1955)", max_death, min_birth, match


def title_key(row):
    """Normalised (base title, first author surname) used to spot duplicate editions."""
    title = row["title"].lower()
    title = re.split(r"\s+[—-]\s+|;|:", title)[0]
    title = re.sub(r"[^a-z ]", "", title)
    title = re.sub(r"^(the|a|an) ", "", title).strip()
    surname = row["authors"].split(",")[0].strip().lower()
    return title, surname


def mark_duplicates(rows):
    """Exclude repeat editions, and single volumes when a 'Complete' edition exists."""
    first_seen = {}
    complete   = {title_key(r) for r in rows if r["decision"] == "include" and "complete" in r["title"].lower()}
    for r in rows:
        if r["decision"] != "include":
            continue
        key = title_key(r)
        is_volume = re.search(r"\bvol(ume)?\.?\s*([0-9]+|[IVX]+)\b", r["title"], re.I)
        if is_volume and key in complete and "complete" not in r["title"].lower():
            r["decision"], r["reason"] = "exclude", "duplicate: volume of a Complete edition"
        elif not is_volume and key in first_seen:
            r["decision"], r["reason"] = "exclude", f"duplicate: same title as pg_id {first_seen[key]}"
        elif not is_volume:
            first_seen[key] = r["pg_id"]


def main():
    catalog_path = sys.argv[1] if len(sys.argv) > 1 else "pg_catalog.csv"
    rows = csv.DictReader(open(catalog_path, encoding="utf-8"))

    out = []
    for row in rows:
        if row["Type"] != "Text" or row["Language"] != "en":
            continue
        result = classify(row)
        if result is None:
            continue
        decision, reason, max_death, min_birth, match = result
        out.append({
            "pg_id": row["Text#"], "decision": decision, "reason": reason,
            "title": " ".join(row["Title"].split()), "authors": row["Authors"],
            "max_death_year": max_death, "min_birth_year": min_birth, "match": match,
            "subjects": row["Subjects"], "bookshelves": row["Bookshelves"],
        })

    out.sort(key=lambda r: int(r["pg_id"]))
    mark_duplicates(out)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(out)

    counts = {}
    for r in out:
        counts[r["decision"]] = counts.get(r["decision"], 0) + 1
    print(f"Candidates : {len(out)}")
    for k, v in sorted(counts.items()):
        print(f"  {k:<8}: {v}")
    print(f"Output     : {OUT_PATH}")


if __name__ == "__main__":
    main()
