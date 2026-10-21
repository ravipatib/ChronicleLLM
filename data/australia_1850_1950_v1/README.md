# Chronicle LLM v1 dataset

Australian writing 1788–1945, rebuilt from scratch for v1 with a written
licensing rule, machine-checked manifests, and verification on every download.

## Current build (21 Oct 2026)

| Source | Documents (cleaned) | Cleaned size | Share |
|---|---|---|---|
| Project Gutenberg (proofread) | 359 | 147.0 MB | 14% |
| Project Gutenberg Australia (proofread) | 281 | 71.4 MB | 7% |
| Internet Archive scanned books (OCR, quality-gated) | 1,380 | 463.9 MB | 43% |
| Commonwealth Hansard 1901–1945 (sample) | 2,137 | 393.2 MB | 37% |
| **Total** | **4,157** | **1,075.4 MB** | |

Tokens (GPT-2 BPE): **247.8M** (235.1M train / 12.7M val). Validation is 5% of
documents, chosen with seed 1850. Dropped at cleaning: 198 IA books with
unreadable OCR, 30 not in English, 62 too short.

Earlier builds for comparison: the first v1 build (20 Oct 2026) was 678.5 MB /
160.6M tokens; v0 was 139 files, 55 MB, ~14M tokens, of which only ~70 files
were valid (see [v0 audit](#v0-audit)).

Training config: [`config/train_chronicle_v1.py`](../../config/train_chronicle_v1.py)
(~73M params, 512 context, 3 epochs).

## Licensing rule

A text is used only if it is public domain in **both** Australia (where the
project is built and the weights are published from) and the United States
(where Project Gutenberg, the Internet Archive and Hugging Face operate).
Australia has no text and data mining exception (the government ruled one out
in October 2025), so "it's only for training" is not a defence for in-copyright
Australian text.

| Test | Rule | Source |
|------|------|--------|
| Australia | Every text contributor died before 1 Jan 1955 | Copyright Act 1968; [NLA: how long does copyright last](https://www.library.gov.au/research/access-collection/copyright-library-collections/how-long-does-copyright-last) |
| Australia, anonymous or corporate | Published before 1955, no named author (includes societies, museums, surveys) | same |
| Australia, government | Crown copyright lasts 50 years from making | Copyright Act s 180 |
| Australia, early authors | Born 1850 or earlier, or the book was published 1870 or earlier: the author would have to have lived past ~104 to die after 1954 | — |
| United States | Hosted on gutenberg.org (US public domain only), **or** first published ≤1930, **or** author died ≤1945 (out of Australian copyright on 1 Jan 1996, so not restored by the URAA) | [Duke CSPD, Public Domain Day 2026](https://web.law.duke.edu/cspd/publicdomainday/2026/) |

Anything that can't be shown to pass is marked `review` and is **not** downloaded.
The rule is applied to the contributors named in each catalogue record; a
compilation may contain short pieces by contributors the record doesn't list.

## Sources

| Source | Manifest | How selected | How fetched |
|--------|----------|--------------|-------------|
| Project Gutenberg | `gutenberg_manifest.csv` | Official offline catalogue (public domain). English texts with an Australian subject or bookshelf, or by a core Australian author; contributor death years come from the catalogue, then `author_deaths.csv` | `scripts/v1/fetch_gutenberg.py`: official mirror (Old Dominion University), rate-limited, never www.gutenberg.org ([robot policy](https://www.gutenberg.org/policy/robot_access.html)) |
| Project Gutenberg Australia | `pga_manifest.csv` | Only authors in `pga_authors.csv` (hand-curated death years). PGA also hosts books that are public domain only in Australia (Orwell, Fitzgerald, Woolf…), so nothing is taken in bulk. Titles needing a per-book decision are settled in `pga_title_decisions.csv` (see below) | `scripts/v1/fetch_pga.py`: 1 request per second, HTML converted to text |
| Internet Archive scanned books | `ia_manifest.csv` | IA scrape API, English texts dated 1788–1930, found two ways (`found_by` column): **topic** (Australian subjects or titles: 1,176 books kept, 420 MB) and **publisher** (printed in an Australian city or by Angus & Robertson / George Robertson, with no Australian subject tag: 204 books kept, 44 MB). Periodicals, journals, newspapers, catalogues, statistics, logbooks and pamphlets (<40 pages) are excluded; repeat scans and titles already taken from Gutenberg/PGA are removed | `scripts/v1/fetch_ia.py`: `<id>_djvu.txt`, 4 connections |
| Commonwealth Hansard 1901–1945 | `hansard_manifest.csv` | Evenly spaced sitting days in every year, both houses, ~400 MB of text in total so parliamentary language stays under ~40% of the corpus | `scripts/v1/fetch_hansard.py`: XML from [wragge/hansard-xml](https://github.com/wragge/hansard-xml) converted to `Speaker. — words` text under each debate heading |

Publisher-route items must name an Australian city as the **first** place of
publication. "London, Melbourne" imprints, British houses with Australian
branches (Heinemann, Cassell, Macmillan…), Sydney (Nova Scotia), Melbourne
(Florida), Hampden-Sydney College and bare city names are excluded.

Every Gutenberg and PGA download is checked against its manifest title, and
mismatches are not used (all 361 Gutenberg files matched; 10 PGA files were
rejected, one of them because PGA's own page misspells the title, and 2 PGA
pages are no longer online). PGA files
may be redistributed under the [PGA licence](https://gutenberg.net.au/licence.html)
provided the PGA header and trademark are removed, which `clean_v1.py` does.

## Author death years (`author_deaths.csv`)

IA and Gutenberg records often give an author's name without dates.
`scripts/v1/resolve_ia_authors.py` looks them up and writes one row per name
with its evidence; the manifest builders count a name only when its status is
`resolved` (died before 1955) or `died_1955_or_later` (excluded).

- Dates already in the record ("1836-1907", "d. 1900") are used as is; born
  1850 or earlier counts as safe.
- Otherwise every Wikidata person with exactly that name (label or alias) is
  collected. A candidate is *plausible* if they could have written a book
  published that year (born 15–100 years before it). If the record gives a
  birth year, it must match.
- A name is resolved only if **every** plausible candidate died before 1955.
  One plausible namesake with no death date, or who died later, leaves it
  unresolved. Library of Congress name authorities are a cross-check: an LC
  birth year for the same name must agree.
- Names that can't identify one person are never looked up: "Mrs" plus a
  husband's name, and religious names ("Agnes, Sister").

Result: 362 resolved, 64 died 1955 or later (e.g. C. E. W. Bean 1968,
Hubert Wilkins 1958, Zora Cross 1964, "Herbert Strang" 1958), 899 unresolved.

## PGA per-title decisions (`pga_title_decisions.csv`)

- **Authors who died 1946–1954** (Miles Franklin, J. H. M. Abbott, Aidan de
  Brune, Erle Cox, Henry Handel Richardson…) are public domain in Australia, but
  in the US only for works first published 1930 or earlier. The first
  publication (book or newspaper serial) was taken from each PGA page or the
  standard bibliography: 32 titles qualify (e.g. *Up the Country* 1928,
  *Ten Creeks Run* 1930, *The Way Home* 1925, *Ultima Thule* 1929), 27 don't
  (e.g. *All That Swagger* 1936, *Landtakers* 1934, *Back to Bool Bool* 1931).
- **Prolific authors with mostly non-Australian settings** (Fergus Hume,
  E. W. Hornung, Guy Boothby, B. L. Farjeon; all licence-safe) were scored by
  Australian place names and bush terms per 10,000 words. London-set books
  score under ~6, Australian-set ones 10–35; titles at 9.5 or above are kept
  (19, e.g. *Stingaree*, *The Belle of Toorak*, *Irralie's Bushranger*,
  *Miss Mephistopheles*, *The Rogue's March*), the other 92 are excluded.

## Build

```bash
curl -o pg_catalog.csv https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv
curl -o gutindex_aus.txt https://gutenberg.net.au/gutindex_aus.txt

python scripts/v1/build_gutenberg_manifest.py pg_catalog.csv
python scripts/v1/build_ia_manifest.py pg_catalog.csv
python scripts/v1/resolve_ia_authors.py          # Wikidata + LC lookups for undated authors
python scripts/v1/build_gutenberg_manifest.py pg_catalog.csv   # re-run with the lookups
python scripts/v1/build_ia_manifest.py pg_catalog.csv
python scripts/v1/fetch_gutenberg.py
python scripts/v1/fetch_pga.py gutindex_aus.txt --download
python scripts/v1/fetch_ia.py
python scripts/v1/fetch_hansard.py
python scripts/v1/clean_v1.py
python data/australia_1850_1950_v1/prepare.py   # needs tiktoken, numpy, tqdm
```

`author_deaths.csv` is committed, so the lookup step only queries names it
hasn't seen. Downloads take roughly 2–3 hours in total; every fetch script can be
re-run and skips files it already has.

`clean_v1.py` reuses the v0 `clean()` and adds:

- removal of Gutenberg and PGA boilerplate, including notes inside the book body
  (transcriber notes, "E-text prepared by", links);
- an English-language check;
- for scanned books: watermark removal, re-joining words hyphenated across
  lines, running-head removal, and a word-validity filter. A vocabulary is built
  from the proofread Gutenberg/PGA text in the same run; OCR paragraphs under
  90% known words are dropped, and a book is kept only if it is still ≥95% known
  words and keeps at least half its text;
- light handling for Hansard (whitespace only, so debate headings survive);
- corpus-wide paragraph de-duplication, processed in the order Gutenberg → PGA →
  IA → Hansard so proofread copies win.

`prepare.py` splits train/val by whole document with a fixed seed. v0 used the
alphabetically last 10%, which put mostly non-Australian books in validation.

## OCR quality (Internet Archive)

Proofread Gutenberg/PGA text scores ~0.999 on the vocabulary check; accepted IA
books score ≥0.95 after paragraph filtering. Expect a few character-level OCR
errors per thousand words in that part of the corpus ("moraUty" for
"morality"), plus about 100 one-off fragments of scanner stamps.

## Review queues (not downloaded)

- `ia_manifest.csv`, `decision=review` (868 rows), and
  `gutenberg_manifest.csv`, `decision=review` (38 rows): no author death year
  could be confirmed. Look the author up (ADB, Wikipedia, AustLit) and add a
  `resolved` row to `author_deaths.csv` only if they died before 1955, then
  re-run the builders.

## Not included, and why

- **Trove newspapers (NLA).** The text is mostly out of copyright, but the
  [Trove API Terms of Use](https://trove.nla.gov.au/about/create-something/using-api/trove-api-terms-use)
  are non-commercial, limit caching to 30 days, require real-time calls, and say
  the rights "do not extend to use of digital objects". A training corpus breaks
  those terms, so it needs written permission from the NLA first.
- **The Sydney Morning Herald scans on the Internet Archive** (23,856 issues,
  1840s–1910s). Licence-safe, but the OCR is unusable: about 4% of words are
  common English words, against ~40% for clean text.
- **Authors who died in or after 1955**, even when gutenberg.org hosts their
  book (US public domain): Mary Grant Bruce, Ethel Turner, Jeannie Gunn, Norman
  Lindsay, C. E. W. Bean, Douglas Mawson, K. S. Prichard, Dorothea Mackellar, etc.
- **Most of Hansard 1901–1945** (~1.1 GB of text is available) and all of it
  after 1945. A ~400 MB sample keeps it under ~40% of the corpus.

## Hansard licensing position

Commonwealth Crown copyright lasts 50 years (Copyright Act s 180), so debates up
to 1945 were out of copyright in Australia by 1996 at the latest, which also
means the US URAA did not restore them. Parliament publishes its Hansard XML
under CC BY-NC-ND. That licence cannot extend to material whose copyright has
expired, but it is a claim on the files, so using them was a deliberate decision
(20 Oct 2026). The project uses them non-commercially, credits the Parliament of
Australia and Tim Sherratt's GLAM Workbench harvest, and does not redistribute
the XML.

## v0 audit

`v0_audit.csv` checks every v0 raw file against its own Gutenberg header.
Of 139 files, **66 were the wrong book** (hand-typed Gutenberg IDs pointed at
Wodehouse, Balzac, Conan Doyle, Darwin, French/German/Dutch/Finnish/Bulgarian
texts, and one duplicate), and **9 are still in Australian copyright** (authors
died 1958–1986). Only ~70 v0 files were valid, so v0 results should be read with
that in mind.
