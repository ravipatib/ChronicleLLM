# Chronicle LLM v1 dataset

Australian writing 1788–1945, rebuilt from scratch for v1 with a written
licensing rule, machine-checked manifests, and verification on every download.

## Current build (20 Oct 2026)

| Source | Documents (cleaned) | Cleaned size | Share |
|---|---|---|---|
| Project Gutenberg (proofread) | 356 | 145.5 MB | 21% |
| Project Gutenberg Australia (proofread) | 232 | 56.1 MB | 8% |
| Internet Archive scanned books (OCR, quality-gated) | 907 | 334.4 MB | 49% |
| Commonwealth Hansard 1901–1945 (sample) | 804 | 142.5 MB | 21% |
| **Total** | **2,299** | **678.5 MB** | |

Tokens (GPT-2 BPE): **160.6M** (151.6M train / 9.0M val). Validation is 5% of
documents, chosen with seed 1850. Dropped at cleaning: 125 IA books with
unreadable OCR, 17 not in English, 40 too short.

v0 for comparison: 139 files, 55 MB, ~14M tokens, of which only ~70 files were
valid (see [v0 audit](#v0-audit)).

Training config: [`config/train_chronicle_v1.py`](../../config/train_chronicle_v1.py)
(~51M params, 512 context, ~2 epochs).

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
| Australia, anonymous | Published before 1955 | same |
| Australia, government | Crown copyright lasts 50 years from making | Copyright Act s 180 |
| Australia, pre-1871 books | Published 1870 or earlier: an author would have to have lived past ~103 to die after 1954 | — |
| United States | Hosted on gutenberg.org (US public domain only), **or** first published ≤1930, **or** author died ≤1945 (out of Australian copyright on 1 Jan 1996, so not restored by the URAA) | [Duke CSPD, Public Domain Day 2026](https://web.law.duke.edu/cspd/publicdomainday/2026/) |

Anything that can't be shown to pass is marked `review` and is **not** downloaded.

## Sources

| Source | Manifest | How selected | How fetched |
|--------|----------|--------------|-------------|
| Project Gutenberg | `gutenberg_manifest.csv` | Official offline catalogue (public domain). English texts with an Australian subject or bookshelf, or by a core Australian author; contributor death years come from the catalogue | `scripts/v1/fetch_gutenberg.py`: official mirror (Old Dominion University), rate-limited, never www.gutenberg.org ([robot policy](https://www.gutenberg.org/policy/robot_access.html)) |
| Project Gutenberg Australia | `pga_manifest.csv` | Only authors in `pga_authors.csv` (hand-curated death years). PGA also hosts books that are public domain only in Australia (Orwell, Fitzgerald, Woolf…), so nothing is taken in bulk | `scripts/v1/fetch_pga.py`: 1 request per second, HTML converted to text |
| Internet Archive scanned books | `ia_manifest.csv` | IA scrape API: English texts with Australian subjects or titles, dated 1788–1930. Undated authors are looked up in the Gutenberg catalogue and PGA list by exact name. Periodicals, journals, newspapers, catalogues, statistics, logbooks and pamphlets (<40 pages) are excluded; repeat scans and titles already taken from Gutenberg/PGA are removed | `scripts/v1/fetch_ia.py`: `<id>_djvu.txt`, 4 connections |
| Commonwealth Hansard 1901–1945 | `hansard_manifest.csv` | Evenly spaced sitting days in every year, both houses, ~150 MB of text in total so parliamentary language doesn't dominate | `scripts/v1/fetch_hansard.py`: XML from [wragge/hansard-xml](https://github.com/wragge/hansard-xml) converted to `Speaker. — words` text under each debate heading |

Every Gutenberg and PGA download is checked against its manifest title, and
mismatches are moved to `raw_rejected/` (all 357 Gutenberg files matched; 9 PGA
files were rejected). PGA files may be redistributed under the
[PGA licence](https://gutenberg.net.au/licence.html) provided the PGA header and
trademark are removed, which `clean_v1.py` does.

## Build

```bash
curl -o pg_catalog.csv https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv
curl -o gutindex_aus.txt https://gutenberg.net.au/gutindex_aus.txt

python scripts/v1/build_gutenberg_manifest.py pg_catalog.csv
python scripts/v1/fetch_gutenberg.py
python scripts/v1/fetch_pga.py gutindex_aus.txt --download
python scripts/v1/build_ia_manifest.py pg_catalog.csv
python scripts/v1/fetch_ia.py
python scripts/v1/fetch_hansard.py
python scripts/v1/clean_v1.py
python data/australia_1850_1950_v1/prepare.py   # needs tiktoken, numpy, tqdm
```

Downloads take roughly 1–2 hours in total; every fetch script can be re-run and
skips files it already has.

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
errors per thousand words in that half of the corpus, plus about 100 one-off
fragments of scanner stamps across the whole corpus.

## Review queues (not downloaded)

- `ia_manifest.csv`, `decision=review` (~1,080 rows), and
  `gutenberg_manifest.csv`, `decision=review` (44 rows): an author's death year
  wasn't found. Look the author up (ADB, Wikipedia, AustLit) and change the row
  to `include` only if they died before 1955.
- `pga_manifest.csv`, `decision=review` (170 rows):
  - authors who died 1946–1954 (Miles Franklin, J. H. M. Abbott, Erle Cox…):
    include a title only if it was **first published 1930 or earlier**;
  - Fergus Hume, E. W. Hornung, Guy Boothby, B. L. Farjeon: licence-safe, but
    most titles are set in London or Europe; pick the Australian ones by hand.

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
- **Hansard after 1945, and most of 1901–1945.** Only a ~150 MB sample is used.

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
