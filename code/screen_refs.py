# -*- coding: utf-8 -*-
"""
Screen the Scopus / Web of Science exports in ../refs and keep only literature
relevant to this study.

Stage 1  parse both export dialects
Stage 2  deduplicate (DOI, then normalised title)
Stage 3  hard eligibility filter (domain AND technology must both be present)
Stage 4  thematic assignment to the eight review blocks
Stage 5  rank within block and export shortlists for manual selection

Outputs to ./output/refs/
"""
from __future__ import annotations

import csv
import re
import sys
import io
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

HERE = Path(__file__).resolve().parent
REFS = HERE.parent / "refs"
OUT = HERE / "output" / "refs"
OUT.mkdir(parents=True, exist_ok=True)

CURRENT_YEAR = 2026


# --------------------------------------------------------------------------- #
# Stage 1 — parsing
# --------------------------------------------------------------------------- #
def split_entries(text: str):
    """Yield (entrytype, key, body) for each @TYPE{key, ...} block."""
    i, n = 0, len(text)
    while True:
        at = text.find("@", i)
        if at == -1:
            return
        brace = text.find("{", at)
        if brace == -1:
            return
        etype = text[at + 1:brace].strip().lower()
        if not re.fullmatch(r"[a-z]+", etype):
            i = at + 1
            continue
        depth, j = 0, brace
        while j < n:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        body = text[brace + 1:j]
        key, _, rest = body.partition(",")
        yield etype, key.strip(), rest
        i = j + 1


FIELD_RE = re.compile(r"([A-Za-z_\-]+)\s*=\s*", re.S)


def parse_fields(body: str) -> dict:
    """Field-by-field parse that respects nested braces and quoted values."""
    out, pos = {}, 0
    while True:
        m = FIELD_RE.search(body, pos)
        if not m:
            break
        name = m.group(1).strip().lower()
        k = m.end()
        while k < len(body) and body[k].isspace():
            k += 1
        if k >= len(body):
            break
        if body[k] == "{":
            depth, j = 0, k
            while j < len(body):
                if body[j] == "{":
                    depth += 1
                elif body[j] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            val, pos = body[k + 1:j], j + 1
        elif body[k] == '"':
            j = k + 1
            while j < len(body) and body[j] != '"':
                j += 1
            val, pos = body[k + 1:j], j + 1
        else:
            j = k
            while j < len(body) and body[j] not in ",\n":
                j += 1
            val, pos = body[k:j], j
        out[name] = " ".join(val.split())
    return out


def clean(s: str) -> str:
    s = re.sub(r"[{}]", "", s or "")
    s = s.replace("\\&", "&").replace("\\%", "%").replace("\\_", "_")
    return " ".join(s.split()).strip()


def norm_title(s: str) -> str:
    s = unicodedata.normalize("NFKD", clean(s).lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s)


records, per_file = [], Counter()
for path in sorted(REFS.glob("*.bib")):
    raw = path.read_text(encoding="utf-8", errors="replace")
    source = "Scopus" if path.name.lower().startswith("scopus") else "WoS"
    cnt = 0
    for etype, key, body in split_entries(raw):
        f = parse_fields(body)
        title = clean(f.get("title", ""))
        if not title:
            continue
        rec = {
            "source": source, "file": path.name, "bibtype": etype, "key": key,
            "authors": clean(f.get("author", "")),
            "title": title,
            "journal": clean(f.get("journal", "") or f.get("booktitle", "")
                             or f.get("series", "")),
            "year": clean(f.get("year", "")),
            "volume": clean(f.get("volume", "")),
            "number": clean(f.get("number", "")),
            "pages": clean(f.get("pages", "")),
            "doi": clean(f.get("doi", "")).lower().replace("https://doi.org/", ""),
            "abstract": clean(f.get("abstract", "")),
            "keywords": clean(f.get("author_keywords", "") or f.get("keywords", "")),
            "doctype": clean(f.get("type", "") or etype),
            "note": clean(f.get("note", "")),
        }
        m = re.search(r"Cited by:\s*(\d+)", rec["note"])
        rec["cited_by"] = int(m.group(1)) if m else None
        records.append(rec)
        cnt += 1
    per_file[path.name] = cnt

print("=" * 76)
print("STAGE 1 — PARSING")
print("=" * 76)
for k, v in per_file.items():
    print(f"   {k:<18} {v:6d} records")
print(f"   {'TOTAL':<18} {len(records):6d} records")

# --------------------------------------------------------------------------- #
# Stage 2 — deduplication
# --------------------------------------------------------------------------- #
by_doi, by_title, unique = {}, {}, []
dupes = 0
for r in records:
    doi, nt = r["doi"], norm_title(r["title"])
    if doi and doi in by_doi:
        keep = by_doi[doi]
        if len(r["abstract"]) > len(keep["abstract"]):
            keep.update({k: v for k, v in r.items() if v})
        dupes += 1
        continue
    if not doi and nt in by_title:
        dupes += 1
        continue
    unique.append(r)
    if doi:
        by_doi[doi] = r
    by_title[nt] = r

print("\n" + "=" * 76)
print("STAGE 2 — DEDUPLICATION")
print("=" * 76)
print(f"   duplicates removed : {dupes}")
print(f"   unique records     : {len(unique)}")
print(f"   with DOI           : {sum(1 for r in unique if r['doi'])}")
print(f"   with abstract      : {sum(1 for r in unique if r['abstract'])}")

# --------------------------------------------------------------------------- #
# Stage 3 — eligibility
# --------------------------------------------------------------------------- #
DOMAIN = r"(educat|teacher|teaching|student|learner|learning|pedagog|curricul|school|university|academ|classroom|instruction|assessment|literacy)"
TECH = r"(artificial intelligence|\bai\b|chatgpt|generative|large language|machine learning|llm|technology accept|utaut|\btam\b|tpack|digital|algorithm)"

# Off-topic domains that survive the keyword gate but are irrelevant here.
EXCLUDE = r"(cancer|tumor|clinical trial|patient|surgery|surgical|nursing home|drug|molecul|protein|genom|crop|agricultur|soil|wastewater|concrete|corrosion|battery|photovolta|antenna|wireless network|traffic flow|stock market|supply chain|bearing fault|remote sensing|radar|combustion|turbine|nanoparticle|catalys|seismic|petroleum|blockchain mining)"


def blob(r) -> str:
    return f"{r['title']} {r['abstract']} {r['keywords']}".lower()


eligible, rejected = [], []
for r in unique:
    b = blob(r)
    ok_domain = re.search(DOMAIN, b) is not None
    ok_tech = re.search(TECH, b) is not None
    off = re.search(EXCLUDE, b) is not None
    if ok_domain and ok_tech and not off:
        eligible.append(r)
    else:
        r["_reason"] = ("no education/teaching context" if not ok_domain else
                        "no AI/technology context" if not ok_tech else
                        "off-topic domain")
        rejected.append(r)

print("\n" + "=" * 76)
print("STAGE 3 — ELIGIBILITY FILTER")
print("=" * 76)
print(f"   eligible : {len(eligible)}")
print(f"   discarded: {len(rejected)}")
for reason, n in Counter(r["_reason"] for r in rejected).most_common():
    print(f"        {n:6d}  {reason}")

# --------------------------------------------------------------------------- #
# Stage 4 — thematic blocks
# --------------------------------------------------------------------------- #
BLOCKS = {
    "B1_acceptance_models": [
        (r"(technology accept|utaut|unified theory|\btam\b|theory of planned behav|behaviou?ral intention|adoption intention)", 3),
        (r"(pre-?service|preservice|student teacher|teacher candidate)", 2),
        (r"(structural equation|path analysis|latent profile)", 1),
    ],
    "B2_ai_literacy": [
        (r"(ai literacy|artificial intelligence literacy|ai competenc|ai readiness|digital competen)", 3),
        (r"(teacher education|teacher training|initial teacher|professional development)", 2),
        (r"(framework|scale|instrument|validat)", 1),
    ],
    "B3_readiness_efficacy": [
        (r"(self-?efficacy|ai anxiety|technology anxiety|teacher readiness|technostress|confidence)", 3),
        (r"(teacher|educator|pre-?service)", 2),
        (r"(attitude|perception|belief)", 1),
    ],
    "B4_tpack_mathematics": [
        (r"(tpack|technological pedagogical content)", 3),
        (r"(mathematic|algebra|calculus|geometry|statistics education|numeracy)", 3),
        (r"(teacher|instruction|classroom)", 1),
    ],
    "B5_ethics_dehumanization": [
        (r"(dehumaniz|humaniz|human-cent|care ethics|relational pedagog|human connection|empathy)", 4),
        (r"(algorithmic bias|ai ethics|ethical|responsible ai|fairness|equity|transparen|privacy)", 2),
        (r"(educat|teach|school)", 1),
    ],
    "B6_authentic_assessment": [
        (r"(authentic assessment|assessment design|assessment reform|academic integrity|rubric|formative assessment|summative)", 3),
        (r"(generative|chatgpt|artificial intelligence)", 2),
        (r"(higher education|mathematic|stem)", 1),
    ],
    "B7_psychometrics": [
        (r"(cronbach|coefficient alpha|mcdonald.{0,3}s omega|internal consistency|exploratory factor analysis|confirmatory factor|parallel analysis|sample size|scale development|psychometric)", 4),
        (r"(reliabilit|validit|measurement invariance)", 2),
    ],
    "B8_latin_america": [
        (r"(latin america|colombia|chile|mexic|peru|brazil|brasil|argentin|ecuador|venezuela|bolivia|uruguay|hispanic)", 4),
        (r"(digital divide|equity|inequalit|access|infrastructure)", 2),
        (r"(teacher|higher education|universit)", 1),
    ],
}

DOCTYPE_BONUS = {"article": 3, "review": 3, "conference paper": 0,
                 "conference": 0, "book chapter": 1, "inproceedings": 0}

for r in eligible:
    b = blob(r)
    scores = {}
    for block, pats in BLOCKS.items():
        s = sum(w for pat, w in pats if re.search(pat, b))
        # require the block's leading (most specific) pattern to fire
        if re.search(BLOCKS[block][0][0], b):
            scores[block] = s
    r["_scores"] = scores
    r["_best"] = max(scores, key=scores.get) if scores else None

    try:
        yr = int(r["year"])
    except ValueError:
        yr = 0
    recency = max(0, 6 - (CURRENT_YEAR - yr)) if yr else 0        # 0..6
    dt = (r["doctype"] or "").lower()
    dtb = next((v for k, v in DOCTYPE_BONUS.items() if k in dt), 1)
    cite = min(3, (r["cited_by"] or 0) / 20)
    r["_year_int"] = yr
    r["_rank"] = (scores.get(r["_best"], 0) * 4 + recency * 1.5 + dtb
                  + cite + (2 if r["doi"] else 0)
                  + (1 if r["abstract"] else 0))

assigned = [r for r in eligible if r["_best"]]
unassigned = [r for r in eligible if not r["_best"]]

print("\n" + "=" * 76)
print("STAGE 4 — THEMATIC ASSIGNMENT")
print("=" * 76)
by_block = defaultdict(list)
for r in assigned:
    by_block[r["_best"]].append(r)
for block in BLOCKS:
    print(f"   {block:<28} {len(by_block[block]):5d}")
print(f"   {'(no block matched)':<28} {len(unassigned):5d}")

# --------------------------------------------------------------------------- #
# Stage 5 — shortlists
# --------------------------------------------------------------------------- #
TOP_N = 30
FIELDS = ["rank", "block", "year", "doctype", "cited_by", "doi", "authors",
          "title", "journal", "volume", "number", "pages", "source", "key",
          "abstract"]

shortlist = []
for block in BLOCKS:
    rows = sorted(by_block[block], key=lambda r: -r["_rank"])[:TOP_N]
    for r in rows:
        shortlist.append({
            "rank": round(r["_rank"], 1), "block": block, "year": r["year"],
            "doctype": r["doctype"], "cited_by": r["cited_by"] or "",
            "doi": r["doi"], "authors": r["authors"], "title": r["title"],
            "journal": r["journal"], "volume": r["volume"],
            "number": r["number"], "pages": r["pages"], "source": r["source"],
            "key": r["key"], "abstract": r["abstract"][:1200],
        })

with open(OUT / "shortlist.csv", "w", encoding="utf-8-sig", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(shortlist)

with open(OUT / "eligible_all.csv", "w", encoding="utf-8-sig", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    w.writeheader()
    for r in sorted(assigned, key=lambda r: (r["_best"], -r["_rank"])):
        w.writerow({
            "rank": round(r["_rank"], 1), "block": r["_best"], "year": r["year"],
            "doctype": r["doctype"], "cited_by": r["cited_by"] or "",
            "doi": r["doi"], "authors": r["authors"], "title": r["title"],
            "journal": r["journal"], "volume": r["volume"],
            "number": r["number"], "pages": r["pages"], "source": r["source"],
            "key": r["key"], "abstract": r["abstract"][:600],
        })

with open(OUT / "discarded.csv", "w", encoding="utf-8-sig", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["reason", "year", "doi", "title", "journal"])
    for r in rejected:
        w.writerow([r["_reason"], r["year"], r["doi"], r["title"], r["journal"]])
    for r in unassigned:
        w.writerow(["eligible but no thematic block", r["year"], r["doi"],
                    r["title"], r["journal"]])

print("\n" + "=" * 76)
print("STAGE 5 — SHORTLISTS WRITTEN")
print("=" * 76)
print(f"   shortlist.csv    {len(shortlist):5d} rows (top {TOP_N} per block)")
print(f"   eligible_all.csv {len(assigned):5d} rows")
print(f"   discarded.csv    {len(rejected) + len(unassigned):5d} rows")
print(f"\n   -> {OUT}")

print("\n" + "=" * 76)
print("TOP 8 PER BLOCK")
print("=" * 76)
for block in BLOCKS:
    print(f"\n--- {block} ---")
    for r in sorted(by_block[block], key=lambda r: -r["_rank"])[:8]:
        a = r["authors"].split(" and ")[0].split(",")[0] if r["authors"] else "?"
        print(f"  [{r['_rank']:5.1f}] {r['year']} {a:<18} {r['title'][:82]}")
        print(f"          {r['journal'][:70]}  doi:{r['doi'] or '—'}")
