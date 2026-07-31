# -*- coding: utf-8 -*-
"""Cross-check the adult-only revision against generated analysis outputs.

This verifier uses section-specific assertions rather than merely asking whether
a number appears somewhere in a document. Any FAIL is a submission blocker.
Placeholders that require author/institutional action are reported separately.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "output_adults"
R = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
A = json.loads((OUT / "reviewer2_addenda.json").read_text(encoding="utf-8"))

DOCS = {
    "manuscript": (ROOT / "sn-article-R1.tex").read_text(encoding="utf-8"),
    "letter": (ROOT / "RESPONSE_TO_REVIEWERS.md").read_text(encoding="utf-8"),
    "supplement": (ROOT / "supplementary" / "Supplementary_Material.tex").read_text(
        encoding="utf-8"),
    "cover": (ROOT / "cover_letter.tex").read_text(encoding="utf-8"),
}

fails: list[str] = []
passes: list[str] = []


def require(doc: str, label: str, pattern: str) -> None:
    if re.search(pattern, DOCS[doc], flags=re.I | re.S):
        passes.append(f"{doc}: {label}")
    else:
        fails.append(f"{doc}: missing/incorrect {label}")


def forbid(doc: str, label: str, pattern: str) -> None:
    hit = re.search(pattern, DOCS[doc], flags=re.I | re.S)
    if hit:
        line = DOCS[doc][:hit.start()].count("\n") + 1
        fails.append(f"{doc}: stale {label} at line {line}")
    else:
        passes.append(f"{doc}: no stale {label}")


n = R["sample"]["n_respondents"]
d = R["demographics"]
rg = R["regression"]
cl = R["clusters"]
pca = R["supplementary_pca"]
coef = {x["predictor"]: x for x in rg["coefficients"]}
corr = {x["variable"]: x for x in R["correlations"]}

# Manuscript: exact adult-only sample flow and headline results.
required_manuscript = {
    "final title": r"\\title\[[^]]*\]\{Exploring the Gap Between AI Experience and Formal Training Among Pre-service Mathematics Teachers\}",
    "adult analytic N": rf"analytic sample[^.]*\bN\s*=\s*{n}\b",
    "28 confirmed minors excluded": r"28\s+(?:respondents|records)[^.]{0,120}(?:minor|younger than 18|under 18)",
    "one unverified age excluded": r"(?:one|1)\s+(?:record|response)[^.]{0,120}(?:unverified|verifiable age|age)",
    "usage 84.5%": r"84\.5\\%",
    "training 22.6%": r"22\.6\\%",
    "gap 61.9 pp": r"61\.9\s+percentage points",
    "R2 .587": r"R\^2\s*=\s*\.587",
    "adjusted R2 .543": r"adjusted\s+\$?R\^2\$?\s*=\s*\.543",
    "F(8,75)=13.32": r"F\(8,\s*75\)\s*=\s*13\.32",
    "Ethical Concern beta .188 p .075": r"Ethical Concern[^\n]{0,300}0\.188[^\n]{0,300}\.075",
    "AI Literacy beta -.257": r"AI Literacy[^\n]{0,300}-0\.257",
    "semester beta .177 p .033": r"Academic semester[^\n]{0,300}0\.177[^\n]{0,300}\.033",
    "ordinal Ethical Concern p .025": r"Ethical Concern[^\n]{0,300}ordered logit[^\n]{0,100}p\s*=\s*\.025",
    "ordinal semester p .090": r"Academic semester[^\n]{0,300}ordered logit[^\n]{0,100}p\s*=\s*\.090",
    "outcome mean 3.81": r"Adoption intention[^\n]{0,160}3\.81",
    "full and predictor alpha": r"\.944[^\n]{0,100}\.942",
    "H2 not supported": r"H2\s+was\s+(?:likewise\s+)?(?:\textbf\{)?not supported",
    "adult figures": r"figures_adults/fig2_experience_profile\.png",
    "ethics approval date": r"Committee minutes dated 24 January 2025",
}
for label, pat in required_manuscript.items():
    require("manuscript", label, pat)

for label, pat in {
    "discarded expanded title": r"Ethical Concern Alongside AI Adoption Readiness",
    "analytic N=88": r"analytic sample[^.]{0,100}\bN\s*=\s*88\b",
    "N=113 figure caption": r"caption\{[^}]*\bN\s*=\s*113\b",
    "old usage/training": r"87\.6\\%[^\n]{0,120}22\.1\\%",
    "old H1 unsupported": r"H1\s+was\s+\\textbf\{not supported\}",
    "old cluster ANOVA": r"F\(2,\s*110\)\s*=\s*30\.41",
    "old Ethical Concern beta": r"Ethical Concern[^\n]{0,250}(?:0\.222|0\.168)",
    "old figures directory": r"includegraphics[^\n]*\{figures/",
    "false committee notification": r"Ethics Committee was notified",
    "false deletion statement": r"raw responses were deleted",
    "bias-corrected bootstrap wording": r"bias-corrected",
}.items():
    forbid("manuscript", label, pat)

# Letter and supplement must identify the final adult analysis and ethical remedy.
for doc in ("letter", "supplement"):
    require(doc, "final N=84", r"\bN\s*=\s*84\b|84 confirmed adults")
    require(doc, "28 minors excluded", r"28.{0,140}(?:minor|younger than\s+18|under\s+18)")
    require(doc, "one unverified age excluded", r"(?:one|1)[^\n.]{0,100}(?:unverified|verifiable age)")
    forbid(doc, "N=88 analytic sample", r"analytic sample[^.]{0,100}\bN\s*=\s*88\b")

require("letter", "voluntary title clarification disclosed",
        r"neither the Editor nor the reviewers explicitly requested a new title")
require("letter", "ethics approval date and no minute number",
        r"24 January 2025[^.]{0,180}minutes do not carry sequential reference numbers")
require("letter", "final title", r"Exploring the Gap Between AI Experience and Formal Training")
require("supplement", "final title", r"Exploring the Gap Between AI Experience and Formal Training")

require("supplement", "55-item PCA", r"55 predictor items")
require("supplement", "KMO adult", rf"{pca['kmo']:.3f}")
require("supplement", "adult cluster silhouette", rf"{cl['silhouette']:.3f}")
forbid("supplement", "minor outcome sensitivity", r"coefficients across three samples")

# The cover letter must describe the revised adult study, not the submitted analysis.
for label, pat in {
    "final N=84": r"84 confirmed adults",
    "final title": r"Exploring the Gap Between AI Experience and Formal Training",
    "title clarification disclosed": r"voluntary title clarification",
    "usage-training gap": r"84\.5\\%[^\n]{0,160}22\.6\\%[^\n]{0,160}61\.9",
    "adult model R2": r"58\.7\\%[^\n]{0,100}R\^2\s*=\s*\.543",
}.items():
    require("cover", label, pat)
for label, pat in {
    "old R2": r"R\^2\s*=\s*0\.652",
    "old negative ethics coefficient": r"-0\.28",
    "old usage/training": r"87\.6\\%|22\.1\\%",
    "old N=113 study": r"(?:study|investigation)[^.]{0,120}\b113\b",
}.items():
    forbid("cover", label, pat)

# Citation-key integrity.
tex = DOCS["manuscript"]
bib = (ROOT / "sn-bibliography-verified.bib").read_text(encoding="utf-8")
cited = {k.strip() for group in re.findall(r"\\cite\w*\{([^}]+)\}", tex)
         for k in group.split(",")}
defined = set(re.findall(r"@\w+\{([^,]+),", bib))
missing = sorted(cited - defined)
unused = sorted(defined - cited)
if missing:
    fails.append("bibliography: cited keys missing: " + ", ".join(missing))
else:
    passes.append("bibliography: all cited keys defined")
if unused:
    fails.append("bibliography: unused entries: " + ", ".join(unused))
else:
    passes.append("bibliography: no unused entries")

print("=" * 76)
print("ADULT-ONLY REVISION VERIFICATION")
print("=" * 76)
for item in passes:
    print("[PASS]", item)
for item in fails:
    print("[FAIL]", item)

blockers = []
for doc, body in DOCS.items():
    if "[COMMITTEE MINUTE NUMBER AND DATE]" in body:
        blockers.append(f"{doc}: committee minute date")
    if "[REPOSITORY DOI" in body or "[DOI to be inserted" in body:
        blockers.append(f"{doc}: repository DOI/access record")
for item in sorted(set(blockers)):
    print("[AUTHOR ACTION]", item)

print("=" * 76)
print("RESULT:", "ALL AUTOMATED CHECKS PASSED" if not fails else f"{len(fails)} CHECK(S) FAILED")
sys.exit(1 if fails else 0)
