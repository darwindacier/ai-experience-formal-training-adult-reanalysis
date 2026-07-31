# -*- coding: utf-8 -*-
"""
================================================================================
REANALYSIS — "Exploring the Gap Between AI Experience and Formal Training
              Among Pre-service Mathematics Teachers"
Discover Education · Manuscript 81b6503e-46c8-490d-875f-7b761a629e07 · Revision R1
================================================================================

Fully reproducible pipeline for the ethics-compliant analytic sample. Every
statistic reported in the revised manuscript is produced after retaining only
respondents whose reported age confirms that they were adults (age >= 18).
No outcome from excluded records is analysed.

Run:      python reanalysis.py
Outputs:  ./output/*.csv, ./output/results.json, ./output/console_log.txt

Requires: pandas, numpy, scipy, scikit-learn
================================================================================
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import io
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from statsmodels.miscmodels.ordinal_model import OrderedModel

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SEED = 42
N_BOOT = 5000
rng = np.random.default_rng(SEED)

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = (HERE.parent.parent
               / "Uso de la Inteligencia Artificial en la Formación de Docentes de Matemáticas .csv")
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=Path, default=DEFAULT_CSV,
                    help="Raw or de-identified survey CSV (default: original export).")
args = parser.parse_args()
CSV = args.data.resolve()
OUT = HERE / "output_adults"
OUT.mkdir(exist_ok=True)

# Tee console output to a log file so the run is auditable.
class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, s):
        for st in self.streams: st.write(s)
    def flush(self):
        for st in self.streams: st.flush()

_log = open(OUT / "console_log.txt", "w", encoding="utf-8")
sys.stdout = _Tee(io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8"), _log)

RESULTS: dict = {}


def head(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# --------------------------------------------------------------------------- #
# 1. Load and classify
# --------------------------------------------------------------------------- #
head("1. DATA LOADING AND VARIABLE CLASSIFICATION")

df_raw = pd.read_csv(CSV, encoding="utf-8")
cols = list(df_raw.columns)

def find_column(fragment: str, fallback: int) -> str:
    hits = [c for c in cols if fragment.casefold() in c.casefold()]
    return hits[0] if hits else cols[fallback]

COL_AGE = find_column("edad", 2)
COL_GENDER = find_column("género", 3)
if "género" not in COL_GENDER.casefold():
    COL_GENDER = find_column("genero", 3)
COL_SEMESTER = find_column("semestre", 4)

# Values include forms such as "17 años" and "20 Años". Numeric coercion alone
# incorrectly treats them as missing. Extract the first 1--3 digit number; a
# response containing a name rather than an age remains unverified.
age_raw = df_raw[COL_AGE].astype("string")
age_parsed = pd.to_numeric(age_raw.str.extract(r"(\d{1,3})", expand=False),
                           errors="coerce")
is_minor = age_parsed.lt(18)
is_adult = age_parsed.ge(18)
is_unverified = age_parsed.isna()

print("ETHICS-COMPLIANT SAMPLE SCREENING")
print(f"   responses received       : {len(df_raw)}")
print(f"   confirmed minors excluded: {int(is_minor.sum())}")
print(f"   age unverified excluded  : {int(is_unverified.sum())}")
print(f"   confirmed adults retained: {int(is_adult.sum())}\n")

df = df_raw.loc[is_adult].copy().reset_index(drop=True)
df[COL_AGE] = age_parsed.loc[is_adult].astype(int).to_numpy()

metadata = {COL_AGE, COL_GENDER, COL_SEMESTER}
metadata.update(c for c in cols if c.casefold() in {
    "marca temporal", "nombre de usuario", "participant_id"})
item_cols = [c for c in cols if c not in metadata]

dichotomous, likert = [], []
for c in item_cols:
    numeric = pd.to_numeric(df[c], errors="coerce")
    if numeric.notna().all() and numeric.between(1, 5).all():
        likert.append(c)
    else:
        dichotomous.append(c)

print(f"Respondents (rows)            : {len(df)}")
print(f"Columns                       : {df.shape[1]}")
print(f"Demographic variables         : 5")
print(f"Questionnaire items           : {len(item_cols)}")
print(f"  - Likert (1-5)              : {len(likert)}")
print(f"  - Dichotomous (Yes/No)      : {len(dichotomous)}")

X = df[likert].astype(float)
print(f"Missing values in Likert items: {int(X.isna().sum().sum())}")

RESULTS["sample"] = {
    "n_received": int(len(df_raw)),
    "n_excluded_confirmed_minors": int(is_minor.sum()),
    "n_excluded_unverified_age": int(is_unverified.sum()),
    "n_respondents": int(len(df)),
    "n_items_total": len(item_cols),
    "n_likert": len(likert),
    "n_dichotomous": len(dichotomous),
    "likert_missing": int(X.isna().sum().sum()),
}

# --------------------------------------------------------------------------- #
# 2. Demographics
# --------------------------------------------------------------------------- #
head("2. PARTICIPANT CHARACTERISTICS")

gender = df[COL_GENDER].astype(str).str.strip()
n_f = int((gender == "Femenino").sum())
n_m = int((gender == "Masculino").sum())
print(f"Gender: female {n_f} ({n_f/len(df)*100:.1f}%) | male {n_m} ({n_m/len(df)*100:.1f}%)")

age = pd.to_numeric(df[COL_AGE], errors="coerce")
n_age = int(age.notna().sum())
n_age_missing = int(len(df) - n_age)
print(f"\nAge: n = {n_age} valid, {n_age_missing} missing "
      f"({n_age_missing/len(df)*100:.1f}%)")
print(f"     range {age.min():.0f}-{age.max():.0f}, "
      f"M = {age.mean():.2f}, SD = {age.std(ddof=1):.2f}, Mdn = {age.median():.0f}")

# Band labels follow the observed minimum, so they stay accurate when the
# sample is restricted (the lowest band is 18-20, not 16-20, under
# --exclude-minors).
_lo = int(age.min())
age_bands = {
    f"{_lo}-20": int((age <= 20).sum()),
    "21-25": int(((age >= 21) & (age <= 25)).sum()),
    "26+":   int((age > 25).sum()),
}
print("\nAge bands (percentages computed over n = %d with valid age):" % n_age)
for k, v in age_bands.items():
    print(f"     {k:>6}: {v:3d} ({v/n_age*100:5.1f}%)")

semester = df[COL_SEMESTER].astype(str).str.extract(r"(\d+)")[0].astype(float)
levels = {
    "Initial (1-3)":      int(semester.between(1, 3).sum()),
    "Intermediate (4-6)": int(semester.between(4, 6).sum()),
    "Advanced (7-9)":     int(semester.between(7, 9).sum()),
}
print("\nAcademic level:")
for k, v in levels.items():
    print(f"     {k:<20}: {v:3d} ({v/len(df)*100:5.1f}%)")

print("\nDichotomous items (count and % answering 'Si'):")
dich_summary = {}
for c in dichotomous:
    s = int((df[c].astype(str).str.strip() == "Si").sum())
    dich_summary[c.strip()] = {"n_yes": s, "pct": round(s / len(df) * 100, 1)}
    print(f"     {s:3d} ({s/len(df)*100:5.1f}%)  {c.strip()[:66]}")

used_ai = dich_summary[[k for k in dich_summary if "utilizado herramientas" in k][0]]
trained = dich_summary[[k for k in dich_summary if "cursos o talleres" in k][0]]
gap = round(used_ai["pct"] - trained["pct"], 1)
print(f"\n>> Usage-training gap: {used_ai['pct']}% used AI vs {trained['pct']}% formally "
      f"trained = {gap} percentage points")

RESULTS["demographics"] = {
    "female": n_f, "male": n_m,
    "female_pct": round(n_f / len(df) * 100, 1),
    "male_pct": round(n_m / len(df) * 100, 1),
    "age_n_valid": n_age, "age_n_missing": n_age_missing,
    "age_missing_pct": round(n_age_missing / len(df) * 100, 1),
    "age_min": float(age.min()), "age_max": float(age.max()),
    "age_mean": round(float(age.mean()), 2),
    "age_sd": round(float(age.std(ddof=1)), 2),
    "age_bands": age_bands,
    "levels": levels,
    "dichotomous": dich_summary,
    "usage_training_gap_pp": gap,
}

# --------------------------------------------------------------------------- #
# 3. Reliability helpers
# --------------------------------------------------------------------------- #
def cronbach_alpha(d: pd.DataFrame) -> float:
    k = d.shape[1]
    if k < 2:
        return float("nan")
    return k / (k - 1) * (1 - d.var(ddof=1).sum() / d.sum(axis=1).var(ddof=1))


def alpha_if_deleted(d: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {c: cronbach_alpha(d.drop(columns=c)) for c in d.columns}, dtype=float
    )


head("3. RELIABILITY OF THE FULL INSTRUMENT")
alpha_full = cronbach_alpha(X)
print(f"Cronbach's alpha, all {len(likert)} Likert items: {alpha_full:.3f}")
RESULTS["alpha_full_scale"] = round(float(alpha_full), 3)

# --------------------------------------------------------------------------- #
# 4. Theory-informed composite scales
# --------------------------------------------------------------------------- #
head("4. THEORY-INFORMED COMPOSITE SCALES (defined for the reanalysis)")

print(f"""Rationale: with N = {len(X)} and {len(likert)} items the participant-to-item
ratio is {len(X)/len(likert):.2f}:1, far below the 5:1-10:1 conventionally required for
exploratory factor analysis. The primary analyses therefore use subscales
defined transparently from item content and the TAM/UTAUT and AI-ethics
literature. They are post hoc composites, not a validated measurement model. The exploratory
component analysis is retained only as supplementary, descriptive evidence
(Section 9).""")

OUTCOME_KEY = "Planeo incorporar"

SUBSCALE_SPEC = {
    "Perceived Value": [
        "Creo que la IA es una herramienta valiosa",
        "La IA puede facilitar el aprendizaje personalizado",
        "La implementación de la IA en el aula es esencial",
        "La IA aumentará la eficiencia",
        "Considero que la IA puede motivar",
        "La IA podría reducir la carga administrativa",
        "Estoy entusiasmado/a con las posibilidades",
        "Veo un gran potencial",
        "Creo que la IA puede ayudar a identificar",
        "La IA puede contribuir a crear",
    ],
    "AI Literacy": [
        "Estoy familiarizado/a con los conceptos",
        "Entiendo cómo funciona",
        "Conozco ejemplos prácticos",
        "Puedo explicar a otros",
        "Estoy al tanto de las últimas",
        "Sé diferenciar entre",
        "Comprendo las limitaciones",
    ],
    "Technical Self-Efficacy": [
        "Me siento preparado/a para integrar",
        "Tengo las habilidades necesarias",
        "Confío en mi capacidad",
        "Me siento cómodo/a experimentando",
        "Puedo identificar recursos",
    ],
    "Ethical Concern": [
        "Me preocupa que la IA pueda deshumanizar",
        "Me preocupa la privacidad",
        "La IA puede aumentar las brechas",
        "La dependencia excesiva de la IA",
        "Me preocupa que la IA pueda generar sesgos",
    ],
    "Ethical Governance": [
        "Es fundamental abordar las implicaciones éticas",
        "Se requieren políticas claras",
        "La IA debe ser utilizada respetando",
        "Es importante garantizar la transparencia",
    ],
    "Institutional Support": [
        "Existe apoyo institucional",
        "Los cursos de mi programa incluyen",
        "Se fomenta activamente",
        "He experimentado el uso de la IA en clases",
        "Mis profesores están capacitados",
        "He tenido oportunidades de desarrollar",
        "He recibido formación formal",
    ],
    "Perceived Barriers": [
        "La falta de recursos tecnológicos",
        "Las dificultades técnicas",
        "Hay resistencia al cambio",
        "La carga académica actual impide",
    ],
}


def match_item(fragment: str) -> str:
    hits = [c for c in likert if fragment.lower() in c.lower()]
    if len(hits) != 1:
        raise ValueError(f"Ambiguous or missing item for fragment {fragment!r}: {hits}")
    return hits[0]


outcome_col = match_item(OUTCOME_KEY)
y = X[outcome_col]
print(f"\nOutcome variable: \"{outcome_col.strip()}\"")
print(f"   M = {y.mean():.2f}, SD = {y.std(ddof=1):.2f}, "
      f"range {y.min():.0f}-{y.max():.0f}")
print("   NOTE: single-item behavioural-intention measure. Reported as a limitation;")
print("         no reliability coefficient can be estimated for a single item.")

subscales, subscale_items, rel_rows = {}, {}, []
for name, fragments in SUBSCALE_SPEC.items():
    items = [match_item(f) for f in fragments]
    if outcome_col in items:                      # never let the outcome predict itself
        items.remove(outcome_col)
    d = X[items]
    a = cronbach_alpha(d)
    subscales[name] = d.mean(axis=1)
    subscale_items[name] = items
    rel_rows.append({
        "subscale": name, "k_items": len(items), "alpha": round(float(a), 3),
        "mean": round(float(d.mean(axis=1).mean()), 2),
        "sd": round(float(d.mean(axis=1).std(ddof=1)), 2),
        "min": round(float(d.mean(axis=1).min()), 2),
        "max": round(float(d.mean(axis=1).max()), 2),
    })

S = pd.DataFrame(subscales)
rel = pd.DataFrame(rel_rows)
print("\n" + rel.to_string(index=False))
rel.to_csv(OUT / "table_subscale_reliability.csv", index=False)
RESULTS["subscales"] = rel.to_dict("records")

n_assigned = sum(len(v) for v in subscale_items.values())
assigned_set = {i for v in subscale_items.values() for i in v} | {outcome_col}
unassigned = [c for c in likert if c not in assigned_set]
print(f"\nItems assigned to subscales: {n_assigned} of {len(likert)} "
      f"(+1 outcome item = {n_assigned + 1}); "
      f"{len(unassigned)} items retained in the instrument but not assigned.")
print("Unassigned items (reported for transparency; they measure single-indicator")
print("constructs or overlap conceptually with more than one subscale):")
for c in unassigned:
    print(f"   - {c.strip()[:74]}")
pd.DataFrame({"unassigned_item": [c.strip() for c in unassigned]}) \
    .to_csv(OUT / "supp_unassigned_items.csv", index=False)
RESULTS["items_assigned"] = n_assigned
RESULTS["items_unassigned"] = [c.strip() for c in unassigned]

# alpha-if-item-deleted, for the supplementary material
rows = []
for name, items in subscale_items.items():
    aid = alpha_if_deleted(X[items])
    base = cronbach_alpha(X[items])
    for it, a in aid.items():
        rows.append({"subscale": name, "item": it.strip(),
                     "alpha_if_deleted": round(float(a), 3),
                     "alpha_current": round(float(base), 3)})
pd.DataFrame(rows).to_csv(OUT / "supp_alpha_if_item_deleted.csv", index=False)

# --------------------------------------------------------------------------- #
# 5. Correlations
# --------------------------------------------------------------------------- #
head("5. BIVARIATE CORRELATIONS WITH ADOPTION INTENTION")

dehum_col = match_item("Me preocupa que la IA pueda deshumanizar")
dehum = X[dehum_col]

corr_rows = []
for label, series in list(S.items()) + [("Academic semester", semester),
                                        ("Dehumanization (single item)", dehum)]:
    ok = series.notna()
    r, p = stats.pearsonr(series[ok], y[ok])
    n_ok = int(ok.sum())
    # Fisher z CI
    z = np.arctanh(r); se = 1 / np.sqrt(n_ok - 3)
    lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    corr_rows.append({"variable": label, "n": n_ok, "r": round(float(r), 3),
                      "ci_low": round(float(lo), 3), "ci_high": round(float(hi), 3),
                      "p": round(float(p), 4)})

corr = pd.DataFrame(corr_rows)
print(corr.to_string(index=False))
corr.to_csv(OUT / "table_correlations.csv", index=False)
RESULTS["correlations"] = corr.to_dict("records")

print(f"\nDehumanization item descriptives: M = {dehum.mean():.2f}, "
      f"SD = {dehum.std(ddof=1):.2f}; "
      f"{(dehum >= 4).mean()*100:.1f}% agreed or strongly agreed (4-5)")
RESULTS["dehumanization_item"] = {
    "mean": round(float(dehum.mean()), 2),
    "sd": round(float(dehum.std(ddof=1)), 2),
    "pct_agree": round(float((dehum >= 4).mean() * 100), 1),
}

# --------------------------------------------------------------------------- #
# 6. Multiple regression
# --------------------------------------------------------------------------- #
head("6. MULTIPLE REGRESSION (simultaneous entry)")


def ols(Xmat: np.ndarray, yvec: np.ndarray):
    A = np.column_stack([np.ones(len(Xmat)), Xmat])
    beta, *_ = np.linalg.lstsq(A, yvec, rcond=None)
    resid = yvec - A @ beta
    n, k = A.shape
    dof = n - k
    mse = (resid ** 2).sum() / dof
    cov = mse * np.linalg.pinv(A.T @ A)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), dof)
    ss_tot = ((yvec - yvec.mean()) ** 2).sum()
    r2 = 1 - (resid ** 2).sum() / ss_tot
    adj = 1 - (1 - r2) * (n - 1) / dof
    f = (r2 / (k - 1)) / ((1 - r2) / dof)
    return dict(beta=beta, se=se, t=t, p=p, r2=r2, adj=adj, f=f,
                df1=k - 1, df2=dof, n=n, resid=resid)


D = pd.concat([S, semester.rename("Academic semester")], axis=1)
D["_y"] = y.values
D = D.dropna()
pred_names = [c for c in D.columns if c != "_y"]

Xz = ((D[pred_names] - D[pred_names].mean()) / D[pred_names].std(ddof=1)).values
yz = ((D["_y"] - D["_y"].mean()) / D["_y"].std(ddof=1)).values

m = ols(Xz, yz)
print(f"N = {m['n']} (listwise; {len(df) - m['n']} dropped for missing semester)")
print(f"R2 = {m['r2']:.3f}   adjusted R2 = {m['adj']:.3f}   "
      f"F({m['df1']}, {m['df2']}) = {m['f']:.2f}, p = {stats.f.sf(m['f'], m['df1'], m['df2']):.3g}")

# Percentile bootstrap CIs for standardized coefficients
boot = np.empty((N_BOOT, len(pred_names)))
for b in range(N_BOOT):
    idx = rng.integers(0, len(Xz), len(Xz))
    boot[b] = ols(Xz[idx], yz[idx])["beta"][1:]
lo_b, hi_b = np.percentile(boot, [2.5, 97.5], axis=0)

# Analytic CIs
tcrit = stats.t.ppf(0.975, m["df2"])
lo_a = m["beta"][1:] - tcrit * m["se"][1:]
hi_a = m["beta"][1:] + tcrit * m["se"][1:]

# HC3 heteroskedasticity-robust inference as a sensitivity check.
A = np.column_stack([np.ones(len(Xz)), Xz])
bread = np.linalg.pinv(A.T @ A)
hat = np.einsum("ij,jk,ik->i", A, bread, A)
scaled = m["resid"] / np.clip(1 - hat, 1e-12, None)
hc3_cov = bread @ (A.T @ ((scaled ** 2)[:, None] * A)) @ bread
hc3_se = np.sqrt(np.diag(hc3_cov))[1:]
hc3_t = m["beta"][1:] / hc3_se
hc3_p = 2 * stats.t.sf(np.abs(hc3_t), m["df2"])

# VIF
vif = []
for i, c in enumerate(pred_names):
    others = np.delete(Xz, i, axis=1)
    r2_i = ols(others, Xz[:, i])["r2"]
    vif.append(1 / (1 - r2_i))

reg = pd.DataFrame({
    "predictor": pred_names,
    "beta": np.round(m["beta"][1:], 3),
    "se": np.round(m["se"][1:], 3),
    "t": np.round(m["t"][1:], 2),
    "p": np.round(m["p"][1:], 4),
    "ci_low": np.round(lo_a, 3), "ci_high": np.round(hi_a, 3),
    "boot_ci_low": np.round(lo_b, 3), "boot_ci_high": np.round(hi_b, 3),
    "hc3_se": np.round(hc3_se, 3), "hc3_p": np.round(hc3_p, 4),
    "vif": np.round(vif, 2),
}).sort_values("p")
print("\n" + reg.to_string(index=False))
reg.to_csv(OUT / "table_regression.csv", index=False)

# Assumption diagnostics
resid = m["resid"]
sw_w, sw_p = stats.shapiro(resid)
# Breusch--Pagan auxiliary regression: squared residuals on all original
# regressors. The reference degrees of freedom equal the number of slopes.
bp_r2 = ols(Xz, resid ** 2)["r2"]
bp_stat = len(resid) * bp_r2
print(f"\nDiagnostics: Shapiro-Wilk W = {sw_w:.3f}, p = {sw_p:.3f} (residual normality)")
print(f"             Breusch-Pagan LM = {bp_stat:.2f}, "
      f"p = {stats.chi2.sf(bp_stat, len(pred_names)):.3f} (homoscedasticity)")
print(f"             max VIF = {max(vif):.2f} (multicollinearity; threshold 3)")

# A priori global-model power for a conventional medium effect (f2=.15), plus
# the effect size required for 80% power at this N and predictor count.
fcrit = stats.f.ppf(.95, m["df1"], m["df2"])
power_medium = float(stats.ncf.sf(fcrit, m["df1"], m["df2"], .15 * m["n"]))
lo_f2, hi_f2 = 0.0, 2.0
for _ in range(80):
    mid_f2 = (lo_f2 + hi_f2) / 2
    pow_mid = stats.ncf.sf(fcrit, m["df1"], m["df2"], mid_f2 * m["n"])
    if pow_mid < .80:
        lo_f2 = mid_f2
    else:
        hi_f2 = mid_f2
detectable_f2_80 = (lo_f2 + hi_f2) / 2
print(f"             achieved design sensitivity for f2=.15 = {power_medium:.3f}; "
      f"80% power begins at f2={detectable_f2_80:.3f}")

RESULTS["regression"] = {
    "n": int(m["n"]), "r2": round(float(m["r2"]), 3),
    "adj_r2": round(float(m["adj"]), 3), "f": round(float(m["f"]), 2),
    "df1": int(m["df1"]), "df2": int(m["df2"]),
    "p": float(stats.f.sf(m["f"], m["df1"], m["df2"])),
    "n_bootstrap": N_BOOT,
    "shapiro_w": round(float(sw_w), 3), "shapiro_p": round(float(sw_p), 4),
    "breusch_pagan_lm": round(float(bp_stat), 2),
    "breusch_pagan_df": len(pred_names),
    "breusch_pagan_p": round(float(stats.chi2.sf(bp_stat, len(pred_names))), 4),
    "max_vif": round(float(max(vif)), 2),
    "power_f2_015": round(power_medium, 3),
    "detectable_f2_80": round(float(detectable_f2_80), 3),
    "coefficients": reg.to_dict("records"),
}

# Ordinal-logit sensitivity for the single 1--5 outcome. Coefficients are on
# the cumulative-logit scale and are not presented as standardized OLS betas.
head("6a. ORDERED-LOGIT SENSITIVITY (single ordinal outcome)")
ord_fit = OrderedModel(D["_y"].astype(int),
                       pd.DataFrame(Xz, columns=pred_names, index=D.index),
                       distr="logit").fit(method="bfgs", disp=False)
ord_rows = []
for name in pred_names:
    est = float(ord_fit.params[name])
    se_o = float(ord_fit.bse[name])
    p_o = float(ord_fit.pvalues[name])
    ord_rows.append({"predictor": name, "logit_b": round(est, 3),
                     "se": round(se_o, 3), "p": round(p_o, 4),
                     "ci_low": round(est - 1.96 * se_o, 3),
                     "ci_high": round(est + 1.96 * se_o, 3)})
ord_table = pd.DataFrame(ord_rows).sort_values("p")
print(ord_table.to_string(index=False))
ord_table.to_csv(OUT / "table_ordered_logit_sensitivity.csv", index=False)
RESULTS["ordinal_sensitivity"] = {
    "model": "proportional-odds ordered logit",
    "coefficients": ord_table.to_dict("records"),
}

# Hierarchical version: demographics first, then subscales
head("6b. HIERARCHICAL REGRESSION (Block 1 demographics -> Block 2 subscales)")
blk1 = ["Academic semester"]
blk2 = [c for c in pred_names if c != "Academic semester"]
i1 = [pred_names.index(c) for c in blk1]
m1 = ols(Xz[:, i1], yz)
print(f"Block 1 (semester)         : R2 = {m1['r2']:.3f}, "
      f"F({m1['df1']}, {m1['df2']}) = {m1['f']:.2f}, "
      f"p = {stats.f.sf(m1['f'], m1['df1'], m1['df2']):.4f}")
print(f"Block 2 (+ 7 subscales)    : R2 = {m['r2']:.3f}, adj R2 = {m['adj']:.3f}")
dr2 = m["r2"] - m1["r2"]
df_ch = len(blk2)
f_ch = (dr2 / df_ch) / ((1 - m["r2"]) / m["df2"])
print(f"Change                     : dR2 = {dr2:.3f}, "
      f"F({df_ch}, {m['df2']}) = {f_ch:.2f}, "
      f"p = {stats.f.sf(f_ch, df_ch, m['df2']):.3g}")
RESULTS["hierarchical"] = {
    "block1_r2": round(float(m1["r2"]), 3),
    "block2_r2": round(float(m["r2"]), 3),
    "delta_r2": round(float(dr2), 3),
    "f_change": round(float(f_ch), 2),
    "p_change": float(stats.f.sf(f_ch, df_ch, m["df2"])),
}

# --------------------------------------------------------------------------- #
# 7. Gender comparisons
# --------------------------------------------------------------------------- #
head("7. GENDER COMPARISONS (independent-samples t-tests)")

is_f = gender == "Femenino"
rows = []
for label, series in list(S.items()) + [("Adoption intention", y),
                                        ("Dehumanization (single item)", dehum)]:
    a, b = series[is_f], series[~is_f]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    welch_df = (va / len(a) + vb / len(b)) ** 2 / (
        (va / len(a)) ** 2 / (len(a) - 1) +
        (vb / len(b)) ** 2 / (len(b) - 1))
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                 / (len(a) + len(b) - 2))
    rows.append({"variable": label,
                 "M_female": round(float(a.mean()), 2), "SD_female": round(float(a.std(ddof=1)), 2),
                 "M_male": round(float(b.mean()), 2), "SD_male": round(float(b.std(ddof=1)), 2),
                 "t": round(float(t), 2), "df": round(float(welch_df), 1),
                 "p": round(float(p), 3),
                 "cohens_d": round(float((a.mean() - b.mean()) / sp), 2)})
gt = pd.DataFrame(rows)
print(gt.to_string(index=False))
gt.to_csv(OUT / "table_gender.csv", index=False)
RESULTS["gender_tests"] = gt.to_dict("records")

# --------------------------------------------------------------------------- #
# 8. Cluster analysis
# --------------------------------------------------------------------------- #
head("8. CLUSTER ANALYSIS (exploratory, descriptive)")

Sz = StandardScaler().fit_transform(S.values)
print("k-means solutions compared (as requested by the pre-submission review):")
comp = []
for k in range(2, 6):
    lab = KMeans(n_clusters=k, n_init=50, random_state=SEED).fit_predict(Sz)
    sil = silhouette_score(Sz, lab)
    f, p = stats.f_oneway(*[y[lab == c] for c in range(k)])
    comp.append({"k": k, "silhouette": round(float(sil), 3),
                 "sizes": [int((lab == c).sum()) for c in range(k)],
                 "anova_F": round(float(f), 2), "anova_p": float(p)})
    print(f"   k = {k}: silhouette = {sil:.3f}, sizes = {comp[-1]['sizes']}, "
          f"ANOVA F = {f:.2f}, p = {p:.3g}")
pd.DataFrame(comp).to_csv(OUT / "table_cluster_comparison.csv", index=False)
RESULTS["cluster_comparison"] = comp

K = 3
labels = KMeans(n_clusters=K, n_init=50, random_state=SEED).fit_predict(Sz)
sil_k = silhouette_score(Sz, labels)

# Order clusters by mean adoption intention (low -> high) for stable numbering
order = np.argsort([y[labels == c].mean() for c in range(K)])
remap = {old: new for new, old in enumerate(order)}
labels = np.array([remap[l] for l in labels])

# Labels are derived from the observed profile, not from intention alone.
# Cluster 1 scores low on every subscale, including ethical concern -> low salience.
# Cluster 2 combines the lowest AI literacy and institutional support with the
#   highest governance demand and perceived barriers -> concerned but underprepared.
# Cluster 3 is high on value, literacy and self-efficacy, yet its ethical concern
#   is indistinguishable from Cluster 2 -> confident adopters, equally concerned.
NAMES = ["Disengaged", "Concerned and Underprepared", "Confident Adopters"]

print(f"\nRetained solution: k = {K}, silhouette = {sil_k:.3f}")
print("NOTE: a silhouette of this magnitude indicates weak separation. The three")
print("      profiles are reported as a descriptive heuristic for differentiated")
print("      professional development, NOT as validated latent types.")

prof_rows = []
for c in range(K):
    msk = labels == c
    row = {"cluster": c + 1, "label": NAMES[c], "n": int(msk.sum()),
           "pct": round(float(msk.mean() * 100), 1),
           "intention_M": round(float(y[msk].mean()), 2),
           "intention_SD": round(float(y[msk].std(ddof=1)), 2)}
    for col in S.columns:
        row[col] = round(float(S.loc[msk, col].mean()), 2)
    prof_rows.append(row)
prof = pd.DataFrame(prof_rows)
print("\n" + prof.to_string(index=False))
prof.to_csv(OUT / "table_cluster_profiles.csv", index=False)

f_cl, p_cl = stats.f_oneway(*[y[labels == c] for c in range(K)])
print(f"\nANOVA on adoption intention: F({K-1}, {len(y)-K}) = {f_cl:.2f}, p = {p_cl:.3g}")

print("\nTukey HSD post-hoc (pairwise):")
mse_w = sum(((y[labels == c] - y[labels == c].mean()) ** 2).sum()
            for c in range(K)) / (len(y) - K)
tukey = []
for i in range(K):
    for j in range(i + 1, K):
        ni, nj = (labels == i).sum(), (labels == j).sum()
        diff = y[labels == i].mean() - y[labels == j].mean()
        se_d = np.sqrt(mse_w / 2 * (1 / ni + 1 / nj))
        q = abs(diff) / se_d
        p_t = stats.studentized_range.sf(q, K, len(y) - K)
        tukey.append({"pair": f"{NAMES[i]} vs {NAMES[j]}",
                      "mean_diff": round(float(diff), 2),
                      "q": round(float(q), 2), "p": round(float(p_t), 4)})
        print(f"   {NAMES[i]:<22} vs {NAMES[j]:<22} "
              f"diff = {diff:+.2f}, q = {q:.2f}, p = {p_t:.4f}")
pd.DataFrame(tukey).to_csv(OUT / "table_tukey.csv", index=False)

RESULTS["clusters"] = {"k": K, "silhouette": round(float(sil_k), 3),
                       "anova_F": round(float(f_cl), 2), "anova_p": float(p_cl),
                       "profiles": prof.to_dict("records"), "tukey": tukey}

np.savetxt(OUT / "cluster_membership.csv", labels + 1, fmt="%d",
           header="cluster", comments="")

# --------------------------------------------------------------------------- #
# 9. Supplementary: exploratory PCA (descriptive only)
# --------------------------------------------------------------------------- #
head("9. SUPPLEMENTARY — EXPLORATORY COMPONENT ANALYSIS (descriptive only)")

Xp = X.drop(columns=[outcome_col])
print("Reported for transparency. NOT used for inference: the outcome item is")
print("excluded from the factor pool. The participant-to-item ratio is")
print(f"{len(Xp)}:{Xp.shape[1]} = {len(Xp)/Xp.shape[1]:.2f}:1, well below the")
print(f"5:1 minimum, which would require {5*Xp.shape[1]} respondents.\n")

R = np.corrcoef(Xp.values, rowvar=False)
n_obs, p_var = Xp.shape

chi2 = -((n_obs - 1) - (2 * p_var + 5) / 6) * np.log(np.linalg.det(R + np.eye(p_var) * 1e-10))
df_b = p_var * (p_var - 1) / 2
print(f"Bartlett's test of sphericity: chi2 = {chi2:.1f}, df = {df_b:.0f}, "
      f"p = {stats.chi2.sf(chi2, df_b):.3g}")

Rinv = np.linalg.pinv(R)
P = -Rinv / np.sqrt(np.outer(np.diag(Rinv), np.diag(Rinv)))
np.fill_diagonal(P, 0)
Rc = R.copy(); np.fill_diagonal(Rc, 0)
kmo_total = (Rc ** 2).sum() / ((Rc ** 2).sum() + (P ** 2).sum())
msa = (Rc ** 2).sum(axis=0) / ((Rc ** 2).sum(axis=0) + (P ** 2).sum(axis=0))
print(f"KMO measure of sampling adequacy: {kmo_total:.3f} "
      f"(item-level MSA range {msa.min():.2f}-{msa.max():.2f}, "
      f"{int((msa < 0.5).sum())} items below 0.50)")

Z = StandardScaler().fit_transform(Xp.values)
pca_all = PCA().fit(Z)
ev = pca_all.explained_variance_
print(f"Components with eigenvalue > 1 (Kaiser): {int((ev > 1).sum())} "
      f"— NOT 8, as previously reported")

# Horn's parallel analysis
n_pa = 500
rand_ev = np.empty((n_pa, min(n_obs, p_var)))
for i in range(n_pa):
    Zr = rng.standard_normal((n_obs, p_var))
    rand_ev[i] = PCA().fit(StandardScaler().fit_transform(Zr)).explained_variance_
pa_thresh = np.percentile(rand_ev, 95, axis=0)
n_pa_keep = int((ev > pa_thresh).sum())
print(f"Horn's parallel analysis (95th pct, {n_pa} sims) retains: {n_pa_keep} components")

def varimax(Phi, gamma=1.0, q=100, tol=1e-6):
    p_, k_ = Phi.shape
    Rot = np.eye(k_); d = 0
    for _ in range(q):
        d_old = d
        L = Phi @ Rot
        u, s, vh = np.linalg.svd(
            Phi.T @ (L ** 3 - (gamma / p_) * L @ np.diag(np.diag(L.T @ L))))
        Rot = u @ vh; d = s.sum()
        if d_old != 0 and d / d_old < 1 + tol:
            break
    return Phi @ Rot

pca8 = PCA(n_components=8).fit(Z)
loadings = pca8.components_.T * np.sqrt(pca8.explained_variance_)
rot = varimax(loadings)
ss = (rot ** 2).sum(axis=0)
idx = np.argsort(-ss)
rot, ss = rot[:, idx], ss[idx]

print(f"\nUnrotated variance (%): "
      f"{np.round(pca_all.explained_variance_ratio_[:8]*100, 1).tolist()}")
print(f"Varimax-rotated (%)   : {np.round(ss/p_var*100, 1).tolist()}")
print(f"Cumulative (8 comps)  : {ss.sum()/p_var*100:.1f}%")
below = int((np.abs(rot).max(axis=1) < 0.40).sum())
print(f"Items with no loading >= 0.40: {below} of {p_var} "
      f"— the earlier claim that all items exceeded 0.40 was incorrect")

pd.DataFrame(rot, index=[c.strip() for c in Xp.columns],
             columns=[f"PC{i+1}" for i in range(8)]).round(3) \
    .to_csv(OUT / "supp_rotated_loadings.csv")
pd.DataFrame({"component": range(1, len(ev) + 1), "eigenvalue": np.round(ev, 3),
              "pct_variance": np.round(pca_all.explained_variance_ratio_ * 100, 2),
              "parallel_95pct": np.round(pa_thresh, 3)}) \
    .to_csv(OUT / "supp_eigenvalues.csv", index=False)
pd.DataFrame({"item": [c.strip() for c in Xp.columns], "MSA": np.round(msa, 3)}) \
    .to_csv(OUT / "supp_item_msa.csv", index=False)

RESULTS["supplementary_pca"] = {
    "n_items": int(Xp.shape[1]),
    "ratio": f"{len(Xp)}:{Xp.shape[1]}",
    "ratio_numeric": round(len(Xp) / Xp.shape[1], 2),
    "n_required_5to1": 5 * Xp.shape[1],
    "kmo": round(float(kmo_total), 3),
    "msa_min": round(float(msa.min()), 3), "msa_max": round(float(msa.max()), 3),
    "bartlett_chi2": round(float(chi2), 1), "bartlett_df": int(df_b),
    "bartlett_p": float(stats.chi2.sf(chi2, df_b)),
    "kaiser_components": int((ev > 1).sum()),
    "parallel_analysis_components": n_pa_keep,
    "unrotated_pct": np.round(pca_all.explained_variance_ratio_[:8] * 100, 1).tolist(),
    "varimax_pct": np.round(ss / p_var * 100, 1).tolist(),
    "cumulative_pct": round(float(ss.sum() / p_var * 100), 1),
    "items_below_040": below,
}

# --------------------------------------------------------------------------- #
# 10. De-identified adult dataset for controlled verification
# --------------------------------------------------------------------------- #
head("10. DE-IDENTIFIED ADULT DATASET FOR CONTROLLED VERIFICATION")

drop_identifiers = [c for c in df.columns if c.casefold() in
                    {"marca temporal", "nombre de usuario", "participant_id"}]
anon = df.drop(columns=drop_identifiers).copy()
anon.insert(0, "participant_id", [f"P{i:03d}" for i in range(1, len(anon) + 1)])
anon["cluster"] = labels + 1
anon.to_csv(OUT / "dataset_anonymised.csv", index=False, encoding="utf-8")
print(f"Written: dataset_anonymised.csv ({len(anon)} rows x {anon.shape[1]} cols)")
print(f"Removed identifying columns: {drop_identifiers}")
RESULTS["anonymised_dataset"] = {
    "rows": int(len(anon)), "cols": int(anon.shape[1]),
    "removed_columns": drop_identifiers,
}

# --------------------------------------------------------------------------- #
with open(OUT / "results.json", "w", encoding="utf-8") as fh:
    json.dump(RESULTS, fh, indent=2, ensure_ascii=False)

head("DONE")
print(f"All outputs written to: {OUT}")
for f in sorted(OUT.iterdir()):
    print(f"   {f.name}")

sys.stdout.flush()
sys.stdout = sys.__stdout__
_log.close()
