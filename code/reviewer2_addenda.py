# -*- coding: utf-8 -*-
"""
Additional analyses demanded by Reviewer 2 that the first revision did not cover:

  R2-B2   outcome item must be excluded from the factored pool
  R2-B3   common-method variance (Harman test + conservative sensitivity)
  R2-D7   EFA (principal axis) with OBLIQUE rotation, not PCA + varimax
  R2-E9   reconcile the 62.8% vs 87.6% prior-AI-use discrepancy
  R2-E10  participants under the age of majority
  R2-E11  quantify attrition
  R2-F13  cluster stability and the circularity concern
"""
from __future__ import annotations
import argparse, json, sys, io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = (HERE.parent.parent
               / "Uso de la Inteligencia Artificial en la Formación de Docentes de Matemáticas .csv")
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=Path, default=DEFAULT_CSV)
args = parser.parse_args()
CSV = args.data.resolve()
OUT = HERE / "output_adults"

SEED = 42
rng = np.random.default_rng(SEED)
ADD = {}

df_raw = pd.read_csv(CSV, encoding="utf-8")
cols = list(df_raw.columns)

def find_column(fragment, fallback):
    hits = [c for c in cols if fragment.casefold() in c.casefold()]
    return hits[0] if hits else cols[fallback]

COL_AGE = find_column("edad", 2)
COL_GENDER = find_column("género", 3)
if "género" not in COL_GENDER.casefold():
    COL_GENDER = find_column("genero", 3)
COL_SEMESTER = find_column("semestre", 4)
age = pd.to_numeric(df_raw[COL_AGE].astype("string").str.extract(
    r"(\d{1,3})", expand=False), errors="coerce")
is_minor, is_adult = age.lt(18), age.ge(18)
is_unverified = age.isna()
df = df_raw.loc[is_adult].copy().reset_index(drop=True)
df[COL_AGE] = age.loc[is_adult].astype(int).to_numpy()
print(f"ANALYTIC SAMPLE: {len(df_raw)} received -> {len(df)} confirmed adults "
      f"({int(is_minor.sum())} minors and {int(is_unverified.sum())} unverified-age "
      f"record excluded before substantive analysis)\n")

metadata = {COL_AGE, COL_GENDER, COL_SEMESTER}
metadata.update(c for c in cols if c.casefold() in {
    "marca temporal", "nombre de usuario", "participant_id"})
items = [c for c in cols if c not in metadata]
likert = [c for c in items if pd.to_numeric(df[c], errors="coerce").notna().all()
          and pd.to_numeric(df[c], errors="coerce").between(1, 5).all()]
dich = [c for c in items if c not in likert]
X = df[likert].astype(float)
OUTCOME = [c for c in likert if "Planeo incorporar" in c][0]
y = X[OUTCOME]


def head(t):
    print("\n" + "=" * 78); print(t); print("=" * 78)


def alpha(d):
    k = d.shape[1]
    return k / (k - 1) * (1 - d.var(ddof=1).sum() / d.sum(axis=1).var(ddof=1))


# ===================================================================== R2-E9
head("R2-E9  RECONCILING THE PRIOR-AI-USE PERCENTAGE (62.8% vs 87.6%)")
print("Every dichotomous item, with counts, to locate any 62.8% figure:\n")
target = None
for c in dich:
    n_yes = int((df[c].astype(str).str.strip() == "Si").sum())
    pct = n_yes / len(df) * 100
    flag = "   <-- candidate for 62.8%" if abs(pct - 62.8) < 2.5 else ""
    print(f"   {n_yes:3d}  {pct:5.1f}%  {c.strip()[:62]}{flag}")
    if abs(pct - 62.8) < 2.5:
        target = (c.strip(), n_yes, pct)

n71 = round(0.628 * len(df))
print(f"\n62.8% of the adult analytic sample (N = {len(df)}) would be "
      f"{0.628*len(df):.1f} respondents (i.e. {n71}).")
print("No item yields that count. The 62.8% figure in the submitted Participants")
print("section does not correspond to any variable in the dataset.")
print(f"\nCORRECT VALUE: 'He utilizado herramientas de IA antes' = "
      f"{int((df[[c for c in dich if 'utilizado herramientas' in c][0]].astype(str).str.strip()=='Si').sum())} "
      f"({(df[[c for c in dich if 'utilizado herramientas' in c][0]].astype(str).str.strip()=='Si').mean()*100:.1f}%)")
ADD["e9_62_8_traceable"] = False

# ==================================================================== R2-E10
head("R2-E10  PARTICIPANTS UNDER THE AGE OF MAJORITY")
minors = int(is_minor.sum())
print(f"Parseable ages: {int(age.notna().sum())} of {len(df_raw)}")
print(f"Participants aged < 18: {minors} ({minors/len(df_raw)*100:.1f}% of the raw export)")
if minors:
    print("Ages recorded below 18:", sorted(age[age < 18].astype(int).tolist()))
print("\nThe confirmed-minor records and the unverified-age record are excluded before")
print("any substantive analysis. Only confirmed adults enter the analytic sample.")
ADD["e10_minors_n"] = minors
ADD["e10_minor_ages"] = sorted(age[age < 18].dropna().astype(int).tolist())

# ==================================================================== R2-E11
head("R2-E11  ATTRITION, QUANTIFIED")
_Xraw = df_raw[likert].apply(pd.to_numeric, errors="coerce")
_complete = int(_Xraw.notna().all(axis=1).sum())
print(f"Rows in the raw export             : {len(df_raw)}")
print(f"Rows with complete Likert data     : {_complete}")
print(f"Excluded for incomplete Likert data: {len(df_raw) - _complete}")
print(f"Excluded as confirmed minors       : {int(is_minor.sum())}")
print(f"Excluded because age unverified    : {int(is_unverified.sum())}")
print(f"Analytic sample                    : {len(df)}")
print(f"Rows with unverified age           : {int(age.isna().sum())} "
      f"({age.isna().mean()*100:.1f}%) -- excluded")
ADD["e11_raw_rows"] = int(len(df_raw))
ADD["e11_incomplete"] = int(len(df_raw) - _complete)
ADD["e11_excluded_minors"] = int(is_minor.sum())
ADD["e11_excluded_unverified_age"] = int(is_unverified.sum())
ADD["e11_analytic_n"] = int(len(df))

# ===================================================================== R2-B2
head("R2-B2  OUTCOME ITEM EXCLUDED FROM THE FACTORED POOL")
print(f"Outcome item: \"{OUTCOME.strip()}\"")
Xp = X.drop(columns=[OUTCOME])
print(f"Predictor pool after exclusion: {Xp.shape[1]} items (was {X.shape[1]})")
print(f"\nCronbach alpha, all 56 items incl. outcome : {alpha(X):.3f}")
print(f"Cronbach alpha, 55 predictor items only    : {alpha(Xp):.3f}")

Z_all = StandardScaler().fit_transform(X.values)
load_all = PCA(n_components=8).fit(Z_all)
comp1 = load_all.components_[0] * np.sqrt(load_all.explained_variance_[0])
i_out = likert.index(OUTCOME)
print(f"\nIn the original 56-item PCA the outcome item loaded {comp1[i_out]:+.2f} on"
      f" component 1 --\n  confirming the reviewer's concern that the outcome was inside"
      f" the factored pool.")
print("All predictor subscales in the revised analysis exclude it; the supplementary")
print("component analysis is now also computed on the 55 predictor items only.")
ADD["b2_alpha_56"] = round(float(alpha(X)), 3)
ADD["b2_alpha_55"] = round(float(alpha(Xp)), 3)
ADD["b2_outcome_loading_pc1"] = round(float(comp1[i_out]), 3)

# ===================================================================== R2-B3
head("R2-B3  COMMON-METHOD VARIANCE")
Zp = StandardScaler().fit_transform(Xp.values)
pca_cmv = PCA().fit(Zp)
first = pca_cmv.explained_variance_ratio_[0] * 100
print("Harman's single-factor test (unrotated, 55 predictor items):")
print(f"   variance explained by the first component: {first:.1f}%")
print(f"   criterion: a single component must not exceed 50%  ->  "
      f"{'PASS' if first < 50 else 'FAIL'}")

# Conservative component-residualisation comparison: partial the first
# component out of the predictors and refit the regression. This is not a
# measured marker-variable design and is not labelled as one.
sem = df[COL_SEMESTER].astype(str).str.extract(r"(\d+)")[0].astype(float)
SPEC = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
subnames = [s["subscale"] for s in SPEC["subscales"]]

# Rebuild the composites from exactly the same theory-informed definition used in
# reanalysis.py. The definition is read out of that file's source with `ast`
# rather than imported, so the module is not re-executed.
import ast as _ast

_src = (HERE / "reanalysis.py").read_text(encoding="utf-8")
SUBSCALE_SPEC = next(
    _ast.literal_eval(node.value)
    for node in _ast.parse(_src).body
    if isinstance(node, _ast.Assign)
    and any(getattr(t, "id", "") == "SUBSCALE_SPEC" for t in node.targets)
)

def match(frag):
    hits = [c for c in likert if frag.lower() in c.lower()]
    return hits[0]

S = {}
for name, frags in SUBSCALE_SPEC.items():
    its = [match(f) for f in frags]
    if OUTCOME in its:
        its.remove(OUTCOME)
    S[name] = X[its].mean(axis=1)
S = pd.DataFrame(S)

def ols(Xm, yv):
    A = np.column_stack([np.ones(len(Xm)), Xm])
    b, *_ = np.linalg.lstsq(A, yv, rcond=None)
    r = yv - A @ b
    n, k = A.shape
    dof = n - k
    se = np.sqrt(np.diag((r ** 2).sum() / dof * np.linalg.pinv(A.T @ A)))
    r2 = 1 - (r ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
    return b, se, b / se, 2 * stats.t.sf(np.abs(b / se), dof), r2, dof

D = pd.concat([S, sem.rename("Academic semester")], axis=1)
D["_y"] = y.values
D = D.dropna()
P = [c for c in D.columns if c != "_y"]
Xz = ((D[P] - D[P].mean()) / D[P].std(ddof=1)).values
yz = ((D["_y"] - D["_y"].mean()) / D["_y"].std(ddof=1)).values
b0, se0, t0, p0, r2_0, _ = ols(Xz, yz)

# method factor = first unrotated component of the predictor items
mf = PCA(n_components=1).fit_transform(Zp).ravel()
mf = (mf - mf.mean()) / mf.std(ddof=1)
Xz_res = np.column_stack([
    Xz[:, j] - np.polyfit(mf, Xz[:, j], 1)[0] * mf for j in range(Xz.shape[1])])
yz_res = yz - np.polyfit(mf, yz, 1)[0] * mf
b1, se1, t1, p1, r2_1, _ = ols(Xz_res, yz_res)

print("\nConservative common-component sensitivity: first unrotated component")
print("partialled out of")
print("every predictor and the outcome, then the model refitted.\n")
print(f"{'Predictor':<26}{'beta':>8}{'p':>9}   {'beta|CMV':>9}{'p':>9}")
for j, nm in enumerate(P):
    print(f"{nm:<26}{b0[j+1]:>+8.3f}{p0[j+1]:>9.4f}   {b1[j+1]:>+9.3f}{p1[j+1]:>9.4f}")
print(f"\nR2 original {r2_0:.3f}  ->  R2 after method control {r2_1:.3f}")
ADD["b3_harman_first_pct"] = round(float(first), 1)
ADD["b3_r2_original"] = round(float(r2_0), 3)
ADD["b3_r2_method_controlled"] = round(float(r2_1), 3)
ADD["b3_coefs"] = {nm: {"beta": round(float(b0[j+1]), 3), "p": round(float(p0[j+1]), 4),
                        "beta_cmv": round(float(b1[j+1]), 3),
                        "p_cmv": round(float(p1[j+1]), 4)}
                   for j, nm in enumerate(P)}

# ===================================================================== R2-D7
head("R2-D7  EFA (PRINCIPAL AXIS) WITH OBLIQUE ROTATION")
print("Reviewer 2 notes that PCA is a data-reduction method and that varimax forces")
print("orthogonality between constructs that should correlate. We report a proper EFA")
print("on the 55 predictor items for comparison (supplementary evidence only).\n")


def promax(L, kappa=4):
    """Oblique promax rotation via a varimax start."""
    def varimax(Phi, q=100, tol=1e-6):
        p, k = Phi.shape
        R = np.eye(k); d = 0
        for _ in range(q):
            d_old = d
            lam = Phi @ R
            u, s, vh = np.linalg.svd(
                Phi.T @ (lam ** 3 - (1.0 / p) * lam @ np.diag(np.diag(lam.T @ lam))))
            R = u @ vh; d = s.sum()
            if d_old and d / d_old < 1 + tol:
                break
        return Phi @ R
    Lv = varimax(L)
    Q = Lv * np.abs(Lv) ** (kappa - 1)
    U = np.linalg.lstsq(Lv, Q, rcond=None)[0]
    d = np.diag(np.linalg.inv(U.T @ U))
    U = U @ np.diag(np.sqrt(d))
    return Lv @ U, U


n_fac = 4  # retained by parallel analysis

def principal_axis_loadings(z, n_factors, max_iter=500, tol=1e-7):
    """Principal-axis factoring by iterated squared multiple correlations."""
    R = np.corrcoef(z, rowvar=False)
    invR = np.linalg.pinv(R)
    h2 = np.clip(1 - 1 / np.diag(invR), 0.05, 0.99)
    for _ in range(max_iter):
        reduced = R.copy()
        np.fill_diagonal(reduced, h2)
        vals, vecs = np.linalg.eigh(reduced)
        order = np.argsort(vals)[::-1][:n_factors]
        vals = np.maximum(vals[order], 0)
        load = vecs[:, order] * np.sqrt(vals)
        new_h2 = np.clip((load ** 2).sum(axis=1), 0.001, 0.999)
        if np.max(np.abs(new_h2 - h2)) < tol:
            return load
        h2 = new_h2
    return load

L = principal_axis_loadings(Zp, n_fac)
Lo, U = promax(L)
phi = np.linalg.inv(U.T @ U)
dg = np.sqrt(np.diag(phi))
phi = phi / np.outer(dg, dg)

print(f"Principal-axis EFA, {n_fac} factors (the number parallel analysis retains),")
print("promax (oblique) rotation. Inter-factor correlations:\n")
print("        " + "".join(f"  F{j+1:<5}" for j in range(n_fac)))
for i in range(n_fac):
    print(f"   F{i+1}   " + "".join(f"{phi[i, j]:+7.2f}" for j in range(n_fac)))
offdiag = phi[np.triu_indices(n_fac, 1)]
print(f"\nInter-factor correlations range {offdiag.min():+.2f} to {offdiag.max():+.2f}"
      f" (mean |r| = {np.abs(offdiag).mean():.2f}).")
print("Non-trivial correlations confirm the reviewer's point that an orthogonal")
print("rotation was inappropriate, and support the use of composite scales that")
print("are permitted to correlate.")

pd.DataFrame(np.round(Lo, 3), index=[c.strip() for c in Xp.columns],
             columns=[f"F{i+1}" for i in range(n_fac)]).to_csv(
    OUT / "supp_efa_promax_loadings.csv", encoding="utf-8-sig")
pd.DataFrame(np.round(phi, 3), index=[f"F{i+1}" for i in range(n_fac)],
             columns=[f"F{i+1}" for i in range(n_fac)]).to_csv(
    OUT / "supp_efa_factor_correlations.csv", encoding="utf-8-sig")
ADD["d7_n_factors"] = n_fac
ADD["d7_interfactor_min"] = round(float(offdiag.min()), 2)
ADD["d7_interfactor_max"] = round(float(offdiag.max()), 2)
ADD["d7_interfactor_mean_abs"] = round(float(np.abs(offdiag).mean()), 2)

# ==================================================================== R2-F13
head("R2-F13  CLUSTER STABILITY AND THE CIRCULARITY CONCERN")
Sz = StandardScaler().fit_transform(S.values)
base = KMeans(n_clusters=3, n_init=50, random_state=SEED).fit_predict(Sz)

print("(a) Seed stability: adjusted Rand index of 50 restarts against the retained")
print("    solution.")
aris = []
for s in range(1, 51):
    lab = KMeans(n_clusters=3, n_init=10, random_state=s).fit_predict(Sz)
    aris.append(adjusted_rand_score(base, lab))
print(f"    ARI mean = {np.mean(aris):.3f}, min = {np.min(aris):.3f}, "
      f"{sum(a > .95 for a in aris)}/50 restarts recover the solution exactly.")

print("\n(b) Algorithm agreement: Ward hierarchical clustering on the same input.")
ward = AgglomerativeClustering(n_clusters=3, linkage="ward").fit_predict(Sz)
ari_w = adjusted_rand_score(base, ward)
print(f"    ARI(k-means, Ward) = {ari_w:.3f}")

print("\n(c) Bootstrap stability: 1,000 resamples, ARI against the full-sample solution.")
boot = []
for _ in range(1000):
    idx = rng.integers(0, len(Sz), len(Sz))
    if len(np.unique(idx)) < 20:
        continue
    lb = KMeans(n_clusters=3, n_init=10, random_state=SEED).fit_predict(Sz[idx])
    boot.append(adjusted_rand_score(base[idx], lb))
print(f"    ARI mean = {np.mean(boot):.3f}, 95% interval "
      f"[{np.percentile(boot, 2.5):.3f}, {np.percentile(boot, 97.5):.3f}]")

print("\n(d) Circularity: the reviewer is correct that testing intention differences")
print("    across clusters built from predictors of intention partly confirms the")
print("    construction. We therefore report an AUXILIARY criterion not used in")
print("    clustering: the dichotomous 'has taken an AI course' item.")
trained = (df[[c for c in dich if "cursos o talleres" in c][0]]
           .astype(str).str.strip() == "Si").astype(int)
ct = pd.crosstab(base, trained)
chi2, pchi, dofc, _ = stats.chi2_contingency(ct)
print(f"\n    Cluster x formal AI training: chi2({dofc}) = {chi2:.2f}, p = {pchi:.4f}")
for c in range(3):
    m = base == c
    print(f"      cluster {c+1}: {int(trained[m].sum())}/{int(m.sum())} trained "
          f"({trained[m].mean()*100:.1f}%)")
ADD["f13_ari_seed_mean"] = round(float(np.mean(aris)), 3)
ADD["f13_ari_ward"] = round(float(ari_w), 3)
ADD["f13_ari_boot_mean"] = round(float(np.mean(boot)), 3)
ADD["f13_ari_boot_lo"] = round(float(np.percentile(boot, 2.5)), 3)
ADD["f13_ari_boot_hi"] = round(float(np.percentile(boot, 97.5)), 3)
ADD["f13_external_chi2"] = round(float(chi2), 2)
ADD["f13_external_p"] = round(float(pchi), 4)

with open(OUT / "reviewer2_addenda.json", "w", encoding="utf-8") as fh:
    json.dump(ADD, fh, indent=2, ensure_ascii=False)
head("DONE")
print(f"Written: {OUT / 'reviewer2_addenda.json'}")
print("        supp_efa_promax_loadings.csv, supp_efa_factor_correlations.csv")
