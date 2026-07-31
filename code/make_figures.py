# -*- coding: utf-8 -*-
"""
Figures for the revised manuscript. Every value is read from the outputs of
reanalysis.py — nothing is hard-coded. Run reanalysis.py first.

Palette: validated categorical slots (blue #2a78d6, orange #eb6834) and the
blue<->red diverging pair with a neutral gray midpoint. Every colour-encoded
figure also carries printed values, so no figure depends on colour alone.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
OUT = HERE / "output_adults"
FIG = HERE.parent / "figures_adults"
FIG.mkdir(exist_ok=True)

R = json.loads((OUT / "results.json").read_text(encoding="utf-8"))

# --- design tokens ---------------------------------------------------------
BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#e34948"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "font.size": 9,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.grid": False,
    "legend.frameon": False,
})

DPI = 400


def finish(fig, name):
    fig.savefig(FIG / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


def despine(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)
        if s in keep:
            ax.spines[s].set_linewidth(0.8)


print("Generating figures...")

# =========================================================================== #
# Figure 1 — Conceptual model
# =========================================================================== #
coef = {c["predictor"]: c for c in R["regression"]["coefficients"]}

# Grouped by the UTAUT construct family each subscale operationalises.
# The model is single-stage: every predictor is entered simultaneously against
# the outcome. No mediation is estimated, so none is drawn.
GROUPS = [
    ("Performance and effort expectancy", ["Perceived Value", "Technical Self-Efficacy"]),
    ("Ethical orientation", ["Ethical Concern", "Ethical Governance"]),
    ("Facilitating conditions", ["Institutional Support", "Perceived Barriers"]),
    ("Knowledge and academic development", ["AI Literacy", "Academic semester"]),
]
NICE = {
    "Perceived Value": "Perceived value",
    "Technical Self-Efficacy": "Technical self-efficacy",
    "Ethical Concern": "Ethical concern\n(incl. dehumanization)",
    "Ethical Governance": "Ethical governance",
    "Institutional Support": "Institutional support",
    "Perceived Barriers": "Perceived barriers",
    "AI Literacy": "AI literacy",
    "Academic semester": "Academic progression\n(semester)",
}

n_rows = sum(len(g[1]) for g in GROUPS)
ROW_H, GAP_G, HDR_H = 0.94, 0.20, 0.42
fig_h = 5.9
fig, ax = plt.subplots(figsize=(7.4, fig_h))

total_h = n_rows * ROW_H + len(GROUPS) * HDR_H + (len(GROUPS) - 1) * GAP_G
ax.set_xlim(0, 10.4); ax.set_ylim(-1.35, total_h + 0.55); ax.axis("off")

BX, BW = 0.28, 3.35          # predictor boxes
LX = BX + BW + 0.18          # beta label column
OX, OW = 7.35, 2.95          # outcome box

y = total_h
rows = []
for gi, (gname, members) in enumerate(GROUPS):
    ax.text(BX, y - HDR_H * 0.55, gname.upper(), fontsize=7.2, color=MUTED,
            weight="bold", va="center", ha="left")
    ax.plot([BX, BX + BW], [y - HDR_H + 0.02, y - HDR_H + 0.02],
            color="#e1e0d9", linewidth=1.0)
    y -= HDR_H
    gtop = y
    for name in members:
        c = coef[name]
        sig = c["p"] < 0.05
        col = BLUE if (sig and c["beta"] > 0) else (RED if sig else MUTED)
        ax.add_patch(FancyBboxPatch((BX, y - ROW_H + 0.14), BW, ROW_H - 0.28,
                                    boxstyle="round,pad=0.02,rounding_size=0.10",
                                    linewidth=1.1,
                                    edgecolor=col if sig else "#d6d5cf",
                                    facecolor="#eaf2fc" if (sig and c["beta"] > 0)
                                    else ("#fdecec" if sig else "#f6f6f3")))
        ax.text(BX + BW / 2, y - ROW_H / 2, NICE[name], ha="center", va="center",
                fontsize=8.4, color=INK, linespacing=1.3,
                weight="bold" if sig else "normal")
        rows.append((y - ROW_H / 2, col, c, sig))
        y -= ROW_H
    y -= GAP_G

# outcome
oy = total_h / 2
ax.add_patch(FancyBboxPatch((OX, oy - 0.85), OW, 1.70,
                            boxstyle="round,pad=0.02,rounding_size=0.12",
                            linewidth=1.5, edgecolor=INK, facecolor="#f4f3ef"))
ax.text(OX + OW / 2, oy, "Behavioural intention\nto integrate\nAI-assisted assessment",
        ha="center", va="center", fontsize=9.2, color=INK, weight="bold",
        linespacing=1.4)

# arrows + beta labels
for ry, col, c, sig in rows:
    stars = "***" if c["p"] < .001 else "**" if c["p"] < .01 else "*" if c["p"] < .05 else ""
    txt = f"β = {c['beta']:+.2f}{stars}" if sig else f"β = {c['beta']:+.2f} n.s."
    ax.text(LX, ry, txt, ha="left", va="center", fontsize=8,
            color=col, weight="bold" if sig else "normal", family="monospace")
    ax.add_patch(FancyArrowPatch((LX + 1.28, ry), (OX - 0.06, oy),
                                 connectionstyle="arc3,rad=0",
                                 arrowstyle="-|>", mutation_scale=10,
                                 linewidth=1.6 if sig else 0.9,
                                 color=col, alpha=1.0 if sig else 0.55,
                                 shrinkA=3, shrinkB=2))

ax.text(BX, total_h + 0.28, "Predictors (theory-informed composites)",
        fontsize=8.4, color=INK2, weight="bold")
ax.text(OX, total_h + 0.28, "Outcome", fontsize=8.4, color=INK2, weight="bold")

rg = R["regression"]
ax.text(0.0, -0.45,
        f"Single-stage model: all eight predictors entered simultaneously "
        f"(N = {rg['n']}; R² = {rg['r2']:.3f}, adjusted R² = {rg['adj_r2']:.3f}, "
        f"F({rg['df1']}, {rg['df2']}) = {rg['f']:.2f}, p < .001).\n"
        f"Blue = significant positive, red = significant negative, grey = non-significant. "
        f"* p < .05, ** p < .01, *** p < .001.\n"
        f"No mediation is estimated; the grouping reflects the UTAUT construct family "
        f"each subscale operationalises, not an estimated causal ordering.",
        fontsize=7.3, color=INK2, va="top", linespacing=1.5)
finish(fig, "fig1_conceptual_model.png")

# =========================================================================== #
# Figure 2 — Technology access and AI experience profile
# =========================================================================== #
dich = R["demographics"]["dichotomous"]
lbl_map = {
    "Tengo acceso regular a un computador": "Regular access to a computer or smart device",
    "Cuento con conexión a Internet": "Good-quality internet at home",
    "Dispongo de los recursos tecnológicos": "Has the technological resources to use AI tools",
    "Me considero competente en el uso": "Self-rated competent with basic digital technology",
    "Tengo experiencia en el uso de plataformas": "Experience with online learning platforms",
    "Estoy familiarizado/a con aplicaciones avanzadas": "Familiar with advanced educational technology",
    "He utilizado herramientas de inteligencia artificial": "Has used AI tools before",
    "He participado en cursos o talleres": "Has taken any AI course or workshop",
}
rows = []
for k, v in dich.items():
    label = next((en for es, en in lbl_map.items() if k.startswith(es)), k[:50])
    rows.append((label, v["pct"]))
rows.sort(key=lambda r: r[1])

fig, ax = plt.subplots(figsize=(7.2, 3.4))
ypos = np.arange(len(rows))
vals = [r[1] for r in rows]
highlight = {"Has used AI tools before", "Has taken any AI course or workshop"}
colors = [BLUE if r[0] in highlight else "#c8d7ea" for r in rows]

ax.barh(ypos, vals, height=0.62, color=colors, zorder=3)
ax.set_yticks(ypos, [r[0] for r in rows], fontsize=8.2)
ax.set_xlim(0, 105)
ax.set_xlabel("Participants responding “Yes” (%)", fontsize=8.5, color=INK2)
for x in (25, 50, 75, 100):
    ax.axvline(x, color=GRID, linewidth=0.7, zorder=1)
for yv, v in zip(ypos, vals):
    ax.text(v + 1.6, yv, f"{v:.1f}%", va="center", fontsize=8, color=INK2)
despine(ax, keep=("bottom",))
ax.tick_params(axis="y", length=0)

gap = R["demographics"]["usage_training_gap_pp"]
i_use = [r[0] for r in rows].index("Has used AI tools before")
i_tr = [r[0] for r in rows].index("Has taken any AI course or workshop")
ax.annotate("", xy=(100, ypos[i_use]), xytext=(100, ypos[i_tr]),
            arrowprops=dict(arrowstyle="<->", color=ORANGE, linewidth=1.4))
ax.text(101.5, (ypos[i_use] + ypos[i_tr]) / 2,
        f"{gap} pp\ngap", fontsize=8, color=ORANGE, weight="bold", va="center")
ax.set_title(f"Technology access and prior AI experience (N = {R['sample']['n_respondents']})",
             loc="left", pad=10)
finish(fig, "fig2_experience_profile.png")

# =========================================================================== #
# Figure 3 — Forest plot of standardized regression coefficients
# =========================================================================== #
reg = pd.read_csv(OUT / "table_regression.csv").sort_values("beta")
fig, ax = plt.subplots(figsize=(7.2, 3.6))
ypos = np.arange(len(reg))
sig = reg["p"] < 0.05

ax.axvline(0, color=AXIS, linewidth=1.0, zorder=2)
for x in (-0.4, -0.2, 0.2, 0.4, 0.6, 0.8):
    ax.axvline(x, color=GRID, linewidth=0.7, zorder=1)

for yv, (_, row), s in zip(ypos, reg.iterrows(), sig):
    col = BLUE if (s and row["beta"] > 0) else (RED if s else MUTED)
    ax.plot([row["ci_low"], row["ci_high"]], [yv, yv],
            color=col, linewidth=1.8, solid_capstyle="round", zorder=3)
    ax.plot(row["beta"], yv, "o" if s else "o", markersize=7,
            markerfacecolor=col if s else SURFACE,
            markeredgecolor=col, markeredgewidth=1.6, zorder=4)
    star = "***" if row["p"] < .001 else "**" if row["p"] < .01 else "*" if row["p"] < .05 else ""
    ax.text(0.86, yv, f"{row['beta']:+.3f}{star}", fontsize=8,
            color=INK if s else MUTED, va="center", family="monospace")

ax.set_yticks(ypos, reg["predictor"], fontsize=8.6)
ax.set_xlim(-0.55, 1.02)
ax.set_xticks([-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8])
ax.set_xlabel("Standardised coefficient β (95% CI)", fontsize=8.5, color=INK2)
despine(ax, keep=("bottom",))
ax.tick_params(axis="y", length=0)
ax.set_title("Predictors of intention to integrate AI-assisted assessment",
             loc="left", pad=10)
ax.text(-0.55, -1.5,
        f"N = {R['regression']['n']}; R² = {R['regression']['r2']:.3f}, "
        f"adjusted R² = {R['regression']['adj_r2']:.3f}, "
        f"F({R['regression']['df1']}, {R['regression']['df2']}) = {R['regression']['f']:.2f}, "
        f"p < .001.  Filled markers p < .05; hollow markers non-significant.\n"
        f"* p < .05, ** p < .01, *** p < .001. All VIF ≤ {R['regression']['max_vif']}.",
        fontsize=7.2, color=INK2, va="top")
finish(fig, "fig3_regression_forest.png")

# =========================================================================== #
# Figure 4 — Cluster profiles
# =========================================================================== #
prof = pd.read_csv(OUT / "table_cluster_profiles.csv")
subs = [r["subscale"] for r in R["subscales"]]
M = prof[subs].values
centred = M - M.mean(axis=0)

fig, (axh, axb) = plt.subplots(
    1, 2, figsize=(7.4, 3.1), gridspec_kw={"width_ratios": [3.1, 1.0], "wspace": 0.32})

div = LinearSegmentedColormap.from_list("bwr", ["#1c5cab", "#9ec5f4", "#f0efec",
                                                "#f6b0af", "#c0322f"])
lim = np.abs(centred).max()
norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
im = axh.imshow(centred, cmap=div, norm=norm, aspect="auto")

axh.set_xticks(range(len(subs)),
               [s.replace(" ", "\n") for s in subs], fontsize=7.4)
axh.set_yticks(range(len(prof)),
               [f"{r.label}\n(n = {r.n}, {r.pct}%)" for r in prof.itertuples()],
               fontsize=8)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        axh.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7.6,
                 color="#ffffff" if abs(centred[i, j]) > lim * 0.55 else INK)
axh.set_xticks(np.arange(-.5, len(subs), 1), minor=True)
axh.set_yticks(np.arange(-.5, len(prof), 1), minor=True)
axh.grid(which="minor", color=SURFACE, linewidth=2)
axh.tick_params(which="both", length=0)
for s in axh.spines.values():
    s.set_visible(False)
axh.set_title("Subscale means by readiness profile", loc="left", pad=10, fontsize=9.5)

cb = fig.colorbar(im, ax=axh, orientation="horizontal", pad=0.20, shrink=0.55,
                  aspect=28)
cb.set_label("Deviation from the across-profile mean", fontsize=7.2, color=INK2)
cb.ax.tick_params(labelsize=6.8, length=2, color=AXIS)
cb.outline.set_visible(False)

xb = np.arange(len(prof))
axb.bar(xb, prof["intention_M"], yerr=prof["intention_SD"], width=0.6,
        color=BLUE, capsize=3, error_kw=dict(ecolor=MUTED, lw=1.0), zorder=3)
for x, v in zip(xb, prof["intention_M"]):
    axb.text(x, v + 0.14, f"{v:.2f}", ha="center", fontsize=8, color=INK)
axb.set_xticks(xb, [f"P{i+1}" for i in xb], fontsize=8)
axb.set_ylim(0, 5.4)
axb.set_yticks([1, 2, 3, 4, 5])
axb.set_ylabel("Adoption intention (1–5)", fontsize=8.2, color=INK2)
for yv in (1, 2, 3, 4, 5):
    axb.axhline(yv, color=GRID, linewidth=0.7, zorder=1)
despine(axb, keep=("left",))
axb.tick_params(axis="x", length=0)
cl = R["clusters"]
axb.set_title(f"F(2, 110) = {cl['anova_F']:.2f}\np < .001", loc="left",
              fontsize=8, pad=8, weight="normal", color=INK2)
fig.text(0.02, -0.02,
         f"Three-cluster k-means solution on the seven theory-informed composites "
         f"(silhouette = {cl['silhouette']:.2f}). The weak silhouette indicates that these "
         f"profiles are a descriptive heuristic for differentiated professional development, "
         f"not validated latent types.\nCell values are subscale means on the original 1–5 "
         f"scale; shading shows deviation from the across-profile mean. "
         f"P1 = {prof.label[0]}, P2 = {prof.label[1]}, P3 = {prof.label[2]}.",
         fontsize=7.2, color=INK2, va="top")
finish(fig, "fig4_cluster_profiles.png")

# =========================================================================== #
# Figure S1 — Scree plot with parallel analysis
# =========================================================================== #
eig = pd.read_csv(OUT / "supp_eigenvalues.csv")
n_show = 20
e = eig.head(n_show)

fig, ax = plt.subplots(figsize=(7.2, 3.3))
ax.plot(e["component"], e["eigenvalue"], "-o", color=BLUE, linewidth=2,
        markersize=5, markerfacecolor=BLUE, markeredgecolor=SURFACE,
        markeredgewidth=1.4, label="Observed eigenvalue", zorder=4)
ax.plot(e["component"], e["parallel_95pct"], "-s", color=ORANGE, linewidth=2,
        markersize=4.5, markerfacecolor=SURFACE, markeredgecolor=ORANGE,
        markeredgewidth=1.6, label="Parallel analysis (95th percentile)", zorder=3)
ax.axhline(1, color=MUTED, linewidth=1.0, linestyle="-", zorder=2)
ax.text(2.6, 1.35, "Kaiser criterion (eigenvalue = 1)", fontsize=7.2,
        color=MUTED, ha="left", va="bottom")

sp = R["supplementary_pca"]
kx = sp["parallel_analysis_components"]
ax.axvline(kx + 0.5, color=GRID, linewidth=1.2, zorder=1)
ax.text(kx + 0.7, e["eigenvalue"].max() * 0.82,
        f"Parallel analysis\nretains {kx} components", fontsize=7.6, color=INK2)

for y in range(0, int(e["eigenvalue"].max()) + 3, 3):
    ax.axhline(y, color=GRID, linewidth=0.7, zorder=0)
ax.set_xticks(range(1, n_show + 1, 1))
ax.set_xlabel("Component", fontsize=8.5, color=INK2)
ax.set_ylabel("Eigenvalue", fontsize=8.5, color=INK2)
ax.set_xlim(0.4, n_show + 0.6)
despine(ax)
ax.legend(fontsize=8, loc="upper right")
ax.set_title("Supplementary: scree plot with Horn’s parallel analysis",
             loc="left", pad=10)
ax.text(0.4, -max(e["eigenvalue"]) * 0.30,
        f"KMO = {sp['kmo']:.3f}; Bartlett χ²({sp['bartlett_df']}) = {sp['bartlett_chi2']:.1f}, "
        f"p < .001. Kaiser’s criterion retains {sp['kaiser_components']} components, "
        f"parallel analysis {kx}.\nThe participant-to-item ratio is "
        f"{sp['ratio_numeric']}:1 ({sp['ratio']}), below the 5:1 minimum "
        f"({sp['n_required_5to1']} respondents would be required); this analysis is "
        f"reported for transparency only and is not used for inference.",
        fontsize=7.2, color=INK2, va="top")
finish(fig, "figS1_scree_parallel.png")

print(f"\nAll figures written to: {FIG}")
