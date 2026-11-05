# Yiming Zhan
# https://github.com/<username>/deductible-truncation-check

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Ingest one CSV per simulation case and stack them
CASES = {"B": "left truncation at the deductible", "C": "truncation + censoring at the policy limit"}
sim = pd.concat([pd.read_csv(f"data/case{c}_summary.csv") for c in CASES], ignore_index=True)
print(f"{len(sim)} rows x {sim.shape[1]} columns; first columns: {list(sim.columns[:9])}")

# 2. Data types: ordered categories for the design factors, floats for the estimates
sim["case"] = pd.Categorical(sim["case"], categories=list(CASES))
sim["d"] = pd.Categorical(sim["d"].astype(str), categories=["0.1", "0.4", "var"], ordered=True)
sim["estimator"] = pd.Categorical(sim["estimator"], categories=["naive", "joint", "oracle_complete"])
sim["tau"] = sim["tau"].astype(float)
print(sim.dtypes[["case", "tau", "d", "estimator", "bias", "coverage95"]].to_string())
print("share of missing coverage95:", sim.groupby("estimator", observed=True)["coverage95"]
      .apply(lambda s: round(s.isna().mean(), 2)).to_dict())   # naive fit reports no SEs -> NaN

# 3. Functions
def bias_in_mcse(bias, mcse):
    """Bias / Monte-Carlo SE; NaN where MC-SE is 0 (parameter held fixed, e.g. naive theta = 0)."""
    return pd.Series(np.where(mcse > 0, bias / mcse, np.nan), index=bias.index)

def coverage_ok(cov, n_reps, level=0.95):
    """True when empirical coverage is within 2 binomial SEs of the nominal level."""
    return (cov - level).abs() <= 2 * np.sqrt(level * (1 - level) / n_reps)

def bias_panel(ax, df, case, param="meanlog"):
    """Naive vs joint bias of one parameter with +/-2 MC-SE bars, one simulation case."""
    sub = df[(df["case"] == case) & (df["param"] == param)].sort_values(["tau", "d"])
    grid = sub.drop_duplicates(["tau", "d"])
    x = np.arange(len(grid))
    for est, offset, colour in [("naive", -0.15, "#d95f02"), ("joint", 0.15, "#1b6ca8")]:
        s = sub[sub["estimator"] == est]
        ax.errorbar(x + offset, s["bias"], yerr=2 * s["bias_mcse"], fmt="o", ms=4, color=colour, label=est)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(x, [f"tau={t:+.1f}\nd={d}" for t, d in zip(grid["tau"], grid["d"])], fontsize=6.5)
    ax.set_title(f"Case {case}: {CASES[case]}", fontsize=9, loc="left")
    ax.set_ylabel(f"bias of {param} (+/- 2 MC-SE)")

# 4. Wrangle: derived column, filter, group-by aggregation, wide table
sim["z"] = bias_in_mcse(sim["bias"], sim["bias_mcse"])
fits = sim[sim["estimator"].isin(["naive", "joint"])].assign(abs_z=lambda t: t["z"].abs())
cells = (fits.groupby(["case", "tau", "d", "estimator"], observed=True)
             .agg(max_abs_z=("abs_z", "max"), coverage=("coverage95", "mean"),
                  n_reps=("n_used", "min"), boundary_fits=("n_boundary", "max"))
             .reset_index())
z_wide = cells.pivot_table(index=["tau", "d"], columns=["estimator", "case"],
                           values="max_abs_z", observed=True).round(1)
print("\nmax |bias / MC-SE| over the 5 model parameters, naive vs joint fit, cases B and C:")
print(z_wide.to_string())

# 5. Analysis
joint = cells[cells["estimator"] == "joint"].copy()
joint["coverage_ok"] = coverage_ok(joint["coverage"], joint["n_reps"])
print(f"\njoint fit: {joint['coverage_ok'].sum()} of {len(joint)} cells have 95% coverage within "
      f"2 MC-SE of nominal; mean coverage = {joint['coverage'].mean():.3f}")
print("cells where the joint fit still has max |z| > 3:")
print(joint.loc[joint["max_abs_z"] > 3, ["case", "tau", "d", "max_abs_z", "coverage", "boundary_fits"]]
      .round(2).to_string(index=False))

# 6. Figure
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.1), sharey=True)
for ax, case in zip(axes, CASES):
    bias_panel(ax, sim, case)
axes[0].legend(fontsize=8, frameon=False)
axes[1].set_ylabel("")
fig.tight_layout()
fig.savefig("figure_bias.png", dpi=200)
