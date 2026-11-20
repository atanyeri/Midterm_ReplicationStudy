# AI appendix

## Statement

Sections 1 and 2 were written without AI, as the project requires.

Sections where AI was used: **Section 3, exercises 1 and 4.** Exercises 2 and 3
were written without AI. Each entry below gives the P.R.I.M.E. prompt, the
output in full, what I changed and why, and what I verified and against what.

---

## Entry 1

**Section and exercise:** Section 3, exercise 1 (placebo cutoffs).
**Outcome:** adopted, with five corrections. The notebook's original six-cutoff
design was replaced by this one, and Section 3.1 of the report was rewritten
around it.

### The P.R.I.M.E. prompt

**[Prep]** You are an applied microeconometrician reviewing a regression
discontinuity replication for a graduate methods course. Assume I know the
econometrics and am asking about implementation and interpretation, not
definitions.

**[Request]** I am replicating Hansen (2015), which estimates the effect of
exceeding Washington State's 0.08 DUI threshold on whether a driver is
breath-tested again within four years. The data is `hansen_dwi.dta`, one row per
breath test, 1999–2007, drivers 21 and over. The running variable is `bac_low`,
the lower of two breath readings; the outcome is `recidivism`, a 0/1 indicator.
Hansen reports −0.021. I want to test whether that jump is specific to the legal
threshold or an artifact of fitting a local linear model to a curved
relationship. Design a placebo-cutoff exercise.

**[Iterate]** I have an estimator `run_rd(rv, c, bw, strict=False,
controls=('male','white'), age_fe=True, year_fe=True, poly=1, kernel='rect',
cluster='bin', ycol='recidivism')` returning a dict with `coef`, `se`, `n`. Use
it; do not write a new estimator. Pandas, numpy, statsmodels only. Choose cutoffs
and a bandwidth such that no placebo window touches either real threshold, 0.08
or 0.15 — state the arithmetic that guarantees this. No nested loops; one
comprehension. Do not test cutoffs where the sample is so thin the standard error
is uninformative without saying so explicitly.

**[Mechanism Check]** Explain why placebo cutoffs test the continuity assumption
rather than the treatment effect, and what specific alternative explanation for
Hansen's result they are designed to rule out. Also explain why the comparison
must be against the real cutoff estimated at the same bandwidth, not against his
published −0.021.

**[Evaluate]** Tell me what pattern of results would undermine the paper's causal
claim and what pattern would support it, in terms of what each implies about
drivers near the threshold. State the smallest rank p-value attainable with the
number of cutoffs you chose, so I know the limits of what this can establish.

### Verification log

#### 1. The prompt, as sent

> You are an applied microeconometrician reviewing a regression discontinuity
> replication for a graduate methods course. Assume I know the econometrics and
> am asking about implementation and interpretation, not definitions.
>
> I am replicating Hansen (2015), which estimates the effect of exceeding
> Washington State's 0.08 DUI threshold on whether a driver is breath-tested
> again within four years. The data is `hansen_dwi.dta`, one row per breath test,
> 1999–2007, drivers 21 and over. The running variable is `bac_low`, the lower of
> two breath readings; the outcome is `recidivism`, a 0/1 indicator. Hansen
> reports −0.021. I want to test whether that jump is specific to the legal
> threshold or an artifact of fitting a local linear model to a curved
> relationship. Design a placebo-cutoff exercise.
>
> I have an estimator `run_rd(rv, c, bw, strict=False, controls=('male','white'),
> age_fe=True, year_fe=True, poly=1, kernel='rect', cluster='bin',
> ycol='recidivism')` returning a dict with `coef`, `se`, `n`. Use it; do not
> write a new estimator. Pandas, numpy, statsmodels only. Choose cutoffs and a
> bandwidth such that no placebo window touches either real threshold, 0.08 or
> 0.15 — state the arithmetic that guarantees this. No nested loops; one
> comprehension. Do not test cutoffs where the sample is so thin the standard
> error is uninformative without saying so explicitly.
>
> Explain why placebo cutoffs test the continuity assumption rather than the
> treatment effect, and what specific alternative explanation for Hansen's result
> they are designed to rule out. Also explain why the comparison must be against
> the real cutoff estimated at the same bandwidth, not against his published
> −0.021.
>
> Tell me what pattern of results would undermine the paper's causal claim and
> what pattern would support it, in terms of what each implies about drivers near
> the threshold. State the smallest rank p-value attainable with the number of
> cutoffs you chose, so I know the limits of what this can establish.

#### 2. What it generated, in full

```python
"""
Placebo-cutoff exercise for Hansen (2015), recidivism at the 0.08 BAC threshold.

Assumes `run_rd` is already importable and reads from `df`, and that `rv` is a
column name in the same way `ycol` is. If it takes an array, pass df["bac_r"].
"""
import numpy as np
import pandas as pd

# from your_module import run_rd

df = pd.read_stata("hansen_dwi.dta")

# BAC is recorded on a 0.001 grid. Do all cutoff arithmetic in integer
# thousandths so that window membership never depends on float rounding.
df["bac_k"] = np.rint(df["bac_low"] * 1000).astype(int)
df["bac_r"] = df["bac_k"] / 1000          # running variable passed to run_rd

# ---- design constants (thousandths) ---------------------------------------
H      = 25                     # half-bandwidth 0.025 (Hansen's narrow-bandwidth robustness spec)
REAL   = 80                     # 0.08
THRESH = np.array([80, 150])    # both legal thresholds
STEP   = 5                      # 0.005 spacing between placebo cutoffs
GRID   = np.arange(30, 276, STEP)   # 0.030 ... 0.275 (top window ends at 0.300)

# The window [c-H, c+H] excludes threshold T iff |c - T| >= H + 1. The extra
# 0.001 means T is never even the edge bin, so this holds whether run_rd uses
# <= or < at the window edge, and whatever `strict` does to the treatment side.
clear = (np.abs(GRID[:, None] - THRESH[None, :]) >= H + 1).all(axis=1)
placebos = GRID[clear]          # 30-50 (5), 110-120 (3), 180-275 (20) -> 28 cutoffs
cuts = np.concatenate([[REAL], placebos])

# ---- thinness screen: outcome-blind, decided before any estimate ------------
SPEC = dict(controls=("male", "white"), age_fe=True, year_fe=True,
            poly=1, kernel="rect", cluster="bin", ycol="recidivism")
est = df.dropna(subset=["recidivism", "male", "white"])
xs = np.sort(est["bac_k"].to_numpy())
n_lo = np.searchsorted(xs, cuts, "left") - np.searchsorted(xs, cuts - H, "left")       # [c-H, c)
n_hi = np.searchsorted(xs, cuts + H, "right") - np.searchsorted(xs, cuts, "left")      # [c, c+H]
n_min = np.minimum(n_lo, n_hi)

# SE scales roughly with 1/sqrt(n), so requiring a quarter of the real cutoff's
# thinner side keeps the placebo SE within about 2x the real-cutoff SE.
keep = n_min >= 0.25 * n_min[0]
keep[0] = True

screen = pd.DataFrame({"cutoff": cuts / 1000, "n_below": n_lo, "n_above": n_hi, "kept": keep})
print("Dropped as too thin (stated, not silently omitted):")
print(screen.loc[~screen.kept].to_string(index=False) or "  none")

# ---- estimation: one comprehension, identical spec at every cutoff ----------
res = pd.DataFrame([
    {"cutoff": c / 1000, "real": c == REAL, **run_rd("bac_r", c / 1000, H / 1000, **SPEC)}
    for c in cuts[keep]
])

res["t"] = res["coef"] / res["se"]
real = res.loc[res["real"]].iloc[0]
res["se_ratio"] = res["se"] / real["se"]
res["region"] = np.select(
    [res["cutoff"] < 0.08, res["cutoff"] < 0.15], ["below_0.08", "0.08_to_0.15"], "above_0.15")
res.loc[res["real"], "region"] = "REAL"

# ---- rank p-values (the real cutoff counts in its own reference set) ---------
K1 = len(res)                                               # K placebos + 1
p_abs_t   = (res["t"].abs() >= abs(real["t"])).mean()       # two-sided, on |t|
p_coef_lo = (res["coef"] <= real["coef"]).mean()            # one-sided, Hansen's sign

print(res.sort_values("cutoff").to_string(index=False))
print(f"\nreal cutoff @ h=0.025: coef={real['coef']:.4f} se={real['se']:.4f} n={real['n']}")
print(f"placebos kept: {K1 - 1};  smallest attainable rank p = 1/{K1} = {1 / K1:.3f}")
print(f"rank p (|t|, two-sided):   {p_abs_t:.3f}")
print(f"rank p (coef, one-sided):  {p_coef_lo:.3f}")
print("\nplacebo coefficients by region:")
print(res.loc[~res["real"]].groupby("region")[["coef", "t", "se_ratio"]]
      .agg(["mean", "min", "max", "count"]).round(4))
```

#### 3. What I changed, and why

**a. Added the derivation of the running variable.** The script reads the raw
`.dta` and immediately does `np.rint(df["bac_low"] * 1000)`. There is no
`bac_low` column in `hansen_dwi.dta`; it is derived in my notebook's data cell
from `low_score`, which the file stores in integer thousandths. Run as written it
raises `KeyError: 'bac_low'` before reaching any estimate. I used the `bac_low`
my notebook already builds.

**b. Removed the module-level `df = pd.read_stata(...)`.** My `run_rd` closes
over the notebook's global `df`, which by the estimator cell carries `bac1_r`,
`bac_low` and the other derived columns. I found this one the hard way: the line
executed before the `KeyError` in (a) and rebound `df` to a freshly loaded frame
without those columns, so every later cell calling `run_rd` would have failed or
run on the wrong frame. I restarted the kernel, re-ran from the top, and deleted
the load.

**c. Dropped the `bac_k` and `bac_r` columns from the estimation path.** The
script converts BAC to integer thousandths and back to avoid float-edge problems
at the window boundary, which is the right instinct, but `run_rd` already does
exactly this internally through `to_milli`, so `bac_r` is a second copy of
`bac_low` reached by the same arithmetic. I pass `bac_low` directly and use
`to_milli` for the screen's integer bins, so the screen and the estimator bin the
data identically rather than by two routes that happen to agree.

**d. Extended the `dropna` list.** The screen dropped rows missing `recidivism`,
`male` or `white`, but the specification it screens for also uses `aged` and
`year` as fixed effects, so the screen could have described a different sample
from the one estimated. Nothing is missing in this file, so the counts are
unchanged, but the screen should describe the regression it screens for.

**e. Fixed the empty-screen message.** `df.to_string() or "  none"` never prints
"none": `to_string` on an empty frame returns a non-empty string reading "Empty
DataFrame", which is truthy. I replaced it with an explicit length check.

**f. Adopted the bandwidth, and rewrote the report around it.** The script uses
a half-bandwidth of 0.025, matching Hansen's narrow robustness specification; my
notebook's original Section 3.1 used 0.02 and six cutoffs. These are not
reconcilable, so I took this one and rewrote Section 3.1 and the Section 4
bullets to match. The reason is in (g).

**g. Kept the design, as an improvement on my own.** My six-cutoff version capped
the attainable rank p-value at 1/7 ≈ 0.14, too coarse to reject anything. This
version tests 28 candidates, computes the rank p-values explicitly with the real
cutoff in its own reference set, which is the correct construction, and applies
an outcome-blind thinness screen before any estimate is seen — a stronger
discipline than my own approach of estimating everything and discussing the noisy
ones afterwards.

#### 4. What I verified, and against what

**a. The clearance arithmetic, by hand.** The script claims the rule
`|c − T| ≥ H + 1` yields 5 cutoffs in 0.030–0.050, 3 in 0.110–0.120 and 20 in
0.180–0.275, for 28. I recomputed the grid independently and got exactly
{30, 35, 40, 45, 50}, {110, 115, 120} and {180, 185, …, 275}: 5 + 3 + 20 = 28. I
also checked the boundary: at c = 55, 125 and 175 the nearest threshold is 25
thousandths away against a requirement of 26, so all three are correctly
excluded. The reasoning for the extra 0.001 is right too — at `|c − T| = H` the
threshold sits in the window's edge bin, and whether it enters depends on the
window being closed, which `run_rd` decides with `k.abs() <= bw`.

**b. The attainable rank p-value, against the run.** My own pre-run estimate was
wrong and the script's was right. I expected all 28 placebos to survive, giving a
floor of 1/29 ≈ 0.034. The screen dropped 11 — the five below 0.060 and the six
above 0.245 — leaving 17, so the realised floor is 1/18 = 0.056. The script
computes the floor from the kept set rather than the candidate set, so its
formula was correct and my projection was not.

**c. The missing column, by running it.** Confirmed as `KeyError: 'bac_low'`,
not merely by inspection. The notebook's data section lists the file's twelve
columns and none is `bac_low`; the data cell builds it from `low_score` and
asserts that `low_score` equals `min(Alcohol1, Alcohol2)` in every row.

**d. The real-cutoff estimate, against my own earlier output.** The script's
`SPEC` at the 0.08 cutoff with a half-bandwidth of 0.025 is the same
specification as the "T3 B: bandwidth 0.025, controls" row of Table 1 in my
report. That row reads −0.0193 on 50,016 observations. The script returns −0.0193
(SE 0.0048) on 50,016. Exact match, so the two are running the same
specification and the placebo comparisons are interpretable.

**e. The empty-frame truthiness, by test.** Confirmed directly that
`pd.DataFrame(columns=['a']).to_string()` returns a non-empty string, so the
`or "  none"` fallback is unreachable.

**f. The screen's stated rationale, against the output — it overshoots.** The
comment says keeping a quarter of the real cutoff's thinner side holds the
placebo standard error "within about 2x the real-cutoff SE". The kept placebos
run to a standard-error ratio of 2.97 at cutoff 0.245, and eight of the 17 exceed
1.5. The screen is still defensible as a rule, but its justification assumes the
standard error depends on the count alone, and it does not. I kept the rule and
corrected the comment rather than retune a threshold after seeing the estimates.

#### 5. What the adopted version produced

The real cutoff's absolute t-statistic, 4.01, is the largest of all 18, so the
two-sided rank p-value is 0.056 — its floor. The one-sided rank p-value on the
coefficient is 0.111, because the placebo at 0.205 estimates −0.0219 and is more
negative than the real cutoff.

Three of the 17 placebos are individually significant at the 5 percent level:
0.185 at t = +2.42, 0.205 at t = −3.14 and 0.240 at t = +2.57. That is 18 percent
against a nominal 5. Their point estimates are not large, between −0.0219 and
+0.0295, so the problem is the standard errors rather than the fit. This finding
is carried into Section 3.1 and the precision bullet of Section 4, and it agrees
with Entry 2's result that clustering on the running variable understates the
standard error in this data.

---

## Entry 2

**Section and exercise:** Section 3, exercise 4 (data-driven bandwidth and
bias-corrected intervals).
**Outcome:** adopted, with five corrections. It reproduced the notebook's
existing Section 3.4 numbers exactly.

### The P.R.I.M.E. prompt

**[Prep]** You are familiar with the post-2014 RD literature on bandwidth
selection and robust bias correction, and with its limitations when the running
variable is discrete.

**[Request]** Hansen (2015) fixes his bandwidth at 0.05 and 0.025 and reports
conventional confidence intervals. I want to re-estimate the 0.08 discontinuity
with an MSE-optimal bandwidth and a bias-corrected robust interval, following
Calonico, Cattaneo and Titiunik (2014), and compare both the point estimate and
the interval width against his. The data is `hansen_dwi.dta`, one row per breath
test in Washington State 1999–2007; the running variable is `bac_low`, the lower
of the two breath readings; the outcome is `recidivism`, a 0/1 indicator for
being tested or refusing a test again within four years.

**[Iterate]** Use `rdrobust` with `bac_low` as the running variable and
`recidivism` as the outcome, clustering on the 0.001 BAC bin as Hansen does.
Restrict the sample so the 0.15 aggravated-DUI threshold cannot enter the
bandwidth search, and explain why that restriction is necessary rather than
cautious. Run both triangular and uniform kernels, since uniform matches his
rectangular weighting. Report the selected bandwidth, the conventional estimate,
the bias-corrected estimate and the robust interval separately — do not collapse
them into one number. Flag any warnings the package emits rather than
suppressing them.

**[Mechanism Check]** Explain why a bandwidth chosen to minimize mean squared
error leaves a bias that a conventional confidence interval ignores, and what the
robust interval does about it. Explain what mass points in the running variable
do to bandwidth selection, given that BAC takes only about 80 distinct values on
each side of the cutoff, and whether that makes this exercise more or less
informative than it would be with a continuous running variable.

**[Evaluate]** Tell me whether the finding survives this estimator in sign and in
magnitude, and how much of Hansen's reported precision is a consequence of his
bandwidth choice. Given the upper end of the robust interval, state the smallest
effect a reader should regard as consistent with the data, and say whether "about
2 percentage points" remains a fair summary.

### Verification log

#### 1. The prompt, as sent

> You are familiar with the post-2014 RD literature on bandwidth selection and
> robust bias correction, and with its limitations when the running variable is
> discrete.
>
> Hansen (2015) fixes his bandwidth at 0.05 and 0.025 and reports conventional
> confidence intervals. I want to re-estimate the 0.08 discontinuity with an
> MSE-optimal bandwidth and a bias-corrected robust interval, following Calonico,
> Cattaneo and Titiunik (2014), and compare both the point estimate and the
> interval width against his. The data is `hansen_dwi.dta`, one row per breath
> test in Washington State 1999–2007; the running variable is `bac_low`, the
> lower of the two breath readings; the outcome is `recidivism`, a 0/1 indicator
> for being tested or refusing a test again within four years.
>
> Use `rdrobust` with `bac_low` as the running variable and `recidivism` as the
> outcome, clustering on the 0.001 BAC bin as Hansen does. Restrict the sample so
> the 0.15 aggravated-DUI threshold cannot enter the bandwidth search, and
> explain why that restriction is necessary rather than cautious. Run both
> triangular and uniform kernels, since uniform matches his rectangular
> weighting. Report the selected bandwidth, the conventional estimate, the
> bias-corrected estimate and the robust interval separately — do not collapse
> them into one number. Flag any warnings the package emits rather than
> suppressing them.
>
> Explain why a bandwidth chosen to minimize mean squared error leaves a bias
> that a conventional confidence interval ignores, and what the robust interval
> does about it. Explain what mass points in the running variable do to bandwidth
> selection, given that BAC takes only about 80 distinct values on each side of
> the cutoff, and whether that makes this exercise more or less informative than
> it would be with a continuous running variable.
>
> Tell me whether the finding survives this estimator in sign and in magnitude,
> and how much of Hansen's reported precision is a consequence of his bandwidth
> choice. Given the upper end of the robust interval, state the smallest effect a
> reader should regard as consistent with the data, and say whether "about 2
> percentage points" remains a fair summary.

#### 2. What it generated, in full

```python
"""
Re-estimate Hansen (2015, AER) 0.08 BAC discontinuity in recidivism with
CCT (2014) MSE-optimal bandwidth + robust bias-corrected inference.

Usage:  python hansen_rbc.py path/to/data.dta [--running bac_low] [--outcome recidivism]
Needs:  pip install rdrobust pandas
"""
import argparse, io, sys, warnings, contextlib
import numpy as np
import pandas as pd
from rdrobust import rdrobust

ap = argparse.ArgumentParser()
ap.add_argument("data")
ap.add_argument("--running", default="bac_low")
ap.add_argument("--outcome", default="recidivism")
ap.add_argument("--cutoff", type=float, default=0.08)
ap.add_argument("--upper", type=float, default=0.15,
                help="exclude obs with running >= this (aggravated-DUI threshold)")
a = ap.parse_args()

df = pd.read_stata(a.data) if a.data.endswith(".dta") else pd.read_csv(a.data)
df = df[[a.running, a.outcome]].dropna()
n_all = len(df)
df = df[df[a.running] < a.upper]          # 0.15 threshold can never enter any window
x, y = df[a.running].to_numpy(float), df[a.outcome].to_numpy(float)
bin_id = np.round(x * 1000).astype(int)    # 0.001 BAC bins, Hansen's cluster unit

print(f"N before/after dropping {a.running} >= {a.upper}: {n_all} / {len(df)}")
print(f"Distinct running values: below cutoff {np.unique(x[x < a.cutoff]).size}, "
      f"at/above {np.unique(x[x >= a.cutoff]).size}")

def run(label, **kw):
    buf = io.StringIO()
    with warnings.catch_warnings(record=True) as w, contextlib.redirect_stdout(buf):
        warnings.simplefilter("always")
        r = rdrobust(y, x, c=a.cutoff, p=1, masspoints="adjust", **kw)
    msgs = [str(m.message) for m in w] + [l for l in buf.getvalue().splitlines()
                                          if "arn" in l or "ass point" in l]
    h, b = r.bws.iloc[0], r.bws.iloc[1]
    print(f"\n=== {label} ===")
    print(f"h (left,right) = ({h.iloc[0]:.4f}, {h.iloc[1]:.4f});  "
          f"b (left,right) = ({b.iloc[0]:.4f}, {b.iloc[1]:.4f});  N_h = {list(r.N_h)}")
    for row in ["Conventional", "Bias-Corrected", "Robust"]:
        c, s = r.coef.loc[row].iloc[0], r.se.loc[row].iloc[0]
        lo, hi = r.ci.loc[row].iloc[0], r.ci.loc[row].iloc[1]
        print(f"  {row:15s} coef {c: .5f}  se {s:.5f}  95% CI [{lo: .5f}, {hi: .5f}]  "
              f"width {hi - lo:.5f}")
    print("  WARNINGS:", *(msgs or ["none"]), sep="\n    ")
    return r

for kern in ["triangular", "uniform"]:
    run(f"{kern}, MSE-optimal (mserd), cluster=0.001 bin",
        kernel=kern, bwselect="mserd", cluster=bin_id, vce="hc1")
    run(f"{kern}, MSE-optimal (mserd), unclustered HC3 [Kolesar-Rothe check]",
        kernel=kern, bwselect="mserd", vce="hc3")
    run(f"{kern}, CER-optimal (cerrd), cluster=0.001 bin",
        kernel=kern, bwselect="cerrd", cluster=bin_id, vce="hc1")

# Hansen's fixed bandwidths, rectangular kernel: 'Conventional' row ~ his spec
# (he also adds covariates; pass covs=df[[...]] to match him exactly)
for hh in [0.05, 0.025]:
    run(f"uniform, fixed h={hh} (Hansen), cluster=0.001 bin",
        kernel="uniform", h=hh, cluster=bin_id, vce="hc1")
```

#### 3. What I changed, and why

**a. Added the derivation of the running variable.** The script defaulted
`--running` to `bac_low` and selected that column straight off the raw `.dta`.
That column does not exist in `hansen_dwi.dta`; it is derived in my notebook's
data cell from `low_score`, which the file stores in integer thousandths. As
generated, the script raises `KeyError` on the first line that touches the data.
I added:

```python
if a.running == "bac_low" and "bac_low" not in df.columns:
    df["bac_low"] = df["low_score"].astype("float64") / 1000
```

the same construction my notebook uses, so both use an identical running variable
rather than two things that share a name. This is the same defect as Entry 1 (a),
in a separately generated script.

**b. Removed `masspoints="adjust"`.** The model passed this explicitly, which
made it read as a deliberate correction for the discreteness of BAC. It is the
package default, so the argument is a no-op. I removed it rather than keep a line
that looks like a methodological choice and changes nothing. Where I did want the
mass points documented I used `masspoints="check"`, which reports the count of
unique observations on each side of the cutoff — about 80 per side, the number
worth putting in the report.

**c. Relabelled the unclustered run.** The model labelled an unclustered HC3
specification "[Kolesar-Rothe check]". Kolesár and Rothe do advise against
clustering on a discrete running variable, so running without clusters is in the
spirit of their argument, but their actual proposal is a pair of honest
confidence intervals implemented in the RDHonest package, not HC3. I renamed the
run to "unclustered HC3; Kolesár and Rothe (2018) advise against clustering on a
discrete running variable", so the label claims only what the specification does.

**d. Corrected the comment on the sample restriction.** The generated comment
said the 0.15 threshold "can never enter any window". Dropping observations at or
above 0.15 keeps them out of the estimation, but the selected bandwidth can still
extend toward 0.15 and truncate the right side, leaving the window unbalanced. I
rewrote the comment to say that.

**e. Removed `vce="hc1"`.** The model passed this alongside `cluster`, where my
notebook passed no `vce` at all. Running it showed the package refuses the
combination and says so: `vce='hc1' is not a cluster option. Switching to
vce='cr1'.` The argument therefore changed nothing — the estimator used was CR1
either way, which is what my notebook's output header reported. I removed it,
because an argument the package silently overrides reads as a deliberate choice
of variance estimator while being the opposite: a request that was declined. The
warning appears on every clustered run and on neither of the unclustered HC3
runs, which is consistent with that reading.

**f. Kept, with a reason.** The model added two specifications the prompt did not
request: a CER-optimal bandwidth (`bwselect="cerrd"`) and Hansen's own fixed
bandwidths of 0.05 and 0.025 under a uniform kernel. I kept both. The CER
bandwidth optimises coverage error rather than mean squared error, which is the
more relevant criterion when the interval is the object of interest, and the
fixed-bandwidth runs turned out to validate my own estimator (item 4g below).

#### 4. What I verified, and against what

**a. The missing column, against my own notebook.** The notebook's data cell
builds `bac_low` and asserts that `low_score` equals `min(Alcohol1, Alcohol2)` in
every row. The file's twelve columns, listed in the notebook's data section,
contain no `bac_low`. The script's default was wrong by inspection, before
running it.

**b. The `masspoints` default, against the package documentation.** The rdrobust
reference states that the default for `masspoints` is `adjust`, which ensures the
preliminary bandwidths contain a minimum number of unique observations, with
`off` and `check` as the alternatives. This confirmed the argument was redundant
rather than load-bearing.

**c. The `vce` behaviour, against the package documentation and the run.** The
same reference states that supplying a cluster variable enables cluster-robust
variance estimation, by default combined with `vce=nn`, the nearest-neighbour
estimator, with `hc0` as an alternative. Running the script then showed the
package rejecting `hc1` outright and switching to `cr1`, which settled it.

**d. The Kolesár–Rothe attribution, against the paper.** Kolesár and Rothe (2018)
recommend against confidence intervals based on standard errors clustered by the
running variable, reporting that in some of their simulations the clustered
intervals covered the true effect only 58 percent of the time at a nominal 95
percent level. Their proposed alternatives are two honest confidence intervals,
one bounding the second derivative of the conditional expectation function and
one bounding the approximation bias at the threshold, both implemented in
RDHonest. The generated label attached their name to a specification they do not
propose.

**e. The estimates, against my notebook's earlier output.** The script reproduces
the notebook's Section 3.4 cell exactly. Triangular kernel, MSE-optimal
bandwidth, clustered: h = 0.0223, conventional −0.01629, bias-corrected −0.01517,
robust interval [−0.02644, −0.00390], against the 0.0223, −0.0163, −0.0152 and
[−0.0264, −0.0039] the notebook records. Uniform kernel: h = 0.0172, conventional
−0.01837, robust interval [−0.03044, −0.00458], against 0.0172, −0.0184 and
[−0.0304, −0.0046]. Every figure agrees to the precision the notebook stores.

**f. The row labels, by running it.** I had flagged that the script indexes with
`r.coef.loc["Bias-Corrected"]` and `r.se.loc["Robust"]` where my notebook used
positional `.iloc`, and that rdrobust's printed summary shows `-` for the Robust
coefficient and standard error. Running it settled the question: the labels exist
and all three rows are populated, so the `-` in the package's own summary is
cosmetic. Bias-Corrected and Robust share a point estimate (−0.01517) and differ
only in standard error (0.00491 against 0.00575), which is what the method
implies — the robust interval widens to account for having estimated the bias, it
does not move the estimate.

**g. My own estimator, against an independent package.** The two fixed-bandwidth
runs the model added are an unplanned validation of `run_rd` itself. With a
uniform kernel, h = 0.05 and bin clustering, rdrobust's conventional estimate is
−0.02185 with a standard error of 0.00420; my notebook's no-controls estimate at
the same bandwidth is −0.0218 with 0.0042. At h = 0.025 the two are −0.01940
against −0.0194, and 0.00494 against 0.0049. Two independent implementations
agreeing to the fourth decimal means the remaining gap against Hansen's published
−0.020 is not an error in my estimator, which supports item 2 of the discrepancy
log: the gap is a sample difference rather than an implementation one.

---

## Note carried into the report

Four things surfaced from verifying these outputs that bear on the report itself,
not just on the code.

**Clustering on the running variable understates the standard error here.**
Kolesár and Rothe (2018) object to exactly the inference Hansen uses and that my
notebook uses throughout, on the grounds that clustered standard errors are often
smaller than heteroskedasticity-robust ones. The unclustered HC3 runs confirm it
in this data: 0.00626 against the clustered 0.00491 under a triangular kernel,
and 0.00673 against 0.00569 under a uniform one. Clustering is buying roughly a
fifth of the reported precision.

**And the placebos confirm it from a third direction.** Three of the 17 placebo
cutoffs in Entry 1, where no sanction changes, are individually significant at
the 5 percent level — 18 percent against a nominal 5. Theory, an alternative
variance estimator, and the placebo distribution all say the same thing. The
precision bullet of Section 4 was rewritten to reflect this; the earlier claim
that "the precision holds" was reached by collapsing to bin means while still
clustering, which does not test the clustering itself.

**Bias correction bites hardest at Hansen's own bandwidths.** At h = 0.05 the
conventional estimate is −0.02185 and the bias-corrected one is −0.01295, a
reduction of 41 percent; at h = 0.025 it is −0.01940 against −0.01309. At the
data-driven bandwidth, roughly half as wide, the correction is far smaller:
−0.01629 against −0.01517. So the distance between the paper's headline and the
bias-corrected estimates is mostly a consequence of how wide its windows are.

**The CER-optimal interval nearly includes zero.** Choosing the bandwidth to
minimise coverage error rather than mean squared error gives h = 0.0134 under a
uniform kernel and a robust interval of [−0.02692, −0.00042]. The upper bound is
four hundredths of a percentage point from zero. The sign survives, but barely,
under the bandwidth rule designed for the coverage of the interval itself.

## References used in verification

- rdrobust package documentation:
  https://search.r-project.org/CRAN/refmans/rdrobust/html/rdrobust.html
- Kolesár, M. and Rothe, C. (2018). Inference in Regression Discontinuity Designs
  with a Discrete Running Variable. *American Economic Review* 108(8): 2277–2304.
