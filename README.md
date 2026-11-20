# Replicating Hansen (2015): the sign survives, the size is a range

ECON 5200 Replication Study. Hansen reports that a BAC above the 0.08 DUI
threshold cuts repeat drunk driving by 0.021 (SE 0.004); this replication gets
−0.0221 (SE 0.0041), and across 192 defensible specifications the estimate runs
from −0.0241 to −0.0072 with a median of −0.0165.

## The one command

```
pip install -r requirements.txt
python reproduce.py
```

`reproduce.py` downloads the Washington State breath-test file if it is missing, executes
`replication.ipynb` end to end in a fresh kernel on the same interpreter that installed
`requirements.txt`, writes the executed copy with all its outputs to
`output/executed/replication.ipynb`, and prints the headline table from
`output/headline_table.csv`. The notebook's last cell prints every number the report
quotes. Runtime: about 1 minute 20 seconds.

`requirements.txt` pins all nine packages: the five the analysis imports and the four
`reproduce.py` needs to run the notebook.

## Data

`hansen_dwi.dta` from
`https://raw.githubusercontent.com/scunning1975/mixtape/master/hansen_dwi.dta`,
downloaded at run time by `download_data.py` rather than committed. The
underlying data is Hansen's replication package for the *American Economic
Review*; the copy used here comes from the Mixtape repository, whose licence
GitHub classifies as "other" rather than a standard open one. Redistributing a
dataset whose terms have not been confirmed to permit it is not worth the risk
for a file anyone can fetch in one command, so the repository ships the fetcher
and not the file. `download_data.py` rejects an empty response or an HTML error page, so a failed fetch
stops rather than leaving a file that is not a `.dta`. A file already in `data/` is not
downloaded again; `--force` replaces it.

Three columns the file does not document decide the analysis. `Alcohol1` and
`Alcohol2` are the two breath readings in integer thousandths, and `low_score` is
their row-wise minimum — the paper's running variable, under a name that says
nothing about it. `bac1` is the lower reading in only 48.3 percent of rows, so it
is not that variable. The notebook's data cell asserts the identity rather than assuming it.

## Layout

| path | what it is |
| --- | --- |
| `replication.ipynb` | Sections 2 to 4, in order; the last cell prints every number the report quotes |
| `download_data.py` | fetches `hansen_dwi.dta` into `data/` and rejects a non-`.dta` response  |
| `reproduce.py` | the one command; executes the notebook and writes `output/` |
| `requirements.txt` | pinned analysis dependencies (Python 3.12) |
| `ReplicationMidtermReport.pdf` | the report |
| `ai-appendix.md` | which sections used AI, with the P.R.I.M.E. entries and verification logs |

## Running it by hand

The notebook runs top to bottom in one pass, which is what `reproduce.py` does. The cells
are not independent: the data cell defines `to_milli` and builds the running variables, the
estimator cell defines `run_rd`, and every cell after it calls `run_rd`. Running a later
cell against a fresh kernel raises `NameError`. After any restart, run all cells from the
top.

## Known gaps

Four discrepancies against the paper are unresolved and documented in the
notebook's discrepancy log: the headline gap of 0.0011 (0.29 paper standard
errors), the sample-size gaps, which change sign across windows so no uniform row
deletion explains them, the triangular-kernel gap of 0.0025, and the baseline
behind the paper's "17 percent". The public file also lacks the county and
prior-test variables, so Table 3's subgroup columns cannot be replicated at all,
and the four-year recidivism window is pre-built into a single indicator that
cannot be re-derived from the one date column.

## AI Use

Section 3 used AI with the entries in ai-appendix.md
