# Best-worst scaling: simulation code

Simulation code for the design, analysis plan and sample-size planning of a
Case 1 (object case) best-worst scaling (BWS) survey with two respondent groups
(clinicians and patients), carried out within the SPRIND Continuous Hormone
Monitoring Challenge.

**Status:** exploratory (20 Oct 2026). These runs support design decisions
(item set, design, analysis model). The simulations reported in the paper will
be re-run with the final design and with utilities and scale estimated in the
pilot, and frozen before the main data collection (pre-registration).

## Model

- Sequential best-worst logit: the best item is chosen from all k items in a
  set, then the worst item from the remaining k - 1 items with reversed sign;
  i.i.d. Gumbel errors.
- Preference heterogeneity: `beta_i = beta + eta_i`, `eta_i ~ N(0, sd^2)`.
- Scale parameter per group (lambda); random responders as lambda = 0.
- Items are randomly permuted to design positions for each respondent.
- Estimation: conditional sequential best-worst logit (maximum likelihood,
  analytic gradient), optional free scale parameter per group and
  group-specific shift `delta_j`, cluster-robust (sandwich) standard errors by
  respondent. Count scores: standardised (B - W) / occurrences.

## Experiments

| id | question |
|---|---|
| exp1 | False positives and power of group comparisons when groups differ in scale (consistency) |
| exp2 | Group-specific items: does a count-based comparison of the common items stay unbiased? |
| exp3 | One joint 14-item design vs. two separate 7-item blocks (BIBD 7,4,2) |
| exp3b | Two blocks plus 1-3 bridging sets |
| exp4 | Cost of a dominant item for the precision of the other items |
| exp5 | Detecting a polarising item from individual count scores (pilot sizes) |
| exp5b | Polarisation detection for three strengths at N = 15-50 (soft-launch sizes) |
| exp6 | Number of tasks per respondent vs. number of respondents |
| exp7 | Power curve for a single-item group difference: scale-adjusted test vs. counts |
| exp7b | Power with a Bonferroni correction over 12 items and under a stress scenario (heterogeneity SD 1.0, 15% random responders) |

Designs used: BIBD(13,4,1) from the difference set {0,1,3,9} mod 13;
BIBD(7,4,2) as complements of the Fano plane; a 4-GDD of type 2^7 for 14 items
in 14 sets (each item 4 times, each pair at most once), found by exact-cover
search in `bws_sim/core.py`.

## How to run

```bash
python -m pip install -r requirements.txt
python tests/test_core.py                     # or: python -m pytest -q
python -m bws_sim.experiments exp5 --out results/latest
python -m bws_sim.experiments exp1 --workers 4 --out results/latest
python scripts/make_figure.py results/latest  # needs exp1, exp2, exp3, exp3b, exp5
```

`--reps` defaults to the number of replications used in the reported run.
Each replication has its own fixed seed, so results do not depend on
`--workers`. Run times on 2 cores: exp1 about 30 min, exp7 about 15 min, exp7b about
10 min, all others 1-5 min.

## Reproducibility

Exploratory results are stored in `results/exploratory_2026-10-20/` and were
produced with Python 3.13.16, numpy 2.5.3, scipy 1.18.1 and matplotlib 3.11.2.
NumPy does not guarantee identical random streams across versions, so other
versions can change the last digits.

## Structure

```
bws_sim/core.py          designs, simulation, estimator, count scores
bws_sim/experiments.py   experiments exp1-exp7 (command line)
scripts/make_figure.py   summary figure
tests/test_core.py       design checks, gradient check, parameter recovery
results/                 stored results per run
```

## Key references

- Flynn TN, Louviere JJ, Peters TJ, Coast J. Best-worst scaling: what it can do for health care research and how to do it. J Health Econ. 2007;26(1):171-189.
- Marley AAJ, Louviere JJ. Some probabilistic models of best, worst, and best-worst choices. J Math Psychol. 2005;49(6):464-480.
- Swait J, Louviere J. The role of the scale parameter in the estimation and comparison of multinomial logit models. J Mark Res. 1993;30(3):305-314.
- Fiebig DG, Keane MP, Louviere J, Wasi N. The generalized multinomial logit model: accounting for scale and coefficient heterogeneity. Mark Sci. 2010;29(3):393-421. doi:10.1287/mksc.1090.0508
- Dyachenko T, Reczek RW, Allenby GM. Models of sequential evaluation in best-worst choice tasks. Mark Sci. 2014;33(6):828-848. doi:10.1287/mksc.2014.0870
- de Bekker-Grob EW, Donkers B, Jonker MF, Stolk EA. Sample size requirements for discrete-choice experiments in healthcare: a practical guide. Patient. 2015;8(5):373-384. doi:10.1007/s40271-015-0118-z
- Hollin IL, Paskett J, Schuster ALR, Crossnohere NL, Bridges JFP. Best-worst scaling and the prioritization of objects in health: a systematic review. PharmacoEconomics. 2022;40(9):883-899.
- Sawtooth Software. The MaxDiff System Technical Paper, version 9; 2020.
