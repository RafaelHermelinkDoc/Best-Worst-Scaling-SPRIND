# Exploratory run, 20 Oct 2026

Purpose: design and analysis decisions before the item list is final. Not the
pre-registered sample-size simulation. All assumptions are placeholders
(true utilities evenly spaced from -1 to +1, heterogeneity SD 0.5, scale 1 in
group 0) and will be replaced by pilot estimates.

Reproduce, from the repository root (versions in the main README):

```bash
for e in exp1 exp2 exp3 exp3b exp4 exp5 exp5b exp6 exp7 exp7b; do
  python -m bws_sim.experiments $e --out results/exploratory_2026-10-20
done
python scripts/make_figure.py results/exploratory_2026-10-20
```

## Main results

| exp | finding |
|---|---|
| exp1 | No true difference, N = 150 per group. Share of items flagged at p < 0.05: equal scale 5% / 6% / 4% (counts / logit with equal scale / logit with free scale). Scale ratio 0.8: 15% / 12% / 5%. Scale ratio 0.6: 47% / 32% / 4%. 15% random responders in one group: 14% / 10% / 5%. With a true shift of 0.5 on one item, power for that item at scale ratio 0.8 is 79% / 52% / 94%. In that case the scale-adjusted single-item test flags 7-9% of the other items. |
| exp2 | Three strong group-specific items in one group and three weak ones in the other, common items identical: 96% of the common items are flagged when counts are compared; the joint logit flags 5%. |
| exp3 | One joint 14-item design (4-GDD 2^7, 14 tasks) vs. two blocks BIBD(7,4,2) (14 tasks). Mean SD of within-domain differences: 0.111 vs. 0.106. Across domains: 0.109 in the joint design; not estimable with two blocks. |
| exp3b | Two blocks plus 1/2/3 bridging sets (15/16/17 tasks). Across-domain SD: 0.161/0.134/0.114. |
| exp4 | A dominant item (best share 45% / 67% / 84% for utility 1 / 2 / 3) barely changes the precision of the other items: SD of their differences 0.107 / 0.109 / 0.110; correct top 3 among the others 94% / 92% / 91%. |
| exp5 | A polarising item (half +1.5, half -1.5) lands in the middle of the aggregate ranking (on average 6 items above it). It had the largest SD of individual count scores in 100% of runs at N = 30, 60 and 150. |
| exp5b | Largest SD of individual count scores (in parentheses: among the two largest), 300 runs. Strength +/-1.5: N = 20 97% (99%), N = 30 100%, N = 50 100%. +/-1.0: 61% (78%), 78% (89%), 91% (95%). +/-0.75: 28% (46%), 40% (57%), 49% (70%). |
| exp6 | N = 100, heterogeneity SD 0.5: 10 instead of 13 tasks raise the SD of differences by 12%, 7 tasks by 32%. 130 respondents x 10 tasks give the same precision as 100 x 13. With heterogeneity SD 1.0, more respondents with fewer tasks is better. |
| exp7 | Power for a shift on one item (item utility 0.5), alpha 0.05, N per group 75/100/150/200/300. Scale-adjusted test, equal scale: shift 0.5 -> 74/88/97/100/100%; shift 0.3 -> 34/45/66/80/92%. Scale ratio 0.8: shift 0.5 -> 64/82/96/98/100%; shift 0.3 -> 30/44/59/72/86%. Counts at scale ratio 0.8: shift 0.5 -> 48/61/80/90/98%, and false positives (exp1). With 13 evenly spaced items, a shift of 0.5 is about 3 rank positions, 0.3 about 2. |
| exp7b | Power for the item with a true shift, scale ratio 0.8, N per group 100/200/300, 200 runs. Standard (heterogeneity SD 0.5): shift 0.5 -> 83/99/100% uncorrected, 48/88/100% with Bonferroni over 12 items; shift 0.3 -> 36/69/83%, 9/31/57%. Stress (SD 1.0, 15% random responders in group 1): shift 0.5 -> 49/80/98%, 12/47/79%; shift 0.3 -> 18/40/61%, 2/14/19%. Bonferroni is a lower bound for Holm. |
