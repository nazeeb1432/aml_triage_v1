# From the replication to this repo, in plain words

The review of `eddin-fp-reduction` ("EDDIN Triage to Production") says three things:
your replication follows the paper, a few parts of it cheat without meaning to, and a bank needs much more around the
model before it can use it. This page explains each finding simply and points to where it is fixed here.

## The idea of the paper in four sentences
Banks' rules raise lots of alerts, and almost all are false. Instead of replacing the rules, put a model *after* them
that scores each alert. Analysts review only the high scores, or review everything in score order. The paper's trick is
to use the money-flow graph: who an account pays and gets paid by, and how close it sits to accounts already found suspicious.

## The audit findings, one by one

**1. GuiltyWalker peeked at the future (critical).**
GuiltyWalker asks "how many steps from this account to a known bad account?". The replication built the "known bad"
list once from the whole year's answers and used it for every day. So an alert in February could be scored using the
fact that its neighbour gets caught in November. A bank cannot know that in February. It's like grading a student with
the answer sheet on the desk: the scores look great and mean nothing.
*Fixed:* `labels/store.py` keeps the date each decision was made; `LabelIndex.illicit_mask_as_of(d)` returns only what
was known on day *d*. `tests/test_point_in_time.py` deletes everything after day *d* and checks no feature changes.

**2. Too few test cases to compare variants (high).**
The replication's test set had about 45 suspicious alerts, so one extra catch moved recall by 2–3 points, which is the size
of every difference being compared.
*Fixed:* a larger synthetic world (≈220 suspicious test alerts), and every number now comes with a 95% range from a
day-level bootstrap. If two ranges overlap, the report says "no detectable difference".

**3. The label is unrealistically clean and instant (high).**
In the replication, "suspicious" was known the moment the alert fired. In a bank it comes from an investigation weeks later.
*Fixed:* each label gets a simulated decision date (positives take longer). Training only uses labels that would exist at
training time, and the graph features only use labels decided before the alert day.

**4. The window sweep chose the graph using the answer (medium).**
The paper keeps suspicious edges longer than normal ones. The replication decided "suspicious" from the row's own label.
*Fixed:* an edge is kept longer only if it touches an account already confirmed suspicious *as of that day*.

**5. Scores were not probabilities and the threshold was implicit (medium).**
*Fixed:* isotonic calibration on validation; the review line τ is fixed on validation and then applied unchanged to test.

**6. Research-grade engineering (medium).**
*Fixed:* typed config that the code actually reads, data contracts, model manifests with data hash and git sha, tests,
numba graph kernels (100k alerts in ~30 s), CI.

**7. Closed-world typologies (medium).**
*Partly addressed:* per-typology recall is reported. A held-out-typology test is milestone M8.

## What "production grade" adds beyond the model
- **Policy separate from model**: hard overrides, the review line, and a blind *below-the-line* sample of suppressed alerts.
  Without that sample, nobody can ever measure what suppression misses, and a regulator will ask.
- **Reason codes** analysts can check against the payments.
- **Audit log** of every decision with model and policy version.
- **Sign-off**: model risk and compliance approve a model before it routes real alerts.
- **Rollout**: shadow mode → prioritise only → suppress one low-risk rule family → widen.

## About the GAN request
A GAN trained on alerts learns what the existing suspicious alerts look like and makes more that look the same.
It cannot teach the model a pattern that isn't already in the data, and its rows have no real transactions behind them.
So this repo treats it as an experiment: same model, trained with no rebalancing, class weights, SMOTE, or CTGAN rows,
compared on real held-out alerts. The ablation table in `reports/<run>/ablation.md` is the answer for your data.
