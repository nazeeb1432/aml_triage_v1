# Data

## IBM AMLworld (real file, recommended)

Altman et al., *Realistic Synthetic Financial Transactions for Anti-Money Laundering Models*, NeurIPS 2023 Datasets & Benchmarks.
Kaggle: `ealtman2019/ibm-transactions-for-anti-money-laundering-aml` (needs a free Kaggle login).

| File | Size (approx.) | Use |
|---|---|---|
| `HI-Small_Trans.csv` | ~5M transactions, ~500k accounts | start here: fits a laptop |
| `HI-Small_Patterns.txt` | laundering attempts by typology | per-typology recall |
| `LI-Small_*` | lower illicit ratio | robustness check (harder) |
| `HI-Medium`, `*-Large` | 30M–180M rows | needs a bigger machine; Large spans ~3 months |

```bash
pip install kaggle   # put kaggle.json in ~/.kaggle/
kaggle datasets download ealtman2019/ibm-transactions-for-anti-money-laundering-aml -f HI-Small_Trans.csv -p data/raw --unzip
kaggle datasets download ealtman2019/ibm-transactions-for-anti-money-laundering-aml -f HI-Small_Patterns.txt -p data/raw --unzip
make run-ibm
```

The ingest stage prints the day span. HI-Small is short (roughly 2–3 weeks, most activity early), so
`configs/ibm_hi_small.yaml` shrinks every window and split. Check the printed span and adjust the split days if needed.

### What the IBM data has and does not have

| Has | Does not have |
|---|---|
| transactions with timestamps, banks, accounts, amounts, currencies, channel | rule-engine alerts |
| transaction-level `Is Laundering` ground truth | analyst dispositions, SARs, case IDs |
| laundering typology per attempt (patterns file) | customer/KYC data, account opening dates |

So this repo **creates alerts with its own rule engine** and **derives alert labels** ("the alerted account touched a
laundering transaction that day"), with simulated decision delays. That is a realistic stand-in for the paper's setting,
not a copy of it. When you move to your bank's data, the alerts and labels come from the real rule engine and case
management, and `ingest/` + `labels/` are the only packages that change.

### Canonical transaction schema (`contracts.TRANSACTIONS`)

`txn_id, ts, day, src, dst, amount_usd, pay_format, currency, cross_currency, is_laundering, typology`.
Account key is `<bank>_<account>` with leading zeros stripped from the bank ID (files disagree on padding).
Amounts are converted to USD with fixed approximate rates (`ingest/ibm.py:FX_TO_USD`).

## IBM AMLSim example (supported, recommended second dataset)

Kaggle: `anshankul/ibm-amlsim-example-dataset` (IBM AMLSim, Weber et al. 2018). 10,000 accounts, 200 days,
1,323,234 transfers, 1,719 laundering transfers in 391 schemes (cycle and fan-in). 60 MB unzipped.

```bash
make get-amlsim     # kagglehub download -> data/raw/amlsim/{transactions,alerts,accounts}.csv
make run-amlsim     # configs/amlsim.yaml; ~25-40 min on a laptop, mostly the CTGAN arm
```

Reader: `ingest/amlsim.py`. Its `alerts.csv` is the ground-truth list of laundering transactions (with typology),
not rule-engine output; alerts are still created by `rules/engine.py`.

What is different about this data, and how the profile handles it:
- Every laundering transfer is under $25 (median ~$11) while legitimate transfers have a median of ~$160-550, and
  every transaction is a plain `TRANSFER` between US accounts. The dollar thresholds tuned for AMLworld never fire on
  laundering here, so `configs/amlsim.yaml` turns on a low-value ("micro-transfer") rule and retunes fan-in/fan-out.
- Calibrated on the full file (2026-10-25). The final rule set (excluding the spike rule, which only adds noise)
  raises ~373,000 alerts (~1,870 a day) at 0.9% precision, covering 100% of laundering account-days. Expected
  alerts / positive alerts per split: train (days 30-139) ~199k / 1,834, validation (140-159) ~41k / 337,
  test (160-199) ~83k / 734. That is more than three times the positives of the synthetic test period.
- Laundering accounts are rarely reused across schemes (1,685 accounts for 391 schemes), so expect GuiltyWalker's
  "prior confirmed" signal to be weaker than on the synthetic world.
- Result (2026-10-26, run `amlsim-20261026T055105Z`): the run reproduced the calibration exactly (373,744 alerts,
  99.1% false, 3,342 positive). Chosen model: all feature families, class weights, Platt calibration. Test recall
  98.5% (98–99) with 78.9% of false alerts removed; 5-period backtest 95.5–98.8%. Held-out pattern: cycle 79%,
  fan-in 86% when never seen in training. Label delay leaves only 75 decided positives in validation, which is why
  selection uses the rolling folds (889 positives) and why calibration switched to Platt (DECISIONS D14).

## Synthetic generator (`ingest/synth.py`)

Writes `SYNTH_Trans.csv` and `SYNTH_Patterns.txt` in the IBM format, so the same reader runs. Default: 12,000 accounts,
150 days, ~1M transactions, 260 laundering attempts across the 8 IBM typologies.

Built to make triage hard in the same ways real data is:
- legitimate behaviour that trips rules: payroll fan-outs, merchant fan-ins, large purchases, invoices just under 10k,
  broker pass-through (rules alone: ~98.5% false positives)
- laundering routed through accounts with normal history (mules), reused across attempts
- a third of schemes are "low and slow" (smaller amounts, days between hops) and often never alert

Use it for development, tests and demos. Do not quote its numbers as evidence about real laundering.

## Generating more data with a GAN

Two different things get called "GAN-generated AML data":
1. **Feature-level augmentation** (implemented, `augment/`): CTGAN on alert feature vectors to add minority rows to training.
2. **Transaction-level synthesis** (not implemented): generating whole ledgers. Tabular GANs do not model the graph and
   time structure that laundering lives in; the IBM generator itself is an agent-based simulator for exactly that reason.
   If you need more transaction data, use a bigger IBM file or an agent-based simulator (AMLSim, IBM's generator), not a GAN.
