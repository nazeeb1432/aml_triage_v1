"""Typed configuration. YAML in, validated pydantic object out.

A run is described by `configs/default.yaml`, optionally deep-merged with a profile
(e.g. `configs/ibm_hi_small.yaml`) and `key.path=value` overrides from the CLI.
The resolved config is written next to every model so any run can be reproduced.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class SyntheticCfg(BaseModel):
    n_accounts: int = 12000
    n_banks: int = 25
    n_days: int = 150
    start_date: str = "2024-01-01"
    laundering_patterns: int = 260
    out_trans_file: str = "SYNTH_Trans.csv"
    out_patterns_file: str = "SYNTH_Patterns.txt"


class DataCfg(BaseModel):
    source: Literal["synthetic", "ibm", "amlsim"] = "synthetic"
    raw_dir: str = "data/raw"
    ibm_trans_file: str = "HI-Small_Trans.csv"
    ibm_patterns_file: str = "HI-Small_Patterns.txt"
    amlsim_dir: str = "data/raw/amlsim"
    synthetic: SyntheticCfg = SyntheticCfg()


class RulesCfg(BaseModel):
    large_txn_usd: float = 10000
    structuring_low_usd: float = 8000
    structuring_high_usd: float = 10000
    structuring_min_count: int = 2
    fan_out_min_counterparties: int = 6
    fan_in_min_counterparties: int = 6
    spike_multiplier: float = 5.0
    spike_window_days: int = 14
    spike_min_usd: float = 2000
    high_risk_channel_min_usd: float = 3000
    high_risk_formats: list[str] = ["Cash", "Bitcoin"]
    pass_through_min_in_usd: float = 3000
    pass_through_ratio: float = 0.8
    low_value_usd: float = 0.0          # 0 disables: "amount < 0" never happens
    low_value_min_count: int = 1


class LabelsCfg(BaseModel):
    lookback_days: int = 0
    delay_median_days_pos: float = 14
    delay_median_days_neg: float = 3
    delay_sigma: float = 0.6


class GraphCfg(BaseModel):
    twl_days: int = 3
    tws_days: int = 30
    hub_degree_quantile: float = 0.99

    @model_validator(mode="after")
    def _tws_ge_twl(self) -> GraphCfg:
        if self.tws_days < self.twl_days:
            raise ValueError("tws_days must be >= twl_days")
        return self


class GuiltyWalkerCfg(BaseModel):
    n_walks: int = 50
    max_steps: int = 8
    illicit_memory_days: int | None = 21


class GwdCfg(BaseModel):
    enabled: bool = True
    pseudo_label_window_days: int = 14
    pseudo_label_threshold: float = 0.5


class FeaturesCfg(BaseModel):
    profile_windows_days: list[int] = [1, 3, 7, 14, 30]
    graph: GraphCfg = GraphCfg()
    guilty_walker: GuiltyWalkerCfg = GuiltyWalkerCfg()
    gwd: GwdCfg = GwdCfg()


class SplitCfg(BaseModel):
    warmup_days: int = 30
    val_start_day: int = 95
    test_start_day: int = 115
    respect_label_delay: bool = True

    @model_validator(mode="after")
    def _ordered(self) -> SplitCfg:
        if not (0 <= self.warmup_days < self.val_start_day < self.test_start_day):
            raise ValueError("need 0 <= warmup_days < val_start_day < test_start_day")
        return self


class TrainingCfg(BaseModel):
    n_trials: int = 20
    timeout_seconds: int = 240
    feature_selection_importance_share: float = 0.95
    target_fpr: float = 0.20
    selection_min_gain: float = 0.01
    calibration: Literal["auto", "isotonic", "sigmoid"] = "auto"
    isotonic_min_positives: int = 200


class CtganCfg(BaseModel):
    epochs: int = 150
    batch_size: int = 500
    negatives_sample: int = 6000


class AugmentationCfg(BaseModel):
    methods: list[Literal["none", "class_weight", "smote", "ctgan"]] = ["none", "class_weight", "smote", "ctgan"]
    target_positive_share: float = 0.15
    select_ctgan_in_folds: bool = False
    ctgan: CtganCfg = CtganCfg()


class BacktestCfg(BaseModel):
    cuts: list[int] = [75, 95, 115, 130]   # first test day of each fold
    val_days: int = 20
    test_days: int = 20
    bootstrap_reps: int = 300
    seeds: list[int] = [1, 2, 3, 4, 5]
    min_typology_positives: int = 15


class EvaluationCfg(BaseModel):
    backtest: BacktestCfg = BacktestCfg()
    bootstrap_reps: int = 1000
    capacity_alerts_per_day: float | None = None
    feature_sets: list[str] = Field(
        default_factory=lambda: [
            "alert",
            "alert+profiles",
            "alert+profiles+degree",
            "alert+profiles+degree+gw",
            "alert+profiles+degree+gwd",
        ]
    )


class BtlStratum(BaseModel):
    lo_frac_of_tau: float
    hi_frac_of_tau: float
    rate: float


class BtlCfg(BaseModel):
    strata: list[BtlStratum] = [
        BtlStratum(lo_frac_of_tau=0.5, hi_frac_of_tau=1.0, rate=0.10),
        BtlStratum(lo_frac_of_tau=0.0, hi_frac_of_tau=0.5, rate=0.02),
    ]


class PolicyCfg(BaseModel):
    mode: Literal["prioritise", "suppress"] = "suppress"
    never_suppress_rules: list[str] = ["structuring"]
    override_prior_positive: bool = True
    btl: BtlCfg = BtlCfg()


class PathsCfg(BaseModel):
    work_dir: str = "data/work"
    artifacts_dir: str = "artifacts"
    reports_dir: str = "reports"


class Config(BaseModel):
    run_name: str = "synthetic-default"
    seed: int = 42
    data: DataCfg = DataCfg()
    rules: RulesCfg = RulesCfg()
    labels: LabelsCfg = LabelsCfg()
    features: FeaturesCfg = FeaturesCfg()
    split: SplitCfg = SplitCfg()
    training: TrainingCfg = TrainingCfg()
    augmentation: AugmentationCfg = AugmentationCfg()
    evaluation: EvaluationCfg = EvaluationCfg()
    policy: PolicyCfg = PolicyCfg()
    paths: PathsCfg = PathsCfg()

    # convenience
    def work(self, name: str) -> Path:
        p = Path(self.paths.work_dir) / self.run_name
        p.mkdir(parents=True, exist_ok=True)
        return p / name


def _deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _set_dotted(d: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = yaml.safe_load(value) if isinstance(value, str) else value


def load_config(path: str | Path = "configs/default.yaml", profile: str | Path | None = None,
                overrides: list[str] | None = None) -> Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if profile:
        raw = _deep_merge(raw, yaml.safe_load(Path(profile).read_text()) or {})
    for item in overrides or []:
        key, _, val = item.partition("=")
        _set_dotted(raw, key.strip(), val.strip())
    return Config.model_validate(raw)


def dump_config(cfg: Config, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(cfg.model_dump(), sort_keys=False))
