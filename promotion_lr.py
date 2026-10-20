"""Weekly CompanyX unit forecasts. Retrospective campaign-calendar experiment; assumptions are documented in README."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORIES = ["ProductID", "Category", "TerritoryID"]
NUMBERS = ["ScheduledCampaignDiscountPct", "Year", "MonthSin", "MonthCos"]
REQUIRED = ["WeekStart", *CATEGORIES, "ScheduledCampaignDiscountPct"]
GRAIN = ["WeekStart", "ProductID", "TerritoryID"]


def load_csv(path: str, training: bool) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={c: "string" for c in CATEGORIES})
    required = REQUIRED + (["UnitsSold"] if training else [])
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    if df.empty:
        raise ValueError("CSV has no data. Templates contain headers only.")
    for c in CATEGORIES:
        df[c] = df[c].str.strip()
        if df[c].isna().any() or df[c].eq("").any():
            raise ValueError(f"{c} contains missing or empty values.")
    df["WeekStart"] = pd.to_datetime(df["WeekStart"], format="%Y-%m-%d", errors="coerce")
    if df["WeekStart"].isna().any() or df["WeekStart"].dt.dayofweek.ne(0).any():
        raise ValueError("WeekStart must be a Monday, formatted YYYY-MM-DD.")
    for c in ["ScheduledCampaignDiscountPct"] + (["UnitsSold"] if training else []):
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if not np.isfinite(df[c].to_numpy(dtype=float)).all():
            raise ValueError(f"{c} contains missing or non-finite values.")
    if not df["ScheduledCampaignDiscountPct"].between(0, 1).all():
        raise ValueError("ScheduledCampaignDiscountPct must be a fraction: 0.10 means 10%, not 10.")
    if training and ((df["UnitsSold"] < 0).any() or df["UnitsSold"].mod(1).ne(0).any()):
        raise ValueError("UnitsSold must contain nonnegative whole numbers.")
    grain = GRAIN if training else GRAIN + ["ScheduledCampaignDiscountPct"]
    for c in ("ProposalID", "ProposalKey"):
        if not training and c in df.columns:
            if df[c].isna().any():
                raise ValueError(f"{c} contains missing values.")
            grain = grain + [c]
    if df.duplicated(grain).any():
        raise ValueError(f"Duplicate rows at the declared grain: {grain}")
    if not training and "ProposalKey" in df.columns:
        if df.groupby("ProposalKey")["ScheduledCampaignDiscountPct"].nunique().gt(1).any():
            raise ValueError("One ProposalKey version cannot have multiple discount rates.")
    return df.sort_values(GRAIN).reset_index(drop=True)


def features(df: pd.DataFrame) -> pd.DataFrame:
    """Do not include UnitsSold, monetary totals, or warehouse row keys in X."""
    x = df[CATEGORIES + ["ScheduledCampaignDiscountPct"]].copy()
    x[CATEGORIES] = x[CATEGORIES].astype(str)
    x["Year"] = df["WeekStart"].dt.year - 2000
    month = df["WeekStart"].dt.month
    x["MonthSin"] = np.sin(2 * np.pi * (month - 1) / 12)
    x["MonthCos"] = np.cos(2 * np.pi * (month - 1) / 12)
    return x


def model() -> Pipeline:
    prep = ColumnTransformer([
        ("category", OneHotEncoder(handle_unknown="ignore"), CATEGORIES),
        ("number", StandardScaler(), NUMBERS),
    ], sparse_threshold=1.0)
    return Pipeline([("prepare", prep), ("regression", LinearRegression())])


def score(actual, predicted) -> dict:
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)) if len(actual) > 1 and np.var(actual) > 0 else None,
    }


def baseline(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Historical group mean using training rows only; global fallback for new groups."""
    means = train.groupby(["ProductID", "TerritoryID"], as_index=False)["UnitsSold"].mean()
    merged = test[["ProductID", "TerritoryID"]].merge(
        means, on=["ProductID", "TerritoryID"], how="left", validate="many_to_one", sort=False
    )
    return merged["UnitsSold"].fillna(train["UnitsSold"].mean()).to_numpy()


def train(args) -> None:
    df = load_csv(args.csv, training=True)
    weeks = df["WeekStart"].drop_duplicates().sort_values()
    if args.test_weeks < 1 or len(weeks) - args.test_weeks < 4:
        raise ValueError("Need at least 4 training weeks plus the requested test weeks.")
    cutoff = weeks.iloc[-args.test_weeks]
    earlier, later = df[df["WeekStart"] < cutoff], df[df["WeekStart"] >= cutoff]
    if earlier["ScheduledCampaignDiscountPct"].nunique() < 2:
        raise ValueError("Training period needs at least two verified discount levels for this promotion model.")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise ValueError("Output folder is not empty. Choose another folder or use --overwrite.")
    out.mkdir(parents=True, exist_ok=True)

    evaluation_model = model().fit(features(earlier), earlier["UnitsSold"])
    raw = evaluation_model.predict(features(later))
    predicted = np.maximum(0.0, raw)
    naive = baseline(earlier, later)
    unknown = {c: sorted(set(later[c]) - set(earlier[c])) for c in CATEGORIES}
    origin = sorted(df["DataOrigin"].dropna().astype(str).unique().tolist()) if "DataOrigin" in df else ["USER_DATA_NOT_INDEPENDENTLY_VERIFIED"]
    metrics = {
        "data_origin": origin,
        "target": "UnitsSold",
        "grain": "ProductID x WeekStart x TerritoryID; Reseller only",
        "feature_note": "ScheduledCampaignDiscountPct is a catalogue calendar signal for unconditional Reseller campaigns. It is not an applied or volume-tier discount.",
        "train_rows": len(earlier), "test_rows": len(later),
        "train_through": earlier["WeekStart"].max().date().isoformat(),
        "test_from": later["WeekStart"].min().date().isoformat(),
        "test_through": later["WeekStart"].max().date().isoformat(),
        "LR_nonnegative": score(later["UnitsSold"], predicted),
        "LR_raw": score(later["UnitsSold"], raw),
        "historical_mean_baseline": score(later["UnitsSold"], naive),
        "always_zero_baseline": score(later["UnitsSold"], np.zeros(len(later))),
        "negative_raw_predictions": int((raw < 0).sum()),
        "unseen_categories_in_holdout": unknown,
        "evaluation_note": "Holdout scores use a model trained only on earlier weeks. The saved forecast model is refit on all supplied completed weeks.",
        "interpretation": "Predictive association only. Scenario differences are not causal promotion lift or profit.",
    }
    for label, mask in [("campaign_active", later["ScheduledCampaignDiscountPct"].gt(0).to_numpy()),
                        ("no_qualifying_campaign", later["ScheduledCampaignDiscountPct"].eq(0).to_numpy())]:
        metrics[label] = {"rows": int(mask.sum()),
                          "weeks": int(later.loc[mask, "WeekStart"].nunique()),
                          "LR": score(later.loc[mask, "UnitsSold"], predicted[mask]) if mask.any() else None,
                          "baseline": score(later.loc[mask, "UnitsSold"], naive[mask]) if mask.any() else None}
    metrics["LR_beats_baseline_RMSE"] = metrics["LR_nonnegative"]["RMSE"] < metrics["historical_mean_baseline"]["RMSE"]
    pd.DataFrame({**{c: later[c].to_numpy() for c in later.columns},
                  "RawPredictedUnits": raw, "PredictedUnits": predicted,
                  "BaselineUnits": naive}).to_csv(out / "holdout_predictions.csv", index=False, date_format="%Y-%m-%d")
    fitted = model().fit(features(df), df["UnitsSold"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fingerprint = hashlib.sha256(Path(args.csv).read_bytes()).hexdigest()[:10]
    bundle = {
        "pipeline": fitted, "model_version": f"lr_{stamp}_{fingerprint}",
        "trained_through": df["WeekStart"].max().date().isoformat(),
        "known_categories": {c: sorted(df[c].unique().tolist()) for c in CATEGORIES},
        "discount_min": float(df["ScheduledCampaignDiscountPct"].min()),
        "discount_max": float(df["ScheduledCampaignDiscountPct"].max()),
        "data_origin": origin, "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
    }
    joblib.dump(bundle, out / "model.joblib")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    (out / "model_info.json").write_text(json.dumps({k: v for k, v in bundle.items() if k != "pipeline"}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"model_version": bundle["model_version"], "LR": metrics["LR_nonnegative"],
                      "baseline": metrics["historical_mean_baseline"], "data_origin": origin}, ensure_ascii=False))
    print(f"Saved model and evaluation files in {out.resolve()}")


def predict(args) -> None:
    df = load_csv(args.csv, training=False)
    if not 1 <= args.run_id <= 2147483647:
        raise ValueError("run-id must be a positive SQL INT. Assign a distinct ID to each forecast batch.")
    bundle = joblib.load(args.model)
    if bundle["sklearn_version"] != sklearn.__version__:
        raise ValueError("Use the scikit-learn version used to train this model.")
    if df["WeekStart"].le(pd.Timestamp(bundle["trained_through"])).any():
        raise ValueError("Forecast weeks must be later than the last week used to fit the saved model.")
    for c in CATEGORIES:
        unknown = sorted(set(df[c]) - set(bundle["known_categories"][c]))
        if unknown:
            raise ValueError(f"Unseen {c}: {unknown[:5]}. Add suitable training data and retrain.")
    if not df["ScheduledCampaignDiscountPct"].between(bundle["discount_min"], bundle["discount_max"]).all():
        raise ValueError("Requested discount is outside the observed training range.")
    raw = bundle["pipeline"].predict(features(df))
    result = df.drop(columns=["UnitsSold"], errors="ignore").copy()
    result["ForecastRunID"] = args.run_id
    result["RawPredictedUnits"] = raw
    result["PredictedUnits"] = np.maximum(0.0, raw)
    result["ModelVersion"] = bundle["model_version"]
    result["PredictedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    result["TrainingDataOrigin"] = ";".join(bundle["data_origin"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False, date_format="%Y-%m-%d")
    print(f"Saved {len(result)} forecasts to {out.resolve()}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train", help="Train and evaluate with a chronological holdout.")
    p.add_argument("--csv", required=True)
    p.add_argument("--out", default="outputs")
    p.add_argument("--test-weeks", type=int, default=52)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=train)
    p = sub.add_parser("predict", help="Predict future weekly units for planned scenarios.")
    p.add_argument("--csv", required=True)
    p.add_argument("--model", default="outputs/model.joblib")
    p.add_argument("--out", default="outputs/forecasts.csv")
    p.add_argument("--run-id", required=True, type=int)
    p.set_defaults(func=predict)
    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, OSError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
