"""V2 - Calendar features + linear regression with a month-end-week gate; selected only on earlier validation weeks.

V2 changes: (1) candidates include a gated variant that predicts 0 outside the week containing the last day of the
month (in the audited data, 94.8% of positive rows and 97% of units fall in that week); (2) wider ridge grid;
(3) default holdout = 44 weeks (2013-07-01 .. last Reseller order week).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy import sparse
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import promotion_lr as original

CALENDAR_NAMES = ['MonthEnd', 'MonthStart', 'MonthEndMinus1', 'MonthEndMinus2', 'MonthEndMinus3']
CANDIDATES = [('original', None, None, False)] + [
    (f'{mode}_{alpha}{"_gated" if gated else ""}', mode, alpha, gated)
    for mode in ('additive', 'interaction') for alpha in (0, 1, 10, 100, 300) for gated in (False, True)
]
GATE_RULE = 'Predict 0 unless the week contains the last calendar day of a month (feature MonthEnd > 0).'


def final_predict(fitted, df, gated):
    """Non-negative prediction; optionally forced to 0 outside the month-end week. Uses dates only."""
    predicted = np.maximum(0, raw_predict(fitted, df))
    if gated:
        predicted = predicted * (calendar_features(df)['MonthEnd'].to_numpy() > 0)
    return predicted


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def calendar_features(df):
    """Dates only. No quantities, fitted parameters, lagged or future outcomes."""
    dates = pd.DatetimeIndex(df['WeekStart'])
    values = {name: np.zeros(len(df)) for name in CALENDAR_NAMES}
    for offset in range(7):
        days = dates + pd.Timedelta(days=offset)
        values['MonthEnd'] += days.is_month_end.astype(float)
        values['MonthStart'] += days.is_month_start.astype(float)
        for n in (1, 2, 3):
            values[f'MonthEndMinus{n}'] += (days + pd.Timedelta(days=n)).is_month_end.astype(float)
    return pd.DataFrame(values, index=df.index)


def categorical_features(df):
    x = original.features(df)[original.CATEGORIES].copy()
    x['Month'] = df['WeekStart'].dt.month.astype(str)
    return x


def matrix(df, design=None, mode='interaction'):
    if design is None:
        encoder = OneHotEncoder(handle_unknown='ignore')
        oh = encoder.fit_transform(categorical_features(df))
        scaler = StandardScaler()
        numerical = scaler.fit_transform(original.features(df)[original.NUMBERS])
        design = {'encoder': encoder, 'scaler': scaler, 'mode': mode}
    else:
        oh = design['encoder'].transform(categorical_features(df))
        numerical = design['scaler'].transform(original.features(df)[original.NUMBERS])
    cal = calendar_features(df).to_numpy()
    blocks = [oh, sparse.csr_matrix(numerical), sparse.csr_matrix(cal)]
    if design['mode'] == 'interaction':
        blocks += [oh.multiply(cal[:, j, None]) for j in range(cal.shape[1])]
    return sparse.hstack(blocks, format='csr'), design


def fit_candidate(df, candidate):
    name, mode, alpha = candidate[:3]
    if name == 'original':
        return {'name': name, 'estimator': original.model().fit(original.features(df), df.UnitsSold)}
    x, design = matrix(df, mode=mode)
    estimator = LinearRegression(tol=1e-8) if alpha == 0 else Ridge(alpha=alpha, solver='lsqr', tol=1e-8)
    estimator.fit(x, df.UnitsSold)
    return {'name': name, 'design': design, 'estimator': estimator}


def raw_predict(fitted, df):
    if fitted['name'] == 'original':
        return fitted['estimator'].predict(original.features(df))
    return fitted['estimator'].predict(matrix(df, design=fitted['design'])[0])


def calendar_baseline(train, test):
    """Training-only product/territory mean, conditional on a month-boundary week."""
    earlier, later = train.copy(), test.copy()
    for frame in (earlier, later):
        frame['MonthBoundary'] = calendar_features(frame)[['MonthEnd', 'MonthStart']].max(axis=1).astype(int)
    means = earlier.groupby(['ProductID', 'TerritoryID', 'MonthBoundary']).UnitsSold.mean()
    predicted = np.array([means.get((r.ProductID, r.TerritoryID, r.MonthBoundary), np.nan)
                          for r in later.itertuples()])
    missing = np.isnan(predicted)
    predicted[missing] = original.baseline(train, test)[missing]
    return predicted


def validation_select(train, out, validation_weeks=13, validation_folds=3):
    weeks = train.WeekStart.drop_duplicates().sort_values().reset_index(drop=True)
    if len(weeks) - validation_weeks * validation_folds < 26:
        raise ValueError('Need at least 26 earlier training weeks before the validation folds.')
    plan = {
        'selection_metric': 'pooled chronological validation RMSE on clipped nonnegative predictions',
        'validation_weeks_per_fold': validation_weeks, 'validation_folds': validation_folds,
        'candidates': [c[0] for c in CANDIDATES],
        'gate_rule': GATE_RULE,
        'calendar_features': CALENDAR_NAMES,
        'calendar_interactions': 'each calendar count times each onehot ProductID/Category/TerritoryID/Month',
        'features_available_at_forecast_origin': True,
        'lags_or_future_targets_used': False,
        'selection_train_last_week': str(train.WeekStart.max().date()),
        'heldout_note': 'The original 52-week holdout has already been inspected. Revised evaluation is retrospective and is not a fresh independent test.',
    }
    write_json(out / 'selection_plan.json', plan)
    rows = []
    for candidate in CANDIDATES:
        ys, ps, folds = [], [], []
        for i in range(validation_folds):
            end = len(weeks) - (validation_folds - 1 - i) * validation_weeks
            start = end - validation_weeks
            earlier = train[train.WeekStart < weeks.iloc[start]]
            later = train[(train.WeekStart >= weeks.iloc[start]) & (train.WeekStart <= weeks.iloc[end - 1])]
            fitted = fit_candidate(earlier, candidate)
            predicted = final_predict(fitted, later, candidate[3])
            ys.extend(later.UnitsSold)
            ps.extend(predicted)
            folds.append({
                'train_through': str(earlier.WeekStart.max().date()),
                'validation_start': str(later.WeekStart.min().date()),
                'validation_end': str(later.WeekStart.max().date()),
                'train_rows': len(earlier), 'validation_rows': len(later),
                **original.score(later.UnitsSold, predicted),
            })
        row = {'candidate': candidate[0], 'mode': candidate[1], 'alpha': candidate[2], 'gated': candidate[3],
               **original.score(ys, ps), 'folds': folds}
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != 'folds'}), flush=True)
    # Tie rule: candidates within 0.1% of the best pooled RMSE are practically equal; prefer the most regularised
    # (largest ridge alpha, gated first) so an OLS fit is not chosen over a ridge fit for a 1e-5 difference.
    best_rmse = min(r['RMSE'] for r in rows)
    near = [r for r in rows if r['RMSE'] <= best_rmse * 1.001]
    winner = max(near, key=lambda r: (r['alpha'] or 0, r['gated']))
    plan['tie_rule'] = 'Within 0.1% of the lowest pooled validation RMSE, choose the largest ridge alpha.'
    write_json(out / 'selection_plan.json', plan)
    write_json(out / 'validation_results.json', {'plan': plan, 'results': rows, 'selected': winner['candidate'], 'lowest_rmse_candidate': min(rows, key=lambda r: r['RMSE'])['candidate']})
    return next(c for c in CANDIDATES if c[0] == winner['candidate'])


def train(args):
    df = original.load_csv(args.csv, True)
    weeks = df.WeekStart.drop_duplicates().sort_values().reset_index(drop=True)
    if args.test_weeks < 1 or len(weeks) - args.test_weeks < 65:
        raise ValueError('Need at least 65 earlier observed weeks before the requested test period.')
    cutoff = weeks.iloc[-args.test_weeks]
    earlier, later = df[df.WeekStart < cutoff], df[df.WeekStart >= cutoff]
    if earlier.ScheduledCampaignDiscountPct.nunique() < 2:
        raise ValueError('Training needs multiple observed campaign discount rates.')
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output folder is not empty. Choose a NEW output folder; existing results are preserved.')
    out.mkdir(parents=True, exist_ok=True)
    chosen = validation_select(earlier, out)
    # Selection is complete before producing any revised holdout predictions.
    evaluation = fit_candidate(earlier, chosen)
    raw = raw_predict(evaluation, later)
    predicted = final_predict(evaluation, later, chosen[3])
    naive = original.baseline(earlier, later)
    calendar_naive = calendar_baseline(earlier, later)
    old_model = fit_candidate(earlier, CANDIDATES[0])
    old_predicted = np.maximum(0, raw_predict(old_model, later))
    metrics = {
        'algorithm': 'Ordinary Least Squares' if chosen[2] == 0 or chosen[0] == 'original' else 'Ridge linear regression',
        'selected_candidate': chosen[0], 'ridge_alpha': chosen[2], 'gated_to_month_end_week': chosen[3], 'gate_rule': GATE_RULE,
        'selection': '3 x 13-week expanding-window validation, entirely before holdout; lowest pooled RMSE',
        'target': 'UnitsSold', 'grain': 'ProductID x WeekStart x TerritoryID; Reseller only',
        'train_rows': len(earlier), 'test_rows': len(later),
        'train_through': str(earlier.WeekStart.max().date()),
        'test_from': str(later.WeekStart.min().date()), 'test_through': str(later.WeekStart.max().date()),
        'LR_nonnegative': original.score(later.UnitsSold, predicted),  # gated when the selected candidate is gated
        'LR_raw': original.score(later.UnitsSold, raw),
        'original_LR': original.score(later.UnitsSold, old_predicted),
        'historical_mean_baseline': original.score(later.UnitsSold, naive),
        'calendar_mean_baseline': original.score(later.UnitsSold, calendar_naive),
        'always_zero_baseline': original.score(later.UnitsSold, np.zeros(len(later))),
        'negative_raw_predictions': int((raw < 0).sum()),
        'data_origin': sorted(df.DataOrigin.dropna().astype(str).unique().tolist()),
        'test_evaluation_note': 'Previously inspected original holdout, same dates and rows. Retrospective development comparison, NOT a new untouched test.',
        'multi_step_evaluation': 'No lag/rolling quantities or holdout outcomes in features. Every holdout calendar date is known at train cutoff.',
        'scope_note': 'The same product eligibility, zero-fill, historical category and campaign-calendar assumptions as the original dataset remain.',
        'interpretation': 'Predictive association. Calendar structure may reflect the sample database generation process. Not causal promotion lift or optimal discount.',
    }
    for label, mask in [('campaign_active', later.ScheduledCampaignDiscountPct.gt(0).to_numpy()),
                        ('no_qualifying_campaign', later.ScheduledCampaignDiscountPct.eq(0).to_numpy())]:
        metrics[label] = {'rows': int(mask.sum()),
                          'LR': original.score(later.loc[mask, 'UnitsSold'], predicted[mask]) if mask.any() else None,
                          'original_LR': original.score(later.loc[mask, 'UnitsSold'], old_predicted[mask]) if mask.any() else None}
    result = later.copy()
    result['RawPredictedUnits'] = raw
    result['PredictedUnits'] = predicted
    result['BaselineUnits'] = naive
    result['CalendarBaselineUnits'] = calendar_naive
    result['OriginalPredictedUnits'] = old_predicted
    result.to_csv(out / 'holdout_predictions.csv', index=False, date_format='%Y-%m-%d')
    pd.DataFrame([{**r} for r in [
        {'Model': 'Original OLS', **metrics['original_LR']},
        {'Model': 'Selected calendar linear model', **metrics['LR_nonnegative']},
        {'Model': 'Historical mean', **metrics['historical_mean_baseline']},
        {'Model': 'Calendar-conditioned historical mean', **metrics['calendar_mean_baseline']},
        {'Model': 'Always zero', **metrics['always_zero_baseline']},
    ]]).to_csv(out / 'model_comparison.csv', index=False)
    fitted = fit_candidate(df, chosen)
    version = f"calendar_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{hashlib.sha256(Path(args.csv).read_bytes()).hexdigest()[:10]}"
    info = {
        'model_version': version, 'selected_candidate': chosen[0], 'gated': chosen[3], 'gate_rule': GATE_RULE, 'algorithm': metrics['algorithm'],
        'trained_through': str(df.WeekStart.max().date()),
        'known_categories': {c: sorted(df[c].unique().tolist()) for c in original.CATEGORIES},
        'product_category': df.groupby('ProductID').Category.unique().map(list).to_dict(),
        'discount_min': float(df.ScheduledCampaignDiscountPct.min()),
        'discount_max': float(df.ScheduledCampaignDiscountPct.max()),
        'discounts_observed_per_product': df.groupby('ProductID').ScheduledCampaignDiscountPct.unique().map(lambda v: sorted(map(float,v))).to_dict(),
        'data_origin': metrics['data_origin'], 'sklearn_version': sklearn.__version__, 'python_version': platform.python_version(),
        'training_csv_sha256': hashlib.sha256(Path(args.csv).read_bytes()).hexdigest(),
        'features': 'Original features + month onehot + calendar counts and selected interactions. No lagged quantities.',
        'forecast_weeks_must_follow_training': True,
    }
    joblib.dump({**info, 'fitted': fitted}, out / 'model.joblib')
    # Keep the actual pre-holdout fitted model for independent out-of-sample verification.
    joblib.dump({'fitted': evaluation, 'train_through': metrics['train_through'],
                 'sklearn_version': sklearn.__version__}, out / 'evaluation_model.joblib')
    write_json(out / 'model_info.json', info)
    write_json(out / 'metrics.json', metrics)
    print(json.dumps({'selected': chosen[0], 'holdout': metrics['LR_nonnegative'], 'original': metrics['original_LR']}), flush=True)


def predict(args):
    df = original.load_csv(args.csv, False)
    if not 1 <= args.run_id <= 2147483647:
        raise ValueError('run-id must be a positive SQL INT.')
    bundle = joblib.load(args.model)
    if bundle['sklearn_version'] != sklearn.__version__:
        raise ValueError('Use the same scikit-learn version as the saved model.')
    if df.WeekStart.le(pd.Timestamp(bundle['trained_through'])).any():
        raise ValueError('Forecast weeks must follow the last fitted training week.')
    for c in original.CATEGORIES:
        unknown = sorted(set(df[c]) - set(bundle['known_categories'][c]))
        if unknown:
            raise ValueError(f'Unseen {c}: {unknown[:5]}')
    for r in df.itertuples():
        if r.Category not in bundle['product_category'][r.ProductID]:
            raise ValueError(f'Category does not match observed product {r.ProductID}.')
        observed = bundle['discounts_observed_per_product'][r.ProductID]
        if not any(np.isclose(r.ScheduledCampaignDiscountPct, x, rtol=0, atol=1e-12) for x in observed):
            raise ValueError(f'Discount not observed for product {r.ProductID}. Supported rates: {observed}')
    raw = raw_predict(bundle['fitted'], df)
    final = final_predict(bundle['fitted'], df, bundle['gated'])
    result = df.drop(columns=['UnitsSold'], errors='ignore').copy()
    result['ForecastRunID'] = args.run_id
    result['RawPredictedUnits'] = raw
    result['PredictedUnits'] = final
    result['ModelVersion'] = bundle['model_version']
    result['PredictedAt'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    result['TrainingDataOrigin'] = ';'.join(bundle['data_origin'])
    path = Path(args.out)
    if path.exists():
        raise ValueError('Prediction output already exists. Choose a new filename.')
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False, date_format='%Y-%m-%d')
    print(f'Saved {len(result)} forecasts to {path.resolve()}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('train')
    p.add_argument('--csv', default='prepared/training.csv')
    p.add_argument('--out', default='outputs_calendar')
    p.add_argument('--test-weeks', type=int, default=44)
    p.set_defaults(func=train)
    p = sub.add_parser('predict')
    p.add_argument('--csv', required=True)
    p.add_argument('--model', default='outputs_calendar/model.joblib')
    p.add_argument('--out', default='outputs_calendar/forecasts_calendar.csv')
    p.add_argument('--run-id', type=int, required=True)
    p.set_defaults(func=predict)
    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, OSError, KeyError) as exc:
        print(f'ERROR: {exc}')
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
