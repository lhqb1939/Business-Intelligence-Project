"""V2 - Independent data reconciliation, evaluation checks, month-end gate and forecast guard tests."""
import argparse
import calendar
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import promotion_lr as original
import promotion_lr_calendar_v2 as revised
from prepare_dwh_data_v2 import validate_warehouse


def verify(root, prepared, results):
    audit = json.loads((prepared / 'data_audit.json').read_text(encoding='utf-8'))
    metrics = json.loads((results / 'metrics.json').read_text(encoding='utf-8'))
    for record in audit['source_files']:
        assert hashlib.sha256((root / 'inputs' / record['name']).read_bytes()).hexdigest() == record['sha256']
    sales = pd.read_csv(root / 'inputs/warehouse_sales_lines_fixed.csv', na_values=['NULL'])
    sales['OrderDate'] = pd.to_datetime(sales.OrderDate)
    loaded = validate_warehouse(sales)
    assert int(loaded.sum()) == audit['loaded_warehouse_rows']
    invalid = sales.copy()
    invalid.loc[loaded.idxmax(), 'WarehouseProductID'] = -99999
    try:
        validate_warehouse(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid warehouse key was accepted')
    invalid = sales.copy()
    invalid.loc[(~loaded).idxmax(), 'OrderQty'] = 1
    try:
        validate_warehouse(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError('Missing-fact source fallback was accepted')

    data = original.load_csv(str(prepared / 'training.csv'), True)
    quarantine = pd.read_csv(prepared / 'quarantined_sales_lines.csv')
    bad = set(zip(quarantine.SalesOrderID, quarantine.SalesOrderDetailID))
    quantities = {}
    for row in sales.itertuples():
        if not row.WarehousePresent or row.CustomerType != 'Reseller' or row.OrderStatus != 5:
            continue
        if (row.SalesOrderID, row.SalesOrderDetailID) in bad:
            continue
        key = ((row.OrderDate - pd.Timedelta(days=row.OrderDate.dayofweek)), str(row.ProductID), str(row.TerritoryID))
        quantities[key] = quantities.get(key, 0) + row.OrderQty
    for row in data.itertuples():
        assert row.UnitsSold == quantities.get((row.WeekStart, row.ProductID, row.TerritoryID), 0)
    assert len(data) == audit['prepared_rows'] and data.UnitsSold.sum() == audit['prepared_total_units']

    # Independently check month/year/leap-year boundaries using Python's calendar.
    probes = pd.DataFrame({'WeekStart': pd.to_datetime(['2012-02-27', '2013-12-30', '2014-07-07', '2014-07-28'])})
    actual_cal = revised.calendar_features(probes)
    expected = []
    for week in probes.WeekStart:
        days = [week + pd.Timedelta(days=i) for i in range(7)]
        expected.append([
            sum(d.day == calendar.monthrange(d.year, d.month)[1] for d in days),
            sum(d.day == 1 for d in days),
            *[sum(d.day == calendar.monthrange(d.year, d.month)[1] - n for d in days) for n in (1, 2, 3)],
        ])
    np.testing.assert_array_equal(actual_cal.to_numpy(), expected)

    training = data[data.WeekStart <= pd.Timestamp(metrics['train_through'])]
    testing = data[data.WeekStart >= pd.Timestamp(metrics['test_from'])]
    assert training.WeekStart.max() < testing.WeekStart.min()
    holdout = pd.read_csv(results / 'holdout_predictions.csv', dtype={c: 'string' for c in original.CATEGORIES}, parse_dates=['WeekStart'])
    pd.testing.assert_frame_equal(testing[list(data.columns)].reset_index(drop=True), holdout[list(data.columns)].reset_index(drop=True))
    assert len(testing) == metrics['test_rows'] and len(training) == metrics['train_rows']
    validation = json.loads((results / 'validation_results.json').read_text(encoding='utf-8'))
    lowest = min(validation['results'], key=lambda r: r['RMSE'])
    winner = next(r for r in validation['results'] if r['candidate'] == validation['selected'])
    assert winner['candidate'] == metrics['selected_candidate'] == validation['selected']
    assert winner['RMSE'] <= lowest['RMSE'] * 1.001 + 1e-12, 'selected candidate is outside the documented tie tolerance'
    # V2 data checks: grid basis and source reconciliation.
    rec = audit['reconciliation_to_source']
    assert rec['units_in_training'] == int(data.UnitsSold.sum()) and rec['share_not_in_training'] < 0.02
    reseller_dates = sales.loc[(sales.CustomerType == 'Reseller') & (sales.OrderStatus == 5), 'OrderDate']
    last_week = reseller_dates.max() - pd.Timedelta(days=reseller_dates.max().dayofweek)
    assert data.WeekStart.max() == last_week, 'grid must end at the last Reseller order week'
    assert data.WeekStart.min() == reseller_dates.min() - pd.Timedelta(days=reseller_dates.min().dayofweek)
    for result in validation['results']:
        for fold in result['folds']:
            assert pd.Timestamp(fold['train_through']) < pd.Timestamp(fold['validation_start'])
            assert pd.Timestamp(fold['validation_end']) < testing.WeekStart.min()

    evaluation = joblib.load(results / 'evaluation_model.joblib')
    assert evaluation['train_through'] == metrics['train_through']
    selected = next(c for c in revised.CANDIDATES if c[0] == winner['candidate'])
    assert metrics['gated_to_month_end_week'] == selected[3]
    # Refit from earlier weeks to prove that the evaluated model excludes test targets.
    checked = revised.fit_candidate(training, selected)
    predicted = revised.raw_predict(checked, testing)
    np.testing.assert_allclose(predicted, revised.raw_predict(evaluation['fitted'], testing), atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(holdout.RawPredictedUnits, predicted, atol=1e-7, rtol=1e-7)
    mutated = testing.copy()
    mutated['UnitsSold'] = 999999
    mutated['NetSales'] = 999999
    mutated['DiscountAmount'] = 999999
    np.testing.assert_array_equal(revised.raw_predict(checked, testing), revised.raw_predict(checked, mutated))
    np.testing.assert_allclose(holdout.PredictedUnits, revised.final_predict(checked, testing, selected[3]), atol=1e-12)
    if selected[3]:
        outside = revised.calendar_features(holdout)['MonthEnd'].to_numpy() == 0
        assert (holdout.loc[outside, 'PredictedUnits'] == 0).all(), 'gate must force 0 outside the month-end week'
        assert (holdout.loc[~outside, 'PredictedUnits'] > 0).any()
    np.testing.assert_allclose(holdout.BaselineUnits, original.baseline(training, testing), atol=1e-12)
    np.testing.assert_allclose(holdout.CalendarBaselineUnits, revised.calendar_baseline(training, testing), atol=1e-12)
    y = holdout.UnitsSold.to_numpy(dtype=float)
    for key, column in [('LR_nonnegative', 'PredictedUnits'), ('LR_raw', 'RawPredictedUnits'),
                        ('original_LR', 'OriginalPredictedUnits'), ('historical_mean_baseline', 'BaselineUnits'),
                        ('calendar_mean_baseline', 'CalendarBaselineUnits')]:
        error = y - holdout[column].to_numpy()
        values = [np.abs(error).mean(), np.sqrt(np.square(error).mean()), 1 - np.square(error).sum() / np.square(y - y.mean()).sum()]
        np.testing.assert_allclose(values, [metrics[key][c] for c in ('MAE', 'RMSE', 'R2')], atol=1e-10, rtol=0)
    bundle = joblib.load(results / 'model.joblib')
    assert bundle['training_csv_sha256'] == hashlib.sha256((prepared / 'training.csv').read_bytes()).hexdigest()
    assert bundle['trained_through'] == str(data.WeekStart.max().date())
    assert bundle['selected_candidate'] == winner['candidate']
    with tempfile.TemporaryDirectory(prefix='companyx-calendar-verify-') as folder:
        case = Path(folder) / 'case.csv'
        output = Path(folder) / 'forecast.csv'
        example = data.iloc[0]
        future = pd.DataFrame([{'WeekStart': str((data.WeekStart.max() + pd.Timedelta(days=14)).date()),
                               **{c: str(example[c]) for c in original.CATEGORIES}, 'ScheduledCampaignDiscountPct': 0.0}])
        command = [sys.executable, str(root / 'promotion_lr_calendar_v2.py'), 'predict', '--csv', str(case),
                   '--model', str(results / 'model.joblib'), '--out', str(output), '--run-id', '100002']
        future.to_csv(case, index=False)
        success = subprocess.run(command, capture_output=True, text=True)
        assert success.returncode == 0, success.stdout + success.stderr
        forecast = pd.read_csv(output)
        assert forecast.PredictedUnits.ge(0).all() and np.isfinite(forecast.PredictedUnits).all()
        assert forecast.ModelVersion.eq(bundle['model_version']).all()
        assert forecast.ForecastRunID.eq(100002).all()
        cases = [
            ('TerritoryID', '999999', 'Unseen TerritoryID'),
            ('WeekStart', bundle['trained_through'], 'Forecast weeks must follow'),
            ('ScheduledCampaignDiscountPct', 0.99, 'Discount not observed'),
            ('Category', 'Clothing', 'Unseen Category'),
        ]
        for column, value, message in cases:
            bad = future.copy()
            bad[column] = value
            bad.to_csv(case, index=False)
            failure = subprocess.run(command, capture_output=True, text=True)
            assert failure.returncode == 2 and message in failure.stdout, failure.stdout + failure.stderr
        future.to_csv(case, index=False)
        failure = subprocess.run(command, capture_output=True, text=True)
        assert failure.returncode == 2 and 'already exists' in failure.stdout
    print('Verified: original data hashes/targets/warehouse keys; calendar boundaries; chronological validation selection; pre-holdout refit; unchanged features after target mutation; independent metrics; baselines; saved model prediction guards.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', default='prepared')
    parser.add_argument('--results', default='results')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    verify(root, (root / args.prepared).resolve(), (root / args.results).resolve())
