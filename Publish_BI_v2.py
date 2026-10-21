"""Publish consistent V2 outputs for Power BI; optionally import the forecast into SQL."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(root, prepared, results, page1):
    import joblib
    import numpy as np
    import pandas as pd
    import sklearn
    sys.path.insert(0, str(root))
    import promotion_lr as original
    import promotion_lr_calendar_v2 as revised
    import Expand_Page1_Calendar_v2 as helper

    info = read_json(results / 'model_info.json')
    metrics = read_json(results / 'metrics.json')
    summary = read_json(page1 / 'SCENARIO_INFO.json')
    bundle = joblib.load(results / 'model.joblib')
    if bundle['sklearn_version'] != sklearn.__version__:
        raise ValueError('Use the same scikit-learn version as the fitted model.')
    if info['model_version'] != bundle['model_version'] or summary['model_version'] != bundle['model_version']:
        raise ValueError('Forecast and model come from different runs. Regenerate Page 1 with this model.')
    if digest(prepared / 'training.csv') != bundle['training_csv_sha256']:
        raise ValueError('Prepared training.csv does not match model.joblib.')
    if metrics['selected_candidate'] != bundle['selected_candidate']:
        raise ValueError('metrics.json and model.joblib select different models.')

    training = original.load_csv(str(prepared / 'training.csv'), True)
    holdout = original.load_csv(str(results / 'holdout_predictions.csv'), True)
    expected = training.loc[training.WeekStart >= pd.Timestamp(metrics['test_from'])].reset_index(drop=True)
    pd.testing.assert_frame_equal(expected, holdout[list(training.columns)], check_dtype=False, atol=1e-10, rtol=0)
    if len(holdout) != metrics['test_rows'] or len(training) - len(holdout) != metrics['train_rows']:
        raise ValueError('Holdout row counts do not match metrics.json.')
    evaluation = joblib.load(results / 'evaluation_model.joblib')
    if evaluation['train_through'] != metrics['train_through']:
        raise ValueError('Evaluation model and metrics use different cutoffs.')
    predicted = revised.final_predict(evaluation['fitted'], holdout, bundle['gated'])
    np.testing.assert_allclose(holdout.PredictedUnits, predicted, atol=1e-7, rtol=1e-7)

    comparison = pd.read_csv(results / 'model_comparison.csv')
    names = {'Original OLS': ('original_LR', 'OriginalPredictedUnits'),
             'Selected calendar linear model': ('LR_nonnegative', 'PredictedUnits'),
             'Historical mean': ('historical_mean_baseline', 'BaselineUnits'),
             'Calendar-conditioned historical mean': ('calendar_mean_baseline', 'CalendarBaselineUnits'),
             'Always zero': ('always_zero_baseline', None)}
    if comparison.Model.duplicated().any() or set(comparison.Model) != set(names):
        raise ValueError('Unexpected model_comparison.csv rows.')
    for name, (key, column) in names.items():
        actual_scores = original.score(holdout.UnitsSold, holdout[column] if column else np.zeros(len(holdout)))
        reported = comparison.loc[comparison.Model.eq(name)].iloc[0]
        for metric in ('MAE', 'RMSE', 'R2'):
            if actual_scores[metric] is None:
                if metrics[key][metric] is not None or not pd.isna(reported[metric]):
                    raise ValueError('Undefined score does not match the comparison table.')
            else:
                np.testing.assert_allclose([metrics[key][metric], reported[metric]], actual_scores[metric], atol=1e-8, rtol=1e-8)

    forecast_path = page1 / 'forecasts_expanded.csv'
    if digest(forecast_path) != summary['csv_sha256']:
        raise ValueError('Forecast CSV differs from SCENARIO_INFO.json.')
    forecast = original.load_csv(str(forecast_path), False)
    if len(forecast) != summary['rows'] or forecast.empty:
        raise ValueError('Unexpected forecast row count.')
    if not forecast.WeekStart.gt(pd.Timestamp(bundle['trained_through'])).all():
        raise ValueError('Forecast weeks overlap fitted history.')
    for column in original.CATEGORIES:
        if not set(forecast[column]).issubset(bundle['known_categories'][column]):
            raise ValueError(f'Forecast has unknown {column}.')
    for product, group in forecast.groupby('ProductID', sort=False):
        if not group.Category.isin(bundle['product_category'][product]).all():
            raise ValueError(f'Invalid product/category pair: {product}')
        rates = np.asarray(bundle['discounts_observed_per_product'][product], dtype=float)
        if not np.isclose(group.ScheduledCampaignDiscountPct.to_numpy()[:, None], rates[None, :], rtol=0, atol=1e-12).any(axis=1).all():
            raise ValueError(f'Unsupported discount for product {product}')
    np.testing.assert_allclose(forecast.PredictedUnits, revised.final_predict(bundle['fitted'], forecast, bundle['gated']), atol=1e-7, rtol=1e-7)
    run_id = int(summary['forecast_run_id'])
    if not 1 <= run_id <= 2147483647 or run_id in (100001, 100002):
        raise ValueError('Invalid/reserved ForecastRunID.')
    sql, _ = helper.make_sql(helper.SQL_TEMPLATE, forecast_path, run_id, bundle, len(forecast))
    files = {
        'forecasts_expanded.csv': forecast_path,
        'holdout_predictions.csv': results / 'holdout_predictions.csv',
        'model_comparison.csv': results / 'model_comparison.csv',
        'metrics.json': results / 'metrics.json',
        'model_info.json': results / 'model_info.json',
        'SCENARIO_INFO.json': page1 / 'SCENARIO_INFO.json',
    }
    return files, sql, summary, metrics


def make_queries(destination):
    prefix = '''let
    Root = if RegressionOutputPath = null then error "Set RegressionOutputPath to the BI_Data folder."
        else Text.TrimEnd(Text.From(RegressionOutputPath), {"\\", "/"}),
    Current = Json.Document(File.Contents(Root & "\\current.json")),
    Folder = Root & "\\" & Text.Replace(Current[relative_path], "/", "\\"),
'''
    schemas = {
        'forecasts_calendar': ('forecasts_expanded.csv', [('WeekStart','type date'),('ProductID','Int64.Type'),('TerritoryID','Int64.Type'),('Category','type text'),('ScheduledCampaignDiscountPct','type number'),('ProposalID','type text'),('ScenarioKind','type text'),('ForecastRunID','Int64.Type'),('RawPredictedUnits','type number'),('PredictedUnits','type number'),('ModelVersion','type text'),('PredictedAt','type datetimezone'),('TrainingDataOrigin','type text')]),
        'holdout_predictions': ('holdout_predictions.csv', [('WeekStart','type date'),('ProductID','Int64.Type'),('TerritoryID','Int64.Type'),('Category','type text'),('ScheduledCampaignDiscountPct','type number'),('UnitsSold','type number'),('RawPredictedUnits','type number'),('PredictedUnits','type number'),('BaselineUnits','type number'),('CalendarBaselineUnits','type number'),('OriginalPredictedUnits','type number')]),
        'model_comparison': ('model_comparison.csv', [('Model','type text'),('MAE','type number'),('RMSE','type number'),('R2','type number')]),
    }
    for query, (filename, types) in schemas.items():
        pairs = ',\n            '.join('{"' + name + '", ' + typ + '}' for name, typ in types)
        text = prefix + f'''    Source = Csv.Document(File.Contents(Folder & "\\{filename}"),
        [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]),
    Headers = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    Types = Table.TransformColumnTypes(Headers, {{
            {pairs}
        }}, "en-US")
in
    Types
'''
        (destination / f'{query}.m').write_text(text, encoding='utf-8')
    text = prefix + '''    RunID = Int64.From(Current[forecast_run_id]),
    Source = if Current[sql_imported] <> true then
        error "This batch was not imported into SQL. Run Publish_BI_v2.py with --sql-server first."
        else Sql.Database(SQLServerName, DatabaseName),
    Forecasts = Source{[Schema="dbo", Item="SALES_FORECAST"]}[Data],
    CurrentBatch = Table.SelectRows(Forecasts, each [ForecastRunID] = RunID)
in
    CurrentBatch
'''
    (destination / 'SALES_FORECAST.m').write_text(text, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', default=str(Path(__file__).resolve().parent))
    parser.add_argument('--prepared', default='prepared_v2')
    parser.add_argument('--results', default='outputs_calendar_v2')
    parser.add_argument('--page1', default='page1_calendar_v2')
    parser.add_argument('--bi-dir', default='BI_Data')
    parser.add_argument('--sql-server', help='Optional: import forecast using Windows integrated SQL authentication.')
    parser.add_argument('--database', default='DW_Promotion_Analytics')
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    resolve = lambda s: (root / s).resolve()
    bi = resolve(args.bi_dir)
    if any(bi == resolve(s) for s in (args.prepared, args.results, args.page1)):
        raise ValueError('BI_Data must be separate from original output folders.')
    files, sql, summary, metrics = validate(root, resolve(args.prepared), resolve(args.results), resolve(args.page1))
    print('Output checks passed: model, training hash, holdout scores and scenario predictions.', flush=True)
    if args.sql_server and os.name != 'nt':
        raise ValueError('--sql-server uses Windows PowerShell; run this option on your Windows SQL machine.')
    helper = Path(__file__).resolve().with_name('Import_Forecast_v2.ps1')
    if args.sql_server and not helper.is_file():
        raise ValueError('Copy Import_Forecast_v2.ps1 beside Publish_BI_v2.py.')
    if args.database != 'DW_Promotion_Analytics':
        if not args.database or len(args.database) > 128:
            raise ValueError('Invalid database name.')
        sql = sql.replace('USE [DW_Promotion_Analytics];', 'USE [' + args.database.replace(']', ']]') + '];', 1)
    bi.mkdir(parents=True, exist_ok=True)
    batches = bi / 'batches'; batches.mkdir(exist_ok=True)
    name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
    stage = Path(tempfile.mkdtemp(prefix='.publishing_', dir=batches))
    pointer_tmp = None
    try:
        for filename, source in files.items():
            shutil.copy2(source, stage / filename)
        (stage / 'Import_Page1_Expanded.sql').write_text(sql, encoding='utf-8-sig')
        if args.sql_server:
            command = ['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(helper),
                       '-Server', args.sql_server, '-Database', args.database,
                       '-SqlFile', str(stage / 'Import_Page1_Expanded.sql'),
                       '-RunId', str(summary['forecast_run_id']), '-ExpectedRows', str(summary['rows'])]
            proc = subprocess.run(command, capture_output=True, text=True)
            (stage / 'sql_import.log').write_text(proc.stdout + '\n' + proc.stderr, encoding='utf-8')
            if proc.returncode or 'SQL_IMPORT_OK' not in proc.stdout:
                raise ValueError('SQL import failed; BI pointer was not changed. ' + proc.stderr.strip() + ' ' + proc.stdout[-1000:])
            print('SQL import succeeded.', flush=True)
        metadata = {'format_version': 1, 'relative_path': f'batches/{name}',
                    'published_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                    'model_version': summary['model_version'], 'forecast_run_id': summary['forecast_run_id'],
                    'forecast_rows': summary['rows'], 'holdout_rows': metrics['test_rows'],
                    'sql_imported': bool(args.sql_server), 'sql_server': args.sql_server,
                    'database': args.database if args.sql_server else None,
                    'sha256': {f: digest(stage / f) for f in files}}
        (stage / 'publication.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        stage.rename(batches / name)
        query_dir = bi / 'PowerQuery'; query_dir.mkdir(exist_ok=True)
        make_queries(query_dir)
        pointer_tmp = bi / f'.current_{uuid.uuid4().hex}.json'
        pointer_tmp.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        os.replace(pointer_tmp, bi / 'current.json')
        print(f'PUBLISHED: {bi}\nForecastRunID: {summary["forecast_run_id"]}\nPower BI: refresh after this command finishes.')
        if not args.sql_server:
            print('CSV queries updated. SQL SALES_FORECAST was not changed; use --sql-server if visuals read that table.')
    finally:
        if stage.exists(): shutil.rmtree(stage)
        if pointer_tmp and pointer_tmp.exists(): pointer_tmp.unlink()


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(2)
