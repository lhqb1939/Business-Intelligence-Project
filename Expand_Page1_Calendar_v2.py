r"""Create expanded hypothetical promotion scenarios and a SQL import for Page 1.

Copy this file beside promotion_lr_calendar_v2.py in CompanyX_LR_Calendar.
Run: .\.venv\Scripts\python.exe Expand_Page1_Calendar.py
No training or SQL execution is performed by this script.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import SimpleNamespace


def sql_text(value):
    text = str(value)
    if any(c in text for c in ('\x00', '\r', '\n')):
        raise ValueError('Invalid control character in SQL text.')
    return "N'" + text.replace("'", "''") + "'"


def decimal_value(value, scale, maximum):
    number = Decimal(str(value))
    if not number.is_finite() or not Decimal(0) <= number < maximum:
        raise ValueError('Nonfinite, negative, or oversized number.')
    rounded = number.quantize(Decimal(1).scaleb(-scale))
    if scale == 8 and rounded != number:
        raise ValueError('Discount precision exceeds 8 decimals.')
    return format(rounded, f'.{scale}f')


def optional_date(text):
    if text is None or text.strip() in ('', 'NULL'):
        return None
    return date.fromisoformat(text.strip()[:10])


def build_scenarios(bundle, catalogue_path, first_week, weeks, observed_products):
    with catalogue_path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    catalogue = {}
    fields = ('ProductName', 'Category', 'ProductSellStartDate', 'ProductSellEndDate', 'ProductDiscontinuedDate')
    for row in rows:
        product = str(int(row['ProductID']))
        item = {field: row[field] for field in fields}
        if product in catalogue and catalogue[product] != item:
            raise ValueError(f'Inconsistent catalogue dates/category for product {product}.')
        catalogue[product] = item
    scenarios, products = [], {}
    excluded = []
    for product in sorted(bundle['known_categories']['ProductID'], key=int):
        if product not in observed_products:
            excluded.append({'ProductID': product, 'reason': 'No positive historical UnitsSold in the fitted warehouse dataset; catalogue-only zero-filled products are not proposal candidates'})
            continue
        if product not in catalogue:
            raise ValueError(f'Missing catalogue metadata for product {product}.')
        item = catalogue[product]
        if item['Category'] not in bundle['product_category'][product]:
            raise ValueError(f'Product {product} category differs from the model snapshot.')
        starts = optional_date(item['ProductSellStartDate'])
        if starts is None:
            raise ValueError(f'Missing sell start for product {product}.')
        endings = [optional_date(item[c]) for c in ('ProductSellEndDate', 'ProductDiscontinuedDate')]
        endings = [d for d in endings if d is not None]
        stop = min(endings) if endings else None
        rates = [Decimal(str(r)) for r in bundle['discounts_observed_per_product'][product]]
        if Decimal(0) not in rates or len(rates) < 2:
            excluded.append({'ProductID': product, 'reason': 'No supported positive campaign rate plus zero baseline'})
            continue
        eligible = []
        for i in range(weeks):
            week = first_week + timedelta(weeks=i)
            if week >= starts and (stop is None or week + timedelta(days=6) < stop):
                eligible.append(week)
        if not eligible:
            excluded.append({'ProductID': product, 'reason': 'Catalogue sale interval does not cover any complete requested week'})
            continue
        products[product] = {**item, 'SupportedRates': [str(r) for r in rates]}
        for week in eligible:
            for territory in sorted(bundle['known_categories']['TerritoryID'], key=int):
                for rate in sorted(rates):
                    scenarios.append({
                        'WeekStart': str(week), 'ProductID': product, 'Category': item['Category'],
                        'TerritoryID': territory, 'ScheduledCampaignDiscountPct': str(rate),
                        'ProposalID': f'JUL_P{product}_T{territory}_W{week:%Y%m%d}_D{int(rate*10000):04}',
                        'ScenarioKind': 'DEMO_HYPOTHETICAL',
                    })
    if not scenarios:
        raise ValueError('No catalogue-active products support comparable rates in the requested period.')
    if len({r['ProposalID'] for r in scenarios}) != len(scenarios):
        raise ValueError('Duplicate generated scenario ID.')
    return scenarios, products, excluded


def make_sql(template, csv_path, run_id, bundle, expected_rows):
    with csv_path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != expected_rows:
        raise ValueError('Forecast count differs from the generated scenario plan.')
    grains, proposals, timestamps = set(), set(), set()
    values = []
    for row in rows:
        key = (row['WeekStart'], row['ProductID'], row['TerritoryID'], Decimal(row['ScheduledCampaignDiscountPct']))
        if key in grains or row['ProposalID'] in proposals:
            raise ValueError('Duplicate forecast scenario.')
        grains.add(key)
        proposals.add(row['ProposalID'])
        if row['ScenarioKind'] != 'DEMO_HYPOTHETICAL' or row['ModelVersion'] != bundle['model_version'] or int(row['ForecastRunID']) != run_id:
            raise ValueError('Mixed model, scenario type or run ID in forecast CSV.')
        if row['TrainingDataOrigin'] != ';'.join(bundle['data_origin']):
            raise ValueError('Forecast provenance differs from the model.')
        week = date.fromisoformat(row['WeekStart'])
        if week.weekday() != 0 or str(week) <= bundle['trained_through']:
            raise ValueError('Forecast week must follow fitted history and start on Monday.')
        stamp = datetime.fromisoformat(row['PredictedAt'])
        if stamp.tzinfo is None:
            raise ValueError('Prediction timestamp lacks timezone.')
        utc = stamp.astimezone(timezone.utc).replace(tzinfo=None).isoformat(timespec='microseconds')
        timestamps.add(utc)
        rate = decimal_value(row['ScheduledCampaignDiscountPct'], 8, Decimal('1.00000001'))
        units = decimal_value(row['PredictedUnits'], 16, Decimal('1000000000000'))
        if len(row['ProposalID']) > 50 or len(row['Category']) > 100 or len(row['ModelVersion']) > 100:
            raise ValueError('CSV text is too wide for SQL staging.')
        values.append(f"    ({sql_text(row['ProposalID'])}, '{week:%Y%m%d}', {int(row['ProductID'])}, {sql_text(row['Category'])}, {int(row['TerritoryID'])}, {rate},\n"
                      f"     {units}, {sql_text(row['ModelVersion'])}, '{utc}')")
    if len(timestamps) != 1:
        raise ValueError('Mixed forecast timestamps.')
    # SQL Server INSERT ... VALUES has a 1,000-row limit. Use blocks of 500.
    prefix = ('INSERT #Input (ProposalID, WeekStart, ProductID, Category, TerritoryID, DiscountPct, '
              'PredictedUnits, ModelVersion, PredictedAtUTC)\nVALUES\n')
    inserts = '\n\n'.join(prefix + ',\n'.join(values[i:i+500]) + ';' for i in range(0, len(values), 500))
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    sql = template.replace('__RUN_ID__', str(run_id)).replace('__CSV_SHA__', digest).replace('__EXPECTED_ROWS__', str(len(rows))).replace('__INPUT_INSERTS__', inserts)
    if any(x in sql for x in ('__RUN_ID__', '__CSV_SHA__', '__EXPECTED_ROWS__', '__INPUT_INSERTS__')):
        raise ValueError('Unresolved template token.')
    return sql, digest


def stable_write(path, text):
    if path.exists():
        if path.read_text(encoding='utf-8-sig') != text:
            raise ValueError(f'{path.name} has different content. Use a new --out and unused --run-id.')
    else:
        path.write_text(text, encoding='utf-8-sig')


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='outputs_calendar_v2/model.joblib')
    parser.add_argument('--prepared', default='prepared_v2')
    parser.add_argument('--out', default='page1_calendar_v2')
    parser.add_argument('--run-id', type=int, default=100006)
    parser.add_argument('--start', default='2014-07-07')
    parser.add_argument('--weeks', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.run_id <= 2147483647 or args.run_id in (100001, 100002):
        raise ValueError('Use a new positive batch ID; existing demo batches 100001/100002 are preserved.')
    first_week = date.fromisoformat(args.start)
    if first_week.weekday() != 0 or not 1 <= args.weeks <= 12:
        raise ValueError('Start must be Monday; weeks must be between 1 and 12.')
    required = [root / p for p in ('promotion_lr_calendar_v2.py', 'promotion_lr.py', 'inputs/promotion_eligibility_dw.csv', str(Path(args.prepared) / 'training.csv'))]
    if any(not p.is_file() for p in required):
        raise ValueError('Copy this script into CompanyX_LR_Calendar beside promotion_lr_calendar_v2.py; keep the inputs folder.')
    import joblib
    import numpy as np
    import sklearn
    import promotion_lr as original
    import promotion_lr_calendar_v2 as revised

    model = (root / args.model).resolve()
    bundle = joblib.load(model)
    if bundle['sklearn_version'] != sklearn.__version__:
        raise ValueError('scikit-learn version differs from the fitted model.')
    if not bundle['model_version'].startswith('calendar_') or 'fitted' not in bundle:
        raise ValueError('Expected the Calendar model.')
    if str(first_week) <= bundle['trained_through']:
        raise ValueError('Requested weeks overlap fitted history.')
    catalogue = root / 'inputs/promotion_eligibility_dw.csv'
    audit_path = root / args.prepared / 'data_audit.json'
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding='utf-8-sig'))
        record = next(r for r in audit['source_files'] if r['name'] == catalogue.name)
        if hashlib.sha256(catalogue.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('Catalogue CSV no longer matches the audited dataset.')
    training_path = root / args.prepared / 'training.csv'
    if hashlib.sha256(training_path.read_bytes()).hexdigest() != bundle['training_csv_sha256']:
        raise ValueError('Prepared training CSV differs from the dataset fitted by this model.')
    history = original.load_csv(str(training_path), True)
    observed_products = set(history.loc[history.UnitsSold.gt(0), 'ProductID'].astype(str))
    scenarios, products, excluded = build_scenarios(bundle, catalogue, first_week, args.weeks, observed_products)
    out = (root / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    forecast = out / 'forecasts_expanded.csv'
    with tempfile.TemporaryDirectory(prefix='companyx-expanded-') as temp:
        scenario_path = Path(temp) / 'scenarios.csv'
        fresh = Path(temp) / 'forecasts.csv'
        with scenario_path.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(scenarios[0]));writer.writeheader();writer.writerows(scenarios)
        revised.predict(SimpleNamespace(csv=str(scenario_path), model=str(model), out=str(fresh), run_id=args.run_id))
        new = original.load_csv(str(fresh), False)
        if forecast.exists():
            previous = original.load_csv(str(forecast), False)
            cols = ['WeekStart', 'ProductID', 'Category', 'TerritoryID', 'ScheduledCampaignDiscountPct', 'ProposalID', 'ScenarioKind']
            if not previous[cols].equals(new[cols]):
                raise ValueError('Existing forecast scenarios differ. Use a new --out and unused --run-id.')
            for name in ('RawPredictedUnits', 'PredictedUnits'):
                if not np.allclose(previous[name], new[name], rtol=0, atol=1e-10):
                    raise ValueError('Existing predictions differ. Use a new --out and unused --run-id.')
            make_sql(SQL_TEMPLATE, forecast, args.run_id, bundle, len(scenarios))
        else:
            make_sql(SQL_TEMPLATE, fresh, args.run_id, bundle, len(scenarios))
            forecast.write_bytes(fresh.read_bytes())
        stable_write(out / 'scenarios_expanded.csv', scenario_path.read_text(encoding='utf-8'))
    sql, digest = make_sql(SQL_TEMPLATE, forecast, args.run_id, bundle, len(scenarios))
    stable_write(out / 'Import_Page1_Expanded.sql', sql)
    query = f"""SELECT
    'DEMO_HYPOTHETICAL' AS ScenarioKind,
    d.FullDate AS WeekStart,
    p.ProductID, p.ProductName, p.Category,
    t.TerritoryID, t.TerritoryName,
    f.DiscountPct, f.PredictedUnits, f.ModelVersion, f.ForecastRunID
FROM dbo.SALES_FORECAST AS f
JOIN dbo.DIM_DATE AS d ON d.DateKey = f.WeekStartDateKey
JOIN dbo.DIM_PRODUCT AS p ON p.ProductKey = f.ProductKey
JOIN dbo.DIM_TERRITORY AS t ON t.TerritoryKey = f.TerritoryKey
WHERE f.ForecastRunID = {args.run_id}
"""
    stable_write(out / 'Page1_Query.sql', query + ';\n')
    escaped = query.replace('"', '""').replace('\n', '#(lf)')
    powerquery = f'''let
    Source = Sql.Database("lpc:localhost\\MSSQLSERVER01", "DW_Promotion_Analytics", [Query="{escaped}"])
in
    Source
'''
    stable_write(out / 'Page1_Query.m', powerquery)
    measure = '''Scenario Units =
IF(
    HASONEVALUE('Query1'[ProductName])
        && HASONEVALUE('Query1'[TerritoryName])
        && HASONEVALUE('Query1'[WeekStart]),
    SUM('Query1'[PredictedUnits]),
    BLANK()
)
'''
    stable_write(out / 'Scenario_Units.dax', measure)
    summary = {'scenario_kind': 'DEMO_HYPOTHETICAL', 'forecast_run_id': args.run_id, 'model_version': bundle['model_version'],
               'model_module': 'promotion_lr_calendar_v2', 'gated': bool(bundle['gated']), 'gate_rule': bundle['gate_rule'],
               'rows': len(scenarios), 'products': len(products), 'territories': len(bundle['known_categories']['TerritoryID']),
               'weeks': args.weeks, 'first_week': str(first_week), 'last_week': str(first_week + timedelta(weeks=args.weeks-1)),
               'discount_rates': sorted({r['ScheduledCampaignDiscountPct'] for r in scenarios}, key=Decimal),
               'catalogue_sale_interval_checked': True, 'rates_supported_per_product': True,
               'scenario_products_require_positive_observed_warehouse_units': True,
               'csv_sha256': digest, 'sql_executed': False, 'product_details': products, 'excluded_products': excluded,
               'scope': 'Historical post-backup hypothetical campaigns, not confirmed offers or causal lift. All-territory availability and snapshot category assumptions remain.'}
    stable_write(out / 'SCENARIO_INFO.json', json.dumps(summary, indent=2, ensure_ascii=False))
    instructions = f'''# Page 1 — Expanded hypothetical campaign scenarios

Created {len(scenarios)} rows: {len(products)} products, {summary['territories']} territories, {args.weeks} weeks from {args.start}.
Model: {bundle['model_version']}; new warehouse batch: {args.run_id}.
No model training, SQL execution or PBIX editing is done by this script.

V2 uses a gated model when selected: {bundle["gate_rule"] if bundle["gated"] else "No month-end gate selected."}
This is a date-based structural rule, not a causal discount effect. PredictedUnits is the gated final output; RawPredictedUnits is before clipping/gating.

1. Open Import_Page1_Expanded.sql in SSMS, connect to lpc:localhost\\MSSQLSERVER01, run the WHOLE file. Default @Apply=1. Expected COMMITTED and {len(scenarios)} ForecastRowsInBatch.
2. Power BI: Home > Transform data > Query1 > Advanced Editor. Replace the whole query with Page1_Query.m. Done > Close & Apply. Keep the query name Query1.
3. Create a New measure from Scenario_Units.dax. Replace the entire default Measure = text when pasting.
4. On Page 1 replace Sum of PredictedUnits in the chart Y axis with Scenario Units. X axis stays DiscountPct, formatted as percentage.
5. Create THREE Slicer visuals with ProductName, TerritoryName, WeekStart. Use raw WeekStart date, not Date hierarchy. Enable Single select on each slicer; choose one item in each.
6. Start with Sport-100 Helmet, Red / Southwest / 2014-07-28. Other products only show the discount levels observed for that product, e.g. LL Road Frame offers 0% and 35%.
7. Title: Hypothetical promotion scenarios — Calendar Ridge V2. Set chart Y-axis minimum to 0. Keep decimal forecasts, do not round them to integer demand. Ctrl+S.

The Scenario Units measure intentionally returns blank until one product, one territory and one week are selected. Discount rates are mutually exclusive scenarios. Do not sum across competing rates into a total forecast. Different products have different supported rate sets; an unfiltered sum by rate would compare different product populations.

Catalogue sale intervals are checked for the entire week. Proposal candidates also require positive historical UnitsSold in the audited fitted dataset. Catalogue-only products that appear only as zero-filled training rows are excluded from these scenarios. This does not remove them from training or change model metrics. On the supplied data, products 724 and 737 are excluded for this reason; they had no source order lines and were not among sold products whose initial warehouse SCD start dates were repaired. This script does not backdate their dimension rows. Supported positive rates come from that product's model training rows, not a blanket 35% rate for every product. The historical campaign does not need to still be running: these are NEW HYPOTHETICAL rate scenarios, not assertions that the old offer is valid in July.

All 10 territories remain an assumed possible distribution scope, as in the training dataset. Category is the observed snapshot. July 2014 is after the backup observation interval; this is not a forecast for 2026. Better holdout RMSE does not prove causal promotion effects or an optimal discount.

Rerunning this Python command keeps the CSV timestamp and identical SQL. Rerunning identical SQL inserts no duplicate rows. A different occupied batch is rejected. Use a new unused --run-id and a new --out folder for a changed model/scenario plan.
'''
    stable_write(out / 'READ_PAGE1.md', instructions)
    print(f'READY: {len(scenarios)} scenarios; {len(products)} products; {summary["territories"]} territories; {args.weeks} weeks.')
    print(f'Open {out / "Import_Page1_Expanded.sql"} in SSMS and run the WHOLE file. Batch {args.run_id}.')
    print(f'Power BI query: {out / "Page1_Query.m"}; chart measure: {out / "Scenario_Units.dax"}.')
    return 0


# SQL_TEMPLATE is inserted into the standalone deliverable below.

SQL_TEMPLATE = r"""/*
CompanyX - Page 1, expanded Calendar linear model, DEMO_HYPOTHETICAL.
Generated from the actual local model/forecast CSV by update_page1.py.

SSMS: connect to lpc:localhost\MSSQLSERVER01 and run THIS WHOLE FILE.
Default @Apply = 1 saves all scenario rows atomically. Success: COMMITTED.
@Apply = 0 is an optional preview. Same payload rerun inserts zero new rows.
Different data in the selected batch is rejected, never overwritten.
Original batch 100001 is preserved. Product/territory/date keys are looked up.
Date calendar is extended only if needed; no Product SCD history is changed.
No local SQL execution was performed by the code author.

Scope: these are hypothetical rates for a historical post-backup week.
This is not causal promotion lift, profit or an optimal-discount recommendation.
After COMMITTED: change Page 1's Query1 SQL filter to the new batch ID.
The local CSV and model version are preserved in the generated SQL.
*/

USE [DW_Promotion_Analytics];
GO
SET NOCOUNT ON;
SET XACT_ABORT ON;
SET ANSI_WARNINGS ON;

DECLARE @Apply bit = 1; -- 0 = preview; 1 = save
DECLARE @ExpectedRows int = __EXPECTED_ROWS__;
DECLARE @TargetForecastRunID int = __RUN_ID__; -- dedicated DEMO batch; keep for reruns

IF @@TRANCOUNT <> 0
    THROW 52000, 'Use a new query window without an existing transaction.', 1;
IF @TargetForecastRunID <= 0
    THROW 52001, 'ForecastRunID must be a positive SQL INT.', 1;
IF OBJECT_ID(N'dbo.SALES_FORECAST', N'U') IS NULL
    THROW 52002, 'Existing dbo.SALES_FORECAST was not found in this database.', 1;

DROP TABLE IF EXISTS #Input, #Calendar, #DateMap, #Load, #Existing, #ToInsert;

CREATE TABLE #Input (
    ProposalID nvarchar(50) NOT NULL PRIMARY KEY,
    WeekStart date NOT NULL,
    ProductID int NOT NULL,
    Category nvarchar(100) NOT NULL,
    TerritoryID int NOT NULL,
    DiscountPct decimal(10,8) NOT NULL,
    PredictedUnits decimal(28,16) NOT NULL,
    ModelVersion nvarchar(100) NOT NULL,
    PredictedAtUTC datetime2(7) NOT NULL
);

-- Exact values generated from the local Calendar model. CSV SHA256: __CSV_SHA__
__INPUT_INSERTS__

BEGIN TRY
    BEGIN TRANSACTION;

    IF EXISTS (
        SELECT 1 FROM #Input WHERE DiscountPct < 0 OR DiscountPct > 1
           OR PredictedUnits < 0
           OR DATEDIFF(day, CONVERT(date, '19000101', 112), WeekStart) % 7 <> 0
    )
        THROW 52003, 'Invalid rate, negative prediction, or non-Monday week.', 1;

    IF EXISTS (
        SELECT WeekStart, ProductID, TerritoryID, DiscountPct FROM #Input
        GROUP BY WeekStart, ProductID, TerritoryID, DiscountPct HAVING COUNT(*) > 1
    )
        THROW 52004, 'Duplicate source scenario grain.', 1;

    IF ISNULL(COLUMNPROPERTY(OBJECT_ID(N'dbo.SALES_FORECAST'), N'ForecastKey', 'IsIdentity'), 0) <> 1
       AND NOT EXISTS (
           SELECT 1 FROM sys.columns
           WHERE object_id = OBJECT_ID(N'dbo.SALES_FORECAST')
             AND name = N'ForecastKey' AND default_object_id <> 0
       )
        THROW 52005, 'ForecastKey has no identity/default. Inspect actual schema before importing.', 1;

    -- Reject integer targets, which would silently lose fractional unit forecasts.
    IF EXISTS (
        SELECT 1 FROM sys.columns AS c
        JOIN sys.types AS t ON t.user_type_id = c.system_type_id
        WHERE c.object_id = OBJECT_ID(N'dbo.SALES_FORECAST')
          AND c.name IN (N'PredictedUnits', N'DiscountPct')
          AND t.name NOT IN (N'decimal', N'numeric', N'float', N'real')
    )
        THROW 52006, 'Forecast quantity/rate types need review; fractional values must be preserved.', 1;

    IF EXISTS (
        SELECT 1 FROM sys.columns
        WHERE object_id = OBJECT_ID(N'dbo.SALES_FORECAST')
          AND name IN (N'PredictedUnits', N'DiscountPct')
          AND system_type_id IN (106, 108) AND scale < 2
    )
        THROW 52020, 'Forecast decimal scale is below 2. Review target precision before importing.', 1;

    -- These scenarios differ only by discount and prediction. A unique key on
    -- only their common fields cannot hold all three. Never remove that key here.
    IF EXISTS (
        SELECT 1 FROM sys.indexes AS ix
        WHERE ix.object_id = OBJECT_ID(N'dbo.SALES_FORECAST')
          AND ix.is_unique = 1 AND ix.is_disabled = 0 AND ix.has_filter = 0
          AND NOT EXISTS (
              SELECT 1 FROM sys.index_columns AS ic
              JOIN sys.columns AS c ON c.object_id = ic.object_id AND c.column_id = ic.column_id
              WHERE ic.object_id = ix.object_id AND ic.index_id = ix.index_id
                AND ic.key_ordinal > 0
                AND c.name NOT IN (N'ForecastRunID', N'WeekStartDateKey', N'ProductKey',
                                   N'TerritoryKey', N'ModelVersion', N'PredictedAt')
          )
    )
        THROW 52007, 'A unique index prevents multiple discount scenarios in one batch. Inspect index before import.', 1;

    IF EXISTS (
        SELECT 1 FROM #Input AS s
        WHERE (SELECT COUNT_BIG(*) FROM dbo.DIM_PRODUCT AS p WITH (HOLDLOCK)
               WHERE p.ProductID = s.ProductID
                 AND CAST(s.WeekStart AS datetime2(7)) >= p.ValidFrom
                 AND (p.ValidTo IS NULL OR CAST(s.WeekStart AS datetime2(7)) < p.ValidTo)) <> 1
           OR (SELECT COUNT_BIG(*) FROM dbo.DIM_TERRITORY AS t WITH (HOLDLOCK)
               WHERE t.TerritoryID = s.TerritoryID) <> 1
    )
    BEGIN
        SELECT DISTINCT s.ProductID, s.TerritoryID, s.WeekStart,
               (SELECT COUNT_BIG(*) FROM dbo.DIM_PRODUCT AS p
                WHERE p.ProductID = s.ProductID AND s.WeekStart >= p.ValidFrom
                  AND (p.ValidTo IS NULL OR s.WeekStart < p.ValidTo)) AS ProductVersionsAtWeek,
               (SELECT MIN(p.ValidFrom) FROM dbo.DIM_PRODUCT AS p
                WHERE p.ProductID = s.ProductID) AS EarliestProductValidFrom,
               (SELECT COUNT_BIG(*) FROM dbo.DIM_TERRITORY AS t
                WHERE t.TerritoryID = s.TerritoryID) AS TerritoryMatches
        FROM #Input AS s
        WHERE (SELECT COUNT_BIG(*) FROM dbo.DIM_PRODUCT AS p
               WHERE p.ProductID = s.ProductID AND s.WeekStart >= p.ValidFrom
                 AND (p.ValidTo IS NULL OR s.WeekStart < p.ValidTo)) <> 1
           OR (SELECT COUNT_BIG(*) FROM dbo.DIM_TERRITORY AS t
               WHERE t.TerritoryID = s.TerritoryID) <> 1
        ORDER BY s.ProductID, s.TerritoryID, s.WeekStart;
        THROW 52008, 'Unmapped scenarios are listed above. Check Product SCD dates and Territory mapping; no rows committed.', 1;
    END;

    IF EXISTS (
        SELECT 1 FROM #Input AS s
        JOIN dbo.DIM_PRODUCT AS p WITH (HOLDLOCK)
          ON p.ProductID = s.ProductID AND s.WeekStart >= p.ValidFrom
         AND (p.ValidTo IS NULL OR s.WeekStart < p.ValidTo)
        WHERE p.Category IS NULL OR p.Category <> s.Category
           OR (p.ValidTo IS NOT NULL
               AND p.ValidTo < DATEADD(day, 7, CAST(s.WeekStart AS datetime2(7))))
    )
        THROW 52009, 'Product category differs from model input or changes during the forecast week.', 1;

    -- Build enough calendar to cover the whole week and bridge any gap from the
    -- existing calendar end. Existing rows are always preserved.
    DECLARE @CalendarEnd date = (SELECT MAX(CONVERT(date, FullDate)) FROM dbo.DIM_DATE WITH (UPDLOCK, HOLDLOCK));
    DECLARE @FirstWeek date = (SELECT MIN(WeekStart) FROM #Input);
    DECLARE @CalendarFrom date = @FirstWeek;
    DECLARE @CalendarThrough date = DATEADD(day, 6, (SELECT MAX(WeekStart) FROM #Input));
    IF @CalendarEnd IS NOT NULL AND @CalendarEnd < @FirstWeek
        SET @CalendarFrom = DATEADD(day, 1, @CalendarEnd);
    IF DATEDIFF(day, @CalendarFrom, @CalendarThrough) > 366
        THROW 52010, 'Calendar extension exceeds one year. Review the date range before importing.', 1;

    ;WITH Days AS (
        SELECT @CalendarFrom AS FullDate
        UNION ALL
        SELECT DATEADD(day, 1, FullDate) FROM Days WHERE FullDate < @CalendarThrough
    )
    SELECT FullDate INTO #Calendar FROM Days OPTION (MAXRECURSION 367);

    IF EXISTS (
        SELECT CONVERT(date, d.FullDate) FROM dbo.DIM_DATE AS d
        JOIN #Calendar AS c ON c.FullDate = CONVERT(date, d.FullDate)
        GROUP BY CONVERT(date, d.FullDate) HAVING COUNT_BIG(*) > 1
    )
        THROW 52011, 'Duplicate FullDate rows in DIM_DATE.', 1;

    SELECT c.FullDate,
           COALESCE(d.DateKey, CONVERT(int, CONVERT(char(8), c.FullDate, 112))) AS DateKey,
           CAST(CASE WHEN d.DateKey IS NULL THEN 1 ELSE 0 END AS bit) AS IsNewDate
    INTO #DateMap
    FROM #Calendar AS c
    LEFT JOIN dbo.DIM_DATE AS d WITH (UPDLOCK, HOLDLOCK)
      ON CONVERT(date, d.FullDate) = c.FullDate;

    IF EXISTS (
        SELECT 1 FROM #DateMap AS dm
        JOIN dbo.DIM_DATE AS d ON d.DateKey = dm.DateKey
        WHERE dm.IsNewDate = 1
    )
        THROW 52012, 'A proposed YYYYMMDD DateKey is already occupied by another date.', 1;

    IF EXISTS (SELECT 1 FROM #DateMap WHERE IsNewDate = 1)
       AND ISNULL(COLUMNPROPERTY(OBJECT_ID(N'dbo.DIM_DATE'), N'DateKey', 'IsIdentity'), 0) = 1
        THROW 52013, 'DIM_DATE.DateKey is identity; review calendar loading instead of forcing a key.', 1;

    -- Cloning the actual target types makes the preview show stored precision.
    SELECT TOP (0) ForecastRunID, WeekStartDateKey, ProductKey, TerritoryKey,
                   DiscountPct, PredictedUnits, ModelVersion, PredictedAt
    INTO #Load FROM dbo.SALES_FORECAST;

    INSERT #Load (ForecastRunID, WeekStartDateKey, ProductKey, TerritoryKey,
                  DiscountPct, PredictedUnits, ModelVersion, PredictedAt)
    SELECT @TargetForecastRunID, dm.DateKey, p.ProductKey, t.TerritoryKey,
           s.DiscountPct, s.PredictedUnits, s.ModelVersion, s.PredictedAtUTC
    FROM #Input AS s
    JOIN #DateMap AS dm ON dm.FullDate = s.WeekStart
    JOIN dbo.DIM_PRODUCT AS p ON p.ProductID = s.ProductID
      AND s.WeekStart >= p.ValidFrom AND (p.ValidTo IS NULL OR s.WeekStart < p.ValidTo)
    JOIN dbo.DIM_TERRITORY AS t ON t.TerritoryID = s.TerritoryID;

    IF (SELECT COUNT(*) FROM #Load) <> @ExpectedRows
        THROW 52014, 'Mapped scenario count differs from the generated scenario plan.', 1;

    IF EXISTS (
        SELECT 1 FROM #Input AS s
        WHERE NOT EXISTS (
            SELECT 1 FROM #Load AS l
            WHERE ABS(CONVERT(float, l.DiscountPct) - CONVERT(float, s.DiscountPct)) < 0.00000001
              AND l.WeekStartDateKey = (SELECT DateKey FROM #DateMap WHERE FullDate = s.WeekStart)
              AND l.ProductKey IN (SELECT ProductKey FROM dbo.DIM_PRODUCT WHERE ProductID = s.ProductID)
              AND l.TerritoryKey IN (SELECT TerritoryKey FROM dbo.DIM_TERRITORY WHERE TerritoryID = s.TerritoryID)
        )
    )
        THROW 52021, 'Target precision changed a supplied discount rate.', 1;

    IF EXISTS (
        SELECT WeekStartDateKey, ProductKey, TerritoryKey, DiscountPct FROM #Load
        GROUP BY WeekStartDateKey, ProductKey, TerritoryKey, DiscountPct HAVING COUNT(*) > 1
    )
        THROW 52015, 'Target precision collapses distinct scenario rates.', 1;

    SELECT ForecastRunID, WeekStartDateKey, ProductKey, TerritoryKey,
           DiscountPct, PredictedUnits, ModelVersion, PredictedAt
    INTO #Existing FROM dbo.SALES_FORECAST WITH (UPDLOCK, HOLDLOCK)
    WHERE ForecastRunID = @TargetForecastRunID;

    IF EXISTS (
        SELECT WeekStartDateKey, ProductKey, TerritoryKey, DiscountPct FROM #Existing
        GROUP BY WeekStartDateKey, ProductKey, TerritoryKey, DiscountPct HAVING COUNT(*) > 1
    )
        THROW 52016, 'Duplicate scenarios already exist in the selected batch.', 1;

    IF EXISTS (
        SELECT * FROM #Existing
        EXCEPT SELECT * FROM #Load
    )
        THROW 52017, 'Batch __RUN_ID__ is occupied by different data. Choose a different unused TargetForecastRunID; do not overwrite.', 1;

    SELECT * INTO #ToInsert FROM #Load
    EXCEPT SELECT * FROM #Existing;

    SELECT CASE WHEN @Apply = 0 THEN 'PREVIEW' ELSE 'APPLY' END AS Mode,
           CAST('DEMO_HYPOTHETICAL' AS varchar(30)) AS ScenarioKind,
           __RUN_ID__ AS SourceCSVRunID, @TargetForecastRunID AS WarehouseRunID,
           (SELECT COUNT(*) FROM #Load) AS ForecastRows,
           (SELECT COUNT(*) FROM #Existing) AS AlreadyPresent,
           (SELECT COUNT(*) FROM #ToInsert) AS RowsToInsert,
           (SELECT COUNT(*) FROM #DateMap WHERE IsNewDate = 1) AS DatesToInsert;

    SELECT s.ProposalID, s.WeekStart, s.ProductID, l.ProductKey,
           s.TerritoryID, l.TerritoryKey, l.WeekStartDateKey,
           l.DiscountPct, s.PredictedUnits AS CSVPredictedUnits,
           l.PredictedUnits AS StoredPredictedUnits, l.ModelVersion,
           l.ForecastRunID, l.PredictedAt
    FROM #Load AS l
    JOIN #Input AS s ON s.DiscountPct = l.DiscountPct
      AND s.WeekStart = CONVERT(date, (SELECT FullDate FROM #DateMap WHERE DateKey = l.WeekStartDateKey))
      AND s.ProductID = (SELECT ProductID FROM dbo.DIM_PRODUCT WHERE ProductKey = l.ProductKey)
      AND s.TerritoryID = (SELECT TerritoryID FROM dbo.DIM_TERRITORY WHERE TerritoryKey = l.TerritoryKey)
    ORDER BY l.DiscountPct;

    IF @Apply = 0
    BEGIN
        ROLLBACK TRANSACTION;
        PRINT 'PREVIEW ONLY. No permanent changes. Set @Apply = 1 to save this demo batch.';
        RETURN;
    END;

    INSERT dbo.DIM_DATE (DateKey, FullDate, [Month], [Year])
    SELECT DateKey, FullDate, MONTH(FullDate), YEAR(FullDate)
    FROM #DateMap WHERE IsNewDate = 1;
    DECLARE @DatesInserted int = @@ROWCOUNT;

    INSERT dbo.SALES_FORECAST
      (ForecastRunID, WeekStartDateKey, ProductKey, TerritoryKey,
       DiscountPct, PredictedUnits, ModelVersion, PredictedAt)
    SELECT ForecastRunID, WeekStartDateKey, ProductKey, TerritoryKey,
           DiscountPct, PredictedUnits, ModelVersion, PredictedAt
    FROM #ToInsert;
    DECLARE @ForecastsInserted int = @@ROWCOUNT;

    IF (SELECT COUNT(*) FROM dbo.SALES_FORECAST WHERE ForecastRunID = @TargetForecastRunID) <> @ExpectedRows
       OR EXISTS (
           SELECT * FROM #Load
           EXCEPT
           SELECT ForecastRunID, WeekStartDateKey, ProductKey, TerritoryKey,
                  DiscountPct, PredictedUnits, ModelVersion, PredictedAt
           FROM dbo.SALES_FORECAST WHERE ForecastRunID = @TargetForecastRunID
       )
        THROW 52018, 'Stored forecasts differ from the validated plan. All changes rolled back.', 1;

    IF EXISTS (
        SELECT 1 FROM #Load AS l
        LEFT JOIN dbo.DIM_DATE AS d ON d.DateKey = l.WeekStartDateKey
        LEFT JOIN dbo.DIM_PRODUCT AS p ON p.ProductKey = l.ProductKey
        LEFT JOIN dbo.DIM_TERRITORY AS t ON t.TerritoryKey = l.TerritoryKey
        WHERE d.DateKey IS NULL OR p.ProductKey IS NULL OR t.TerritoryKey IS NULL
    )
        THROW 52019, 'A forecast dimension key is missing. All changes rolled back.', 1;

    COMMIT TRANSACTION;

    SELECT 'COMMITTED' AS Result, 'DEMO_HYPOTHETICAL' AS ScenarioKind,
           @TargetForecastRunID AS WarehouseRunID,
           @ForecastsInserted AS ForecastsInserted, @DatesInserted AS DatesInserted,
           @ExpectedRows AS ForecastRowsInBatch;

    -- For Power BI, copy this SELECT only and replace the variable in its WHERE
    -- with the committed numeric batch ID (__RUN_ID__ by default).
    SELECT 'DEMO_HYPOTHETICAL' AS ScenarioKind,
           f.ForecastKey, f.ForecastRunID,
           d.FullDate AS WeekStart, p.ProductID, p.ProductName, p.Category,
           t.TerritoryID, t.TerritoryName,
           f.DiscountPct, f.PredictedUnits, f.ModelVersion, f.PredictedAt
    FROM dbo.SALES_FORECAST AS f
    JOIN dbo.DIM_DATE AS d ON d.DateKey = f.WeekStartDateKey
    JOIN dbo.DIM_PRODUCT AS p ON p.ProductKey = f.ProductKey
    JOIN dbo.DIM_TERRITORY AS t ON t.TerritoryKey = f.TerritoryKey
    WHERE f.ForecastRunID = @TargetForecastRunID
    ORDER BY f.DiscountPct;
END TRY
BEGIN CATCH
    IF XACT_STATE() <> 0 ROLLBACK TRANSACTION;
    THROW;
END CATCH;
"""


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError, InvalidOperation) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(2)
