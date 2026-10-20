# CompanyX weekly sales forecasting (V2)

This repository contains the code for the **Reseller** weekly units forecast. The prediction grain is `ProductID × TerritoryID × WeekStart`. It covers the 38 products selected for the promotion-calendar analysis; it is not a model for Individual customers or the full product catalog.

`promotion_lr_calendar_v2.py` is the V2 training and prediction entry point. `promotion_lr.py` supplies shared loading, feature, baseline, and scoring functions imported by V2. Run the V2 entry point for the final model.

## Inputs

Place the following warehouse exports in `inputs/` before running. They are not included in this code-only repository:

- `warehouse_sales_lines_fixed.csv`
- `promotion_eligibility_dw.csv`

The CSV exports must match the expected columns in `prepare_dwh_data_v2.py`. Do not use a dashboard aggregate in place of the line-level sales export.

## Run in PowerShell (Python 3.11)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

.\.venv\Scripts\python.exe prepare_dwh_data_v2.py --sales inputs/warehouse_sales_lines_fixed.csv --eligibility inputs/promotion_eligibility_dw.csv --out prepared_v2 --assume-source-export-complete --assume-catalog-availability --assume-calendar-known --allow-snapshot-category

.\.venv\Scripts\python.exe promotion_lr_calendar_v2.py train --csv prepared_v2/training.csv --out outputs_calendar_v2 --test-weeks 44

.\.venv\Scripts\python.exe verify_calendar_v2.py --prepared prepared_v2 --results outputs_calendar_v2
```

The four `--assume-*` and `--allow-*` flags explicitly record assumptions for this retrospective dataset. Check `prepared_v2/data_audit.json`, `outputs_calendar_v2/metrics.json`, and `outputs_calendar_v2/holdout_predictions.csv` after running. In the reference run there were 32,198 prepared cells, and the selected model's holdout MAE was about 0.931 units and RMSE about 3.110 units. Use the results from your own run when reporting numbers.

## Forecast and Power BI

The small example file `scenarios_v2_demo.csv` can be used to test prediction:

```powershell
.\.venv\Scripts\python.exe promotion_lr_calendar_v2.py predict --csv scenarios_v2_demo.csv --model outputs_calendar_v2/model.joblib --out outputs_calendar_v2/forecasts_v2_demo.csv --run-id 100006
```

For the Page 1 July 2014 scenario and SQL/Power BI import helpers:

```powershell
.\.venv\Scripts\python.exe Expand_Page1_Calendar_v2.py --prepared prepared_v2 --model outputs_calendar_v2/model.joblib --out page1_calendar_v2 --run-id 100006
```

Use a new `--run-id` and output path for another batch. The script creates the scenario CSV, forecast CSV, SQL import, and query files inside `page1_calendar_v2/`. It does **not** run the SQL or modify a PBIX file. Import the forecast batch into the intended warehouse, set the Page 1 query to that batch, and point Page 2 to `outputs_calendar_v2/holdout_predictions.csv`.

Forecast weeks must follow the model's last training week. Product/category values and discount rates are checked against training data; a new product or customer segment requires a new data preparation and model evaluation.

`prepared_v2/`, `outputs_calendar_v2/`, and `page1_calendar_v2/` are generated locally and excluded from this repository. See the scripts for the complete validation rules.
