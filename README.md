# CompanyX Promotion Analytics — Weekly Sales Forecasting V2

This project forecasts weekly sales quantities for selected CompanyX Reseller products and publishes the results for Power BI. It includes warehouse-export preparation, chronological model selection, holdout evaluation, future-week scenario generation, and an optional SQL import.

The prediction grain is **ProductID × TerritoryID × WeekStart**. `WeekStart` is a Monday, and the target `UnitsSold` is the sum of eligible `OrderQty` across the seven days of that week. The source warehouse fact table remains at sales-order-line grain.

**Already have a completed V2 run?** Start at [Publish results](#7-publish-results-for-power-bi), then complete the [one-time Power BI setup](#8-configure-power-bi-once). Retraining is not required just to publish existing results.

## 1. Scope and model

- Shipped **Reseller** sales only; Individual customers are outside this model's training scope.
- The supplied snapshot produces **38 selected products** and **32,198 prepared weekly observations**. These counts describe the reference dataset, not a requirement for every future dataset.
- The scheduled-discount feature describes qualifying non-volume campaigns. Quantity-tier sales remain in the eligible sales-quantity target.
- Candidate models include the original OLS specification, additive calendar models, interaction models, Ridge penalties, and optional month-end gates.
- Model selection uses three chronological expanding-window validation folds of 13 weeks. The reference selection is `interaction_10_gated`, a Ridge model with α = 10 and a month-end gate.
- The gate sets predictions to zero outside weeks containing the last calendar day of a month. This is an explicit rule motivated by the source data; it can miss sales in other periods.

The current workflow supports **scenario-based quantity forecasting**. Differences between forecast discount scenarios do not establish causal promotion lift or identify an optimal discount.

## 2. Repository files

| File | Purpose |
| --- | --- |
| `prepare_dwh_data_v2.py` | Validate the warehouse exports, prepare the weekly grid, and produce data-audit and quarantine outputs. |
| `promotion_lr_calendar_v2.py` | V2 training, chronological model selection, holdout evaluation, final refitting, and prediction. |
| `promotion_lr.py` | Shared loading, feature, baseline, and scoring functions imported by V2. Keep this dependency alongside the V2 scripts. |
| `verify_calendar_v2.py` | Reconcile source data and targets; check evaluation predictions, scores, calendar logic, and forecast guards. |
| `Expand_Page1_Calendar_v2.py` | Generate supported hypothetical forecast scenarios, forecast outputs, and SQL/Power BI helpers. |
| `Publish_BI_v2.py` | Check output consistency, optionally import forecasts into SQL, and publish a current batch for Power BI. |
| `Import_Forecast_v2.ps1` | SQL import helper called by the publisher, using Windows integrated authentication. |
| `Run_And_Publish_BI_v2.ps1` | Optional wrapper to train, verify, generate scenarios, and publish in one command. |
| `scenarios_v2_demo.csv` | Small forecast-input example for testing the prediction command. |
| `requirements.txt` | Pinned Python dependencies. |
| `CompanyX_LR_V2_GitHub.zip` | Optional source bundle. When the scripts are already present at the repository root, run those files directly. |

Input exports, fitted models, generated forecasts, the Power BI report/template, and database backups are not included in the source-only bundle. The Power Query files used by the publisher are generated under `BI_Data/PowerQuery/`; they do not have to exist in the repository before the first publication.

## 3. Prerequisites and environment

- Python **3.11**; the original Windows setup used Python 3.11.9.
- Windows PowerShell for the wrapper and optional SQL import.
- Power BI Desktop and the group's report/template for dashboard integration.
- SQL Server with `DW_Promotion_Analytics` restored and populated when using the SQL publication option.
- SQL permissions to read the dimensions and load `dbo.SALES_FORECAST`; the import can also add missing dates to `dbo.DIM_DATE`.

An already populated warehouse backup can be used directly. Re-running the group's SSIS project in Visual Studio is needed only when loading or updating the warehouse from the source database.

The examples below assume this local project folder:

```text
D:\CompanyX_LR_Calendar
```

Open PowerShell in the folder containing the scripts, then create the environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Use the virtual environment's Python for every command. Saved models require the same scikit-learn version that was used to fit them.

## 4. Required input data

Create an `inputs` folder and place these two original warehouse exports inside it:

```text
inputs/
    warehouse_sales_lines_fixed.csv
    promotion_eligibility_dw.csv
```

Obtain these exports from the project data package. They are not created by installing the Python dependencies, and `scenarios_v2_demo.csv` is not a substitute for the training inputs. The preparation script expects its documented line-level sales and eligibility columns; a dashboard aggregate cannot replace the sales export.

## 5. First run: prepare, train, and verify

### 5.1 Prepare the weekly dataset

```powershell
.\.venv\Scripts\python.exe prepare_dwh_data_v2.py --sales inputs/warehouse_sales_lines_fixed.csv --eligibility inputs/promotion_eligibility_dw.csv --out prepared_v2 --assume-source-export-complete --assume-catalog-availability --assume-calendar-known --allow-snapshot-category
```

The four flags explicitly acknowledge assumptions used in this retrospective experiment:

| Flag | Assumption |
| --- | --- |
| `--assume-source-export-complete` | The supplied export contains the relevant source transactions. |
| `--assume-catalog-availability` | Product availability is approximated from the available catalogue information. |
| `--assume-calendar-known` | The campaign schedule is treated as available before forecasting. |
| `--allow-snapshot-category` | Snapshot category information is used where earlier category history cannot be reconstructed. |

Eligible weekly cells without transactions are assigned zero under these assumptions. Cells affected by quarantined source errors are excluded rather than treated as zero sales. Review `prepared_v2/data_audit.json` and the quarantine files before interpreting results.

### 5.2 Train and evaluate V2

```powershell
.\.venv\Scripts\python.exe promotion_lr_calendar_v2.py train --csv prepared_v2/training.csv --out outputs_calendar_v2 --test-weeks 44
```

Training performs chronological validation within the earlier data, evaluates the selected specification on the last 44 weeks, and refits a separate forecast model on the complete prepared dataset.

Important outputs:

| Output | Meaning |
| --- | --- |
| `holdout_predictions.csv` | Actual quantities, model predictions, and baseline predictions for the historical holdout. |
| `metrics.json` | Holdout scores, split dates, and selected-model information. |
| `model_comparison.csv` | Holdout comparison table for the model and baselines. |
| `validation_results.json` | Chronological validation results and the selection rule. |
| `evaluation_model.joblib` | Model fitted using only the pre-holdout training period. |
| `model.joblib` | Model refitted on all prepared data for subsequent forecast weeks. |
| `model_info.json` | Forecast-model version, training hash, scope, and supported categories/rates. |

### 5.3 Verify the run

```powershell
.\.venv\Scripts\python.exe verify_calendar_v2.py --prepared prepared_v2 --results outputs_calendar_v2
```

Do not run the verifier with Python's `-O` option, because its checks use assertions. Keep the matching source exports, prepared data, models, and results together.

Preparation and training expect new output folders. To preserve an earlier run, use new folder names and pass them consistently to every downstream command.

## 6. Generate forecast scenarios

### 6.1 Small prediction example

```powershell
.\.venv\Scripts\python.exe promotion_lr_calendar_v2.py predict --csv scenarios_v2_demo.csv --model outputs_calendar_v2/model.joblib --out outputs_calendar_v2/forecasts_v2_demo.csv --run-id 100006
```

The example is a small prediction check. To create the expanded Page 1 outputs expected by the publisher, run the next command as well.

### 6.2 Expanded Page 1 scenarios

```powershell
.\.venv\Scripts\python.exe Expand_Page1_Calendar_v2.py --prepared prepared_v2 --model outputs_calendar_v2/model.joblib --out page1_calendar_v2 --run-id 100006 --start 2014-07-07 --weeks 4
```

For the reference snapshot, this generates **1,800 scenarios** covering 21 catalogue-active supported products, 10 territories, and four July 2014 weeks. The scenario set is smaller than the 38-product training scope because forecast eligibility is checked separately.

The dates are historical post-backup scenarios, not forecasts for the current calendar year. With the selected gate, the weeks starting 7, 14, and 21 July predict zero; the week starting 28 July can have positive predictions.

The folder includes:

- `forecasts_expanded.csv`: forecast rows, scenario identifiers, model version, and run ID.
- `scenarios_expanded.csv`: scenario inputs.
- `SCENARIO_INFO.json`: scenario scope and the forecast CSV hash.
- `Import_Page1_Expanded.sql`: generated SQL import.
- `Page1_Query.m`, `Page1_Query.sql`, and `Scenario_Units.dax`: helpers for the earlier direct-import workflow.

Use the publisher-based setup in the following sections for automatic current-batch selection. The publisher generates a separate set of Power Query files under `BI_Data/PowerQuery/`.

Forecast constraints:

- `--start` must be a Monday after the final fitted training week; the expansion helper accepts 1–12 weeks.
- Forecast categories and discount rates must be supported by the fitted model; rates are checked per product.
- The helper also checks observed sales and catalogue activity for the requested dates.
- Use a new output folder and an unused positive SQL INT run ID for a different forecast batch. IDs `100001` and `100002` are reserved by the expansion helper.

## 7. Publish results for Power BI

### 7.1 Publish CSV outputs and import the forecast into SQL

Use this option when the report reads `SALES_FORECAST` from the warehouse:

```powershell
.\.venv\Scripts\python.exe Publish_BI_v2.py --prepared prepared_v2 --results outputs_calendar_v2 --page1 page1_calendar_v2 --sql-server "lpc:localhost\MSSQLSERVER01"
```

Replace the example server with the instance used by your machine. Add `--database YOUR_DATABASE` when the warehouse has a different name.

The publisher checks model/training consistency, holdout quantities and scores, scenario metadata, supported inputs, and saved predictions. It regenerates the SQL import from the validated forecast CSV, loads the batch using the current Windows account, and then publishes the completed files.

The SQL import maps the warehouse dimension keys and can add missing date rows. It preserves Product history and previous forecast batches. A reused run ID containing different data is rejected; re-importing the same payload does not duplicate rows.

Successful output includes:

```text
Output checks passed: model, training hash, holdout scores and scenario predictions.
SQL import succeeded.
PUBLISHED: D:\CompanyX_LR_Calendar\BI_Data
ForecastRunID: 100006
Power BI: refresh after this command finishes.
```

### 7.2 CSV-only publication

```powershell
.\.venv\Scripts\python.exe Publish_BI_v2.py --prepared prepared_v2 --results outputs_calendar_v2 --page1 page1_calendar_v2
```

This publishes the three CSV tables without updating SQL. Use the CSV queries for the forecast visuals in this mode. The publisher-generated `SALES_FORECAST.m` deliberately rejects a publication that was not imported into SQL.

## 8. Configure Power BI once

Open the group's Power BI report/template and select **Transform data**.

### 8.1 Set the parameters

Under **Manage Parameters**, set each parameter to Type **Text**, with a non-empty Current Value:

| Parameter | Example Current Value |
| --- | --- |
| `SQLServerName` | `lpc:localhost\MSSQLSERVER01` |
| `DatabaseName` | `DW_Promotion_Analytics` |
| `RegressionOutputPath` | `D:\CompanyX_LR_Calendar\BI_Data` |

`RegressionOutputPath` points to the **BI_Data folder**, not a CSV and not `outputs_calendar_v2`. The SQL parameters must match the server/database used for publication.

### 8.2 Update the existing queries

After the first successful publication, open each query's **Advanced Editor** and replace its code with the corresponding generated file:

| Existing query name | Generated Power Query file |
| --- | --- |
| `forecasts_calendar` | `BI_Data/PowerQuery/forecasts_calendar.m` |
| `holdout_predictions` | `BI_Data/PowerQuery/holdout_predictions.m` |
| `model_comparison` | `BI_Data/PowerQuery/model_comparison.m` |
| `SALES_FORECAST` — SQL publication only | `BI_Data/PowerQuery/SALES_FORECAST.m` |

Keep the query names so existing report references remain valid. The CSV queries read the current publication; the SQL query filters `ForecastRunID` to that same publication. Keep the warehouse dimension and actual-sales queries connected to the warehouse.

If the report does not already contain these query names, create matching queries and bind the relevant visuals to them. The Python scripts do not create or modify a PBIX/PBIT file.

### 8.3 Check forecast relationships

For `SALES_FORECAST`, use **Many to one (*:1)**, with the dimension on the `1` side and **Single** filtering from the dimension:

| Forecast column — many side | Dimension column — one side |
| --- | --- |
| `SALES_FORECAST[WeekStartDateKey]` | `DIM_DATE[DateKey]` |
| `SALES_FORECAST[ProductKey]` | `DIM_PRODUCT[ProductKey]` |
| `SALES_FORECAST[TerritoryKey]` | `DIM_TERRITORY[TerritoryKey]` |

Repeating product, territory, and week keys are expected in the forecast table. Do not remove forecast rows merely because one of these keys repeats. Retain the report's explicit scenario selection so totals do not accidentally combine alternative discount scenarios.

Select **Close & Apply**. For later publications, wait for `PUBLISHED`, then select **Refresh**. File selection, M-code changes, and manual run-ID edits are not required again while the schemas and project location stay compatible.

## 9. Subsequent runs

### Publish a run that has already finished

Run the publisher with the matching prepared, results, and Page 1 directories. Publishing existing outputs does not retrain the model or create a new forecast period.

### Train, verify, forecast, and publish together

From an existing prepared dataset:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Run_And_Publish_BI_v2.ps1 -Retrain -Prepared prepared_v2 -Start 2014-07-07 -Weeks 4 -SqlServer "lpc:localhost\MSSQLSERVER01"
```

The wrapper creates new results/scenario folders under `runs/<timestamp>/`, runs the verifier, generates forecasts, and publishes only after those stages succeed. If `-RunId` is omitted, it generates an ID from the current timestamp. A specific unused ID can be supplied with `-RunId`.

Without `-Retrain`, the wrapper publishes the existing folders specified by `-Prepared`, `-Results`, and `-Page1`. Without `-SqlServer`, it uses CSV-only publication.

The wrapper starts from prepared data. It does not restore databases, run SSIS, export new warehouse inputs, or rerun preparation. When source data changes, refresh the warehouse exports and run preparation into a **new folder**, then pass that folder with `-Prepared`. Adjust `-Start` so forecast weeks follow the updated training period.

Training a different population, such as Individual customers or additional products, requires appropriate preparation changes and a new evaluation. Changing a Power BI slicer does not change the fitted model's scope.

## 10. Saved history and refresh behaviour

| Location | What is retained |
| --- | --- |
| `dbo.SALES_FORECAST` | Previous forecast batches distinguished by `ForecastRunID`. |
| `BI_Data/batches/<publication>/` | Published CSVs, metrics, model metadata, scenario metadata, generated import SQL, and publication details. |
| `BI_Data/current.json` | Pointer to the currently selected publication. It is updated after successful publication. |
| Original results folders / `runs/` | Fitted model files and run outputs, provided those folders are retained. |

Publishing the same outputs again creates another publication snapshot while the identical SQL batch remains free of duplicate rows. The default BI queries display the current publication rather than summing every historical run. Keep original results folders if fitted models need to be restored; the publication snapshots contain model metadata, not `model.joblib` itself.

Power BI Desktop in Import mode still requires **Refresh** after publication. This repository automates output checks, optional SQL loading, and current-batch selection; it does not automate the Desktop refresh button, run the model from slicers, or configure a refresh schedule. Finish publication before refreshing the report.

## 11. Reference results and interpretation

The reference V2 evaluation fits through the week starting **24 June 2013** and evaluates 44 subsequent weeks, **1 July 2013–28 April 2014**. The forecast model is then refitted through **28 April 2014**.

| Model | Holdout MAE | Holdout RMSE | Holdout R² |
| --- | ---: | ---: | ---: |
| Original OLS | 2.393 | 4.235 | 0.021 |
| Calendar-conditioned mean baseline | 0.989 | 3.266 | 0.417 |
| Selected calendar Ridge + gate | 0.931 | 3.110 | 0.472 |

Use your run's `metrics.json` as the authority for reported results. Scores are calculated at product–territory–week grain; weekly-total charts use a different aggregation. MAE and RMSE are in sales units, and R² is not an accuracy percentage or confidence rate.

The selected α = 10 candidate is within 0.1% of the lowest pooled validation RMSE and is chosen under the documented near-tie rule. It is not the absolute minimum-RMSE candidate. Much of the improvement over the original specification comes from calendar features, interactions, and the gate.

Holdout results were inspected during model development, so this is a retrospective evaluation rather than a fresh independent test. The source has unusual month-end concentration, and discount evidence is too sparse and confounded to support causal promotion conclusions. Later independent data, verified campaign timing, and product availability information are needed for stronger evaluation.

## 12. Troubleshooting

| Message or symptom | Action |
| --- | --- |
| `py` is not recognized | Use `python` to create the environment, then use `.\.venv\Scripts\python.exe`. |
| Missing input CSV or source-hash mismatch | Put the matching original exports in `inputs/`; do not mix sources and prepared data from different runs. |
| Preparation output folder is not empty | Choose a new `--out` directory and use it in downstream commands. |
| Model and forecast come from different runs | Regenerate Page 1 using the `model.joblib` from the selected results directory. |
| Forecast weeks overlap fitted history | Select a Monday after the model's final fitted week. |
| Unsupported product/category/discount | Use supported inputs or prepare and evaluate a model for the expanded scope. |
| `Keyword not supported: 'DataSource'` | Update `Import_Forecast_v2.ps1` to the fixed version using keys such as `$builder['Data Source']` and `$builder['Initial Catalog']`. |
| SQL run ID already contains different data | Generate a new forecast batch in a new folder with an unused run ID. |
| `ProposalID`, `OriginalPredictedUnits`, or `Model` column not found | Use the matching publisher-generated M query and V2 outputs; avoid CSV readers with stale fixed column counts. |
| `Cannot convert the value null to type Text` | Check parameter Current Values first, then identify the first failing Applied Step. Setting Type Text alone does not populate a blank value. |
| Duplicate date/product key on the one side | Correct the relationship to `*:1`, with the dimension on the `1` side. |
| Privacy / `Formula.Firewall` error | Review privacy settings for the local publication files and SQL source used by the same report. |
| Publisher reports `PUBLISHED` but BI still shows old data | Complete the one-time query setup, confirm the BI_Data path, and refresh after the publisher finishes. |

## 13. Files to keep outside Git

Exclude `.venv/`, `__pycache__/`, `inputs/*.csv`, generated prepared/results/scenario folders, `runs/`, `BI_Data/`, fitted `.joblib` files, and database backups from the source repository. Distribute the required course data separately as appropriate. Keep the source scripts, dependency list, example scenarios, and this README in Git.
