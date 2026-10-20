"""V2 - Prepare a retrospective weekly LR dataset from a validated warehouse export.

V2 changes: grid from Reseller orders; partial campaign/product weeks kept (reference order day);
CampaignCoverageDays column; source-to-training unit reconciliation in data_audit.json.

Targets always use FACT_SALES_LINE.OrderQty. Source columns only validate or
identify unavailable target groups; they never fill a missing warehouse sale.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SALES_COLUMNS = "SalesOrderID SalesOrderDetailID OrderDate OrderStatus OnlineOrderFlag ProductID ProductName Category TerritoryID TerritoryName CustomerType OrderQty UnitPrice UnitPriceDiscount LineTotal SpecialOfferID OfferDescription OfferType OfferCategory CatalogDiscountPct OfferStartDate OfferEndDate MinQty MaxQty".split()
PLAN_COLUMNS = "SpecialOfferID ProductID ProductName Category ProductSellStartDate ProductSellEndDate ProductDiscontinuedDate OfferDescription OfferType OfferCategory CatalogDiscountPct OfferStartDate OfferEndDate MinQty MaxQty InvalidQtyRange".split()
GRAIN = ["WeekStart", "ProductID", "TerritoryID"]
WAREHOUSE_COLUMNS = "WarehousePresent SalesLineKey WarehouseDateKey WarehouseOrderDate WarehouseProductKey WarehouseProductID WarehouseCategory WarehouseCustomerKey WarehouseCustomerID WarehouseCustomerType SourceCustomerID WarehousePromotionKey WarehouseSpecialOfferID WarehouseCatalogDiscountPct WarehouseTerritoryKey WarehouseTerritoryID SourceOrderQty SourceLineTotal WarehouseGrossSales WarehouseDiscountAmount WarehouseMappingIssue".split()
MONEY_ATOL = 0.00011  # DW decimal(18,4); observed source/net difference <= 0.000096.


def read_export(path, columns):
    frame = pd.read_csv(path, header=None, encoding="utf-8-sig", na_values=["NULL"])
    if frame.shape[1] != len(columns):
        raise ValueError(f"Expected {len(columns)} columns in {path}; got {frame.shape[1]}")
    if frame.iloc[0].astype(str).tolist() == columns:
        frame = frame.iloc[1:].reset_index(drop=True)
    frame.columns = columns
    return frame


def validate_warehouse(s):
    """Reject broken mappings before scope filtering, including unused products."""
    for c in WAREHOUSE_COLUMNS:
        if c not in ("WarehouseOrderDate", "WarehouseCategory", "WarehouseCustomerType"):
            s[c] = pd.to_numeric(s[c], errors="raise")
    if s.empty or s.WarehousePresent.isna().any() or not s.WarehousePresent.isin([0, 1]).all():
        raise ValueError("WarehousePresent must be 0 or 1 for every source detail.")
    loaded = s.WarehousePresent.eq(1)
    if not loaded.any():
        raise ValueError("There are no loaded warehouse facts.")
    f = s.loc[loaded]
    required = ["SalesLineKey", "WarehouseDateKey", "WarehouseOrderDate",
                "WarehouseProductKey", "WarehouseProductID", "WarehouseCategory",
                "WarehouseCustomerKey", "WarehouseCustomerID", "WarehouseCustomerType",
                "SourceCustomerID", "WarehousePromotionKey", "WarehouseSpecialOfferID",
                "WarehouseCatalogDiscountPct", "WarehouseTerritoryKey",
                "WarehouseTerritoryID", "WarehouseMappingIssue"]
    if f[required].isna().any().any() or f.SalesLineKey.duplicated().any():
        raise ValueError("Missing/duplicate warehouse identifiers or dimension metadata.")
    if f.WarehouseMappingIssue.ne(0).any():
        raise ValueError("WarehouseMappingIssue is nonzero. Repair ETL keys before training.")
    for source, dim in [("ProductID", "WarehouseProductID"),
                        ("SourceCustomerID", "WarehouseCustomerID"),
                        ("TerritoryID", "WarehouseTerritoryID"),
                        ("SpecialOfferID", "WarehouseSpecialOfferID"),
                        ("Category", "WarehouseCategory"),
                        ("CustomerType", "WarehouseCustomerType")]:
        if f[source].ne(f[dim]).any():
            raise ValueError(f"Warehouse mapping mismatch: {source} versus {dim}.")
    dates = pd.to_datetime(f.WarehouseOrderDate, errors="raise").dt.normalize()
    if dates.ne(f.OrderDate.dt.normalize()).any():
        raise ValueError("Warehouse order date differs from the source date.")
    if not np.isclose(f.CatalogDiscountPct, f.WarehouseCatalogDiscountPct,
                      atol=MONEY_ATOL, rtol=0).all():
        raise ValueError("Warehouse promotion rate differs from the source catalogue.")
    # Quantity and net sales must reconcile for every loaded row, even if the
    # source values are invalid for ML and will be quarantined afterward.
    if not np.isclose(f.OrderQty, f.SourceOrderQty, atol=0, rtol=0, equal_nan=True).all():
        raise ValueError("Loaded warehouse quantities differ from the source.")
    if not np.isclose(f.LineTotal, f.SourceLineTotal,
                      atol=MONEY_ATOL, rtol=0, equal_nan=True).all():
        raise ValueError("Warehouse net sales differ from source beyond decimal(18,4) tolerance.")
    absent = s.loc[~loaded]
    if absent[["SalesLineKey", "OrderQty", "LineTotal", "WarehouseGrossSales",
               "WarehouseDiscountAmount"]].notna().any().any():
        raise ValueError("A missing fact unexpectedly contains warehouse measures or a row key.")
    return loaded


def prepare(args):
    required_assumptions = [args.assume_complete_export, args.assume_catalog_availability,
                            args.assume_calendar_known, args.allow_snapshot_category]
    if not all(required_assumptions):
        raise ValueError("This retrospective experiment requires all four explicit assumption flags. Read README first.")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Preparation output folder is not empty. Use a new --out folder.")
    out.mkdir(parents=True, exist_ok=True)
    s = pd.read_csv(args.sales, encoding="utf-8-sig", na_values=["NULL"])
    if s.columns.tolist() != SALES_COLUMNS + WAREHOUSE_COLUMNS:
        raise ValueError("Expected the 45-column warehouse export WITH HEADERS. Use the included SQL export.")
    p = read_export(args.eligibility, PLAN_COLUMNS)
    for c in ["SalesOrderID", "SalesOrderDetailID", "OrderStatus", "OnlineOrderFlag", "ProductID", "TerritoryID", "OrderQty", "UnitPrice", "UnitPriceDiscount", "LineTotal", "SpecialOfferID", "CatalogDiscountPct", "MinQty", "MaxQty"]:
        s[c] = pd.to_numeric(s[c], errors="raise")
    for c in ["SpecialOfferID", "ProductID", "CatalogDiscountPct", "MinQty", "MaxQty", "InvalidQtyRange"]:
        p[c] = pd.to_numeric(p[c], errors="raise")
    for c in ["OrderDate", "OfferStartDate", "OfferEndDate"]:
        s[c] = pd.to_datetime(s[c], errors="raise")
    for c in ["ProductSellStartDate", "ProductSellEndDate", "ProductDiscontinuedDate", "OfferStartDate", "OfferEndDate"]:
        p[c] = pd.to_datetime(p[c], errors="raise")
    loaded = validate_warehouse(s)
    if s.duplicated(["SalesOrderID", "SalesOrderDetailID"]).any() or p.duplicated(["SpecialOfferID", "ProductID"]).any():
        raise ValueError("Duplicate source keys; investigate before preparing data.")
    essential = ["SalesOrderID", "SalesOrderDetailID", "OrderDate", "OrderStatus", "ProductID", "TerritoryID", "SpecialOfferID", "Category", "CustomerType"]
    if s[essential].isna().any().any():
        raise ValueError("Missing key/date/classification values; investigate first.")
    if s.groupby("ProductID").Category.nunique().gt(1).any():
        raise ValueError("Multiple product categories require an explicit historical mapping.")
    s["WeekStart"] = s.OrderDate - pd.to_timedelta(s.OrderDate.dt.dayofweek, unit="D")
    num = s[["OrderQty", "UnitPrice", "UnitPriceDiscount", "LineTotal",
             "WarehouseGrossSales", "WarehouseDiscountAmount"]]
    missing = num.isna().any(axis=1) | ~np.isfinite(num.to_numpy(dtype=float)).all(axis=1)
    invalid = ((s.OrderQty <= 0) | s.OrderQty.mod(1).ne(0) | (s.UnitPrice < 0)
               | ~s.UnitPriceDiscount.between(0, 1) | (s.LineTotal < 0)
               | (s.WarehouseGrossSales < 0) | (s.WarehouseDiscountAmount < 0)) & ~missing
    formula_bad = ((~np.isclose(s.LineTotal, s.OrderQty * s.UnitPrice * (1 - s.UnitPriceDiscount), atol=MONEY_ATOL, rtol=0))
                   | (~np.isclose(s.WarehouseGrossSales, s.OrderQty * s.UnitPrice, atol=MONEY_ATOL, rtol=0))
                   | (~np.isclose(s.WarehouseDiscountAmount, s.WarehouseGrossSales - s.LineTotal, atol=MONEY_ATOL, rtol=0))) & ~missing
    bad = ~loaded | missing | invalid | formula_bad
    reason = pd.Series("", index=s.index)
    for mask, label in [(~loaded, "missing_warehouse_fact"), (missing, "missing_numeric"), (invalid, "invalid_numeric"), (formula_bad, "formula_mismatch")]:
        reason.loc[mask] += label + ";"
    s.assign(QualityIssue=reason).loc[bad].to_csv(out / "quarantined_sales_lines.csv", index=False)
    p["InvalidDates"] = p.OfferStartDate.isna() | p.OfferEndDate.isna() | (p.OfferEndDate < p.OfferStartDate)
    p["InvalidQuantity"] = p.MinQty.isna() | (p.MinQty < 0) | (p.MaxQty < 0) | (p.MaxQty.notna() & (p.MaxQty < p.MinQty))
    p["InvalidRate"] = ~p.CatalogDiscountPct.between(0, 1)
    p.loc[p.InvalidDates | p.InvalidQuantity | p.InvalidRate].to_csv(out / "quarantined_offer_products.csv", index=False)
    campaign = p.OfferCategory.eq("Reseller") & p.MinQty.eq(0) & p.MaxQty.isna() & p.CatalogDiscountPct.gt(0)
    # An invalid non-volume campaign could make a 'no campaign' label incorrect.
    # Remove all associated products rather than repairing source dates by guessing.
    invalid_calendar_products = set(p.loc[campaign & (p.InvalidDates | p.InvalidRate), "ProductID"])
    c = p.loc[campaign & ~p.InvalidDates & ~p.InvalidQuantity & ~p.InvalidRate & ~p.ProductID.isin(invalid_calendar_products)].copy()
    c.to_csv(out / "campaign_calendar_used.csv", index=False)
    products = p[p.ProductID.isin(c.ProductID)].drop_duplicates("ProductID")
    if products.empty:
        raise ValueError("No eligible campaign products.")
    # V2: the weekly grid spans the modelled population (Reseller, Shipped), not every customer type.
    # It starts at the Monday of the first order and ends at the Monday of the last order, so no week
    # before/after the observed Reseller orders is zero-filled and no real order week is dropped.
    in_scope_dates = s.loc[s.CustomerType.eq("Reseller") & s.OrderStatus.eq(5), "OrderDate"]
    first = in_scope_dates.min() - pd.to_timedelta(in_scope_dates.min().dayofweek, unit="D")
    last = in_scope_dates.max() - pd.to_timedelta(in_scope_dates.max().dayofweek, unit="D")
    weeks = pd.date_range(first, last, freq="W-MON")
    territories = sorted(s.TerritoryID.unique())
    calendar = []
    ambiguous = partial_kept = 0

    def reference_day(start):
        """Orders are booked around month end. Use the month-end day inside the week, otherwise Thursday.
        This date is known before the week, so it is a legitimate forecast-time feature."""
        for k in range(7):
            day = start + pd.Timedelta(days=k)
            if day.is_month_end:
                return day
        return start + pd.Timedelta(days=3)

    for row in products.itertuples():
        offers = c[c.ProductID.eq(row.ProductID)]
        for start in weeks:
            end = start + pd.Timedelta(days=6)
            # V2: a product-week is kept when the product can be sold on ANY day of the week.
            # (V1 required the whole week, which deleted real sales in launch/discontinuation weeks.)
            if pd.isna(row.ProductSellStartDate) or row.ProductSellStartDate > end:
                continue
            if pd.notna(row.ProductSellEndDate) and row.ProductSellEndDate <= start:
                continue
            if pd.notna(row.ProductDiscontinuedDate) and row.ProductDiscontinuedDate <= start:
                continue
            ref = reference_day(start)
            overlap = offers[(offers.OfferStartDate <= ref) & (offers.OfferEndDate >= ref)]
            if len(overlap) > 1:
                ambiguous += 1
                continue
            coverage = 0
            if len(overlap) == 1:
                offer = overlap.iloc[0]
                days = pd.date_range(start, end)
                coverage = int(((days >= offer.OfferStartDate) & (days <= offer.OfferEndDate)).sum())
                if coverage < 7:
                    partial_kept += 1
                rate, oid = float(offer.CatalogDiscountPct), int(offer.SpecialOfferID)
            else:
                rate, oid = 0.0, 0
            for territory in territories:
                calendar.append([start, int(row.ProductID), row.Category, int(territory), rate, oid, coverage])
    d = pd.DataFrame(calendar, columns=["WeekStart", "ProductID", "Category", "TerritoryID", "ScheduledCampaignDiscountPct", "CalendarOfferID", "CampaignCoverageDays"])
    reseller_shipped = s.CustomerType.eq("Reseller") & s.OrderStatus.eq(5)
    # Dropping only a bad row would understate its weekly target. Drop its entire target group.
    contaminated = s.loc[reseller_shipped & bad, GRAIN].drop_duplicates().assign(Contaminated=True)
    totals = s.loc[reseller_shipped & ~bad].groupby(GRAIN, as_index=False).OrderQty.sum().rename(columns={"OrderQty": "UnitsSold"})
    d = d.merge(contaminated, on=GRAIN, how="left", validate="one_to_one")
    contaminated_count = int(d.Contaminated.eq(True).sum())
    d = d.loc[d.Contaminated.ne(True)].drop(columns="Contaminated")
    d = d.merge(totals, on=GRAIN, how="left", validate="one_to_one")
    d["ObservedSalesGroup"] = d.UnitsSold.notna()
    d["UnitsSold"] = d.UnitsSold.fillna(0).astype(int)
    d["DataOrigin"] = "REAL_COMPANYX_DWH_RETROSPECTIVE_ASSUMPTIONS"
    d["CustomerType"] = "Reseller"
    d = d.sort_values(GRAIN).reset_index(drop=True)
    if d.duplicated(GRAIN).any() or d.WeekStart.dt.dayofweek.ne(0).any() or (d.UnitsSold < 0).any():
        raise ValueError("Invalid prepared dataset.")
    # Reconcile retained targets to source rows at the same declared grain.
    direct = totals.merge(d[GRAIN], on=GRAIN, how="inner", validate="one_to_one")
    if int(d.UnitsSold.sum()) != int(direct.UnitsSold.sum()):
        raise ValueError("Target total reconciliation failed.")
    in_scope = s.loc[reseller_shipped & ~bad & s.ProductID.isin(d.ProductID.unique())]
    kept_units = int(d.UnitsSold.sum())
    in_scope_units = int(in_scope.OrderQty.sum())
    lost = totals.merge(d[GRAIN], on=GRAIN, how="left", indicator=True)
    lost = lost.loc[lost._merge.eq("left_only") & lost.ProductID.isin(d.ProductID.unique())]
    reconciliation = {
        "source_units_reseller_shipped_in_scope_products": in_scope_units,
        "units_in_training": kept_units,
        "units_not_in_training": in_scope_units - kept_units,
        "share_not_in_training": round((in_scope_units - kept_units) / in_scope_units, 6),
        "product_weeks_not_in_training": int(lost[["WeekStart", "ProductID"]].drop_duplicates().shape[0]),
    }
    d.to_csv(out / "training.csv", index=False, date_format="%Y-%m-%d")
    offers = p.drop_duplicates("SpecialOfferID")
    audit = {
        "source_files": [{"name": Path(f).name, "sha256": hashlib.sha256(Path(f).read_bytes()).hexdigest()} for f in [args.sales, args.eligibility]],
        "source_sales_rows": len(s), "source_orders": int(s.SalesOrderID.nunique()),
        "loaded_warehouse_rows": int(loaded.sum()),
        "source_details_missing_from_warehouse": int((~loaded).sum()),
        "loaded_warehouse_mapping_issues": int(s.loc[loaded, "WarehouseMappingIssue"].ne(0).sum()),
        "target_source": "FACT_SALES_LINE.OrderQty from the validated DWH export; no fallback to source quantities.",
        "warehouse_money_absolute_tolerance": MONEY_ATOL,
        "max_loaded_net_sales_source_difference": float((s.loc[loaded, "LineTotal"] - s.loc[loaded, "SourceLineTotal"]).abs().max()),
        "warehouse_product_scd_evidence": "The user's repair COMMITTED result reported zero mapping issues after as-of-date validation. This 45-column export itself does not contain ValidFrom/ValidTo.",
        "source_offer_product_pairs": len(p), "source_duplicate_detail_keys": 0,
        "source_date_min": str(s.OrderDate.min().date()), "source_date_max": str(s.OrderDate.max().date()),
        "source_status_counts": {str(k): int(v) for k, v in s.OrderStatus.value_counts().items()},
        "missing_quantity_rows": int(s.OrderQty.isna().sum()), "missing_unit_price_rows": int(s.UnitPrice.isna().sum()),
        "negative_quantity_rows": int((s.OrderQty < 0).sum()), "zero_quantity_rows": int(s.OrderQty.eq(0).sum()), "negative_unit_price_rows": int((s.UnitPrice < 0).sum()),
        "quarantined_numeric_rows": int(bad.sum()), "formula_mismatch_rows_with_nonmissing_inputs": int(formula_bad.sum()),
        "quarantined_loaded_warehouse_rows": int((bad & loaded).sum()),
        "catalog_vs_applied_rate_mismatch_rows": int((~np.isclose(s.CatalogDiscountPct, s.UnitPriceDiscount)).sum()),
        "invalid_quantity_offer_ids": sorted(offers.loc[offers.InvalidQuantity, "SpecialOfferID"].astype(int).tolist()),
        "invalid_date_offer_ids": sorted(offers.loc[offers.InvalidDates, "SpecialOfferID"].astype(int).tolist()),
        "campaign_offer_ids_used": sorted(c.SpecialOfferID.astype(int).unique().tolist()),
        "products_excluded_for_invalid_nonvolume_calendar": sorted(map(int, invalid_calendar_products)),
        "reconciliation_to_source": reconciliation,
        "weekly_grid_basis": "Reseller shipped order dates (first and last order week)",
        "prepared_rows": len(d), "prepared_products": int(d.ProductID.nunique()), "prepared_territories": len(territories),
        "prepared_weeks": int(d.WeekStart.nunique()), "prepared_first_week": str(d.WeekStart.min().date()), "prepared_last_week": str(d.WeekStart.max().date()),
        "prepared_total_units": int(d.UnitsSold.sum()), "prepared_zero_target_rows": int(d.UnitsSold.eq(0).sum()),
        "zero_filled_rows_without_source_sales_group": int((~d.ObservedSalesGroup).sum()),
        "campaign_active_rows": int(d.ScheduledCampaignDiscountPct.gt(0).sum()),
        "campaign_active_weeks": int(d.loc[d.ScheduledCampaignDiscountPct.gt(0), "WeekStart"].nunique()),
        "kept_partial_campaign_product_weeks": partial_kept, "excluded_ambiguous_campaign_product_weeks": ambiguous,
        "excluded_contaminated_target_groups": contaminated_count,
        "assumptions": [
            "The source-anchored exports cover the observed interval. Counts were checked against supplied SQL results; the author did not connect to SQL Server. Missing DW details are explicit and contaminate their entire target group, never become zero sales.",
            "Each catalogue-active product can be sold in all ten observed territories; catalogue dates do not prove local stock or territory availability.",
            "The observed offer calendar and product mappings were known before each forecast week; a single backup cannot verify publication/revision history.",
            "The initial observed product Category is assumed to apply to earlier sales. The DW repair extended initial ValidFrom to the earliest source order per sold product, without reconstructing unobserved category changes.",
            "After excluding whole target groups with missing or invalid warehouse facts, other absent eligible sales groups are zero only under the source completeness and territory availability assumptions.",
        ],
        "limitations": [
            "ScheduledCampaignDiscountPct is an offered non-volume campaign rate, not the applied transaction rate or a sales-weighted average.",
            "Only Reseller target sales are modelled. Quantity-tier offers are excluded from the calendar feature; valid sales using them remain in total target units.",
            "Zero means no qualifying non-volume campaign in this scope, not that no discounts were possible.",
            "Catalogue campaigns with no observed use remain on the calendar; offer availability does not prove customer exposure or redemption.",
            "The experiment is retrospective and conditional on unverified availability/publication assumptions. It is not production forecast validation or causal promotion lift.",
        ],
    }
    (out / "data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ["prepared_rows", "prepared_products", "prepared_weeks", "prepared_total_units", "campaign_active_rows", "excluded_contaminated_target_groups"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sales", required=True)
    parser.add_argument("--eligibility", required=True)
    parser.add_argument("--out", default="prepared")
    parser.add_argument("--assume-source-export-complete", dest="assume_complete_export", action="store_true")
    parser.add_argument("--assume-catalog-availability", action="store_true")
    parser.add_argument("--assume-calendar-known", action="store_true")
    parser.add_argument("--allow-snapshot-category", action="store_true")
    prepare(parser.parse_args())
