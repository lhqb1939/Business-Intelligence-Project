# ETL Pipeline & Data Warehouse Scripts
Package Components:
DW_Promotion_Analytics.bak (Target Data Warehouse Backup file containing the completed Star Schema and 121,317 rows of processed data)
ETL_Promotion.7z (Visual Studio SSIS Project source code including the automated ETL Data Flow pipeline)
CompanyX.bak (Source OLTP Database backup for source validation or re-running ETL packages)

## STAGE 1: Database Restoration in SSMS 
    Step 1.1: Restore the Data Warehouse (DW_Promotion_Analytics)
    This is the primary repository that powers downstream dashboards and predictive modeling.
        - Launch SQL Server Management Studio (SSMS) and connect to your local SQL Server instance.
        - In the Object Explorer panel on the left, right-click the Databases folder  select Restore Database...
        - In the dialog window:
            + Select the Device option  click the ... (browse) button.
            + Click Add  locate DW_Promotion_Analytics.bak  click OK  click OK.
        - Click OK to execute the restore process. A confirmation prompt will state: "Database 'DW_Promotion_Analytics' restored successfully."
    Step 1.2: Restore the OLTP Source Database (CompanyX)
    This database acts as the operational source system (required if you want to inspect source data or execute the SSIS package).
        - Right-click the Databases folder  select Restore Database...
        - Select Device  click Add  select CompanyX.bak  click OK.
        - Click OK to complete the database restoration.
    Step 1.3: Verify Data Integrity
    Open a New Query window in SSMS, paste the script below, and run Execute (F5):
    USE DW_Promotion_Analytics;
    GO
    
    -- 1. Check total rows loaded into the Fact table
    SELECT COUNT(*) AS TotalFactRows FROM FACT_SALES_LINE;
    
    -- 2. Verify financial metrics integrity (Ensure no NULL values or numeric overflows)
    SELECT TOP 10 
        SalesOrderID,
        ProductKey,
        CustomerKey,
        PromotionKey,
        OrderQty,
        GrossSales,
        DiscountAmount,
        NetSales
    FROM FACT_SALES_LINE
    ORDER BY SalesLineKey DESC;
    
    -- 3. Reconcile actual revenue by promotion
    SELECT 
        p.PromotionKey,
        p.OfferType,
        p.Category,
        p.DiscountPct,
        COUNT(f.SalesLineKey) AS TotalOrders,
        SUM(f.GrossSales) AS TotalGrossSales,
        SUM(f.DiscountAmount) AS TotalDiscountGiven,
        SUM(f.NetSales) AS TotalNetRevenue
    FROM FACT_SALES_LINE f
    JOIN DIM_PROMOTION p ON f.PromotionKey = p.PromotionKey
    GROUP BY p.PromotionKey, p.OfferType, p.Category, p.DiscountPct
    ORDER BY TotalDiscountGiven DESC;


## STAGE 2: SSIS ETL Solution Setup in Visual Studio 
    Step 2.1: Extract the Source Code
        - Extract ETL_Promotion.7z into your local directory.
    Step 2.2: Open the Solution
        - Ensure your Visual Studio environment (2019 or 2022) has the SQL Server Integration Services Projects (SSIS) extension installed.
        - Open ETL_Promotion.slnx (or double-click Package.dtsx).
    Step 2.3: Reconfigure Connection Managers (Resolving Server Name Mismatch)
    Because server instance names differ across machines, update the server path for both connection managers:
        - In the Connection Managers window at the bottom of the designer:
        - Update the Source Connection:
            + Double-click the CompanyX connection.
            + Set Server name to . or localhost (or your local instance name).
            + Under Initial catalog, select CompanyX.
            + Click Test Connection  click OK.
        - Update the Destination Connection:
            + Double-click the DW_Promotion_Analytics connection.
            + Set Server name to . or localhost.
            + Under Initial catalog, select DW_Promotion_Analytics.
            + Click Test Connection  click OK.
        - Press Start (F5) to execute the ETL pipeline. All 4 transformation components (OLE DB Source  Derived Column  Lookup  OLE DB Destination) turning green checkmark → a successful run.

