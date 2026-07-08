import pandas as pd

listings = pd.read_csv("data/combined_listings.csv", low_memory=False)
sold = pd.read_csv("data/combined_sold.csv", low_memory=False)

# 1. Dataset Shape 
print("=" * 60)
print("1. DATASET SHAPE")
print(f"Listings: {listings.shape[0]:,} rows, {listings.shape[1]} columns")
print(f"Sold:     {sold.shape[0]:,} rows, {sold.shape[1]} columns")

# 2. Unique Property Types 
print("\n" + "=" * 60)
print("2. UNIQUE PROPERTY TYPES")
print("Listings:", listings["PropertyType"].unique())
print("Sold:    ", sold["PropertyType"].unique())

# 3. Column Data Types 
print("\n" + "=" * 60)
print("3. COLUMN DATA TYPES (Sold)")
print(sold.dtypes.to_string())

# 4. Missing Value Report
print("4. MISSING VALUE REPORT")
def missing_report(df, name):
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    report = pd.DataFrame({"missing_count": missing, "missing_pct": pct})
    report = report[report["missing_count"] > 0].sort_values("missing_pct", ascending=False)

    print(f"-- {name}: All columns with missing values --")
    print(report.to_string())

    high_missing = report[report["missing_pct"] > 90]
    print(f"-- {name}: Columns >90% missing ({len(high_missing)} columns) --")
    print(high_missing.index.tolist())

    return report

listings_missing = missing_report(listings, "Listings")
sold_missing = missing_report(sold, "Sold")

# Save missing reports
listings_missing.to_csv("data/listings_missing_report.csv")
sold_missing.to_csv("data/sold_missing_report.csv")
print("\nMissing reports saved to data/")

# 5. Numeric Distribution Summary
print("5. NUMERIC DISTRIBUTION SUMMARY (Sold)")

numeric_fields = [
    "ClosePrice", "ListPrice", "OriginalListPrice",
    "LivingArea", "LotSizeAcres", "BedroomsTotal",
    "BathroomsTotalInteger", "DaysOnMarket", "YearBuilt"
]

# Only include columns that is exist in the dataset
existing_fields = [f for f in numeric_fields if f in sold.columns]
print(sold[existing_fields].describe(percentiles=[.10, .25, .50, .75, .90]).round(2).to_string())

# 6. EDA Questions 
print("6. EDA INSIGHTS")

import glob

sold_files = sorted(glob.glob("data/Sold/CRMLSSold*.csv"))
sold_raw = pd.concat(
    [pd.read_csv(f, low_memory=False) for f in sold_files],
    ignore_index=True
)

# (1)
print("(1)PROPERTY TYPE SHARE (Raw Sold Data)")
type_counts = sold_raw["PropertyType"].value_counts()
type_pct = (sold_raw["PropertyType"].value_counts(normalize=True) * 100).round(2)
type_summary = pd.DataFrame({"count": type_counts, "percentage": type_pct})
print(type_summary.to_string())

# (2)
print(f"\n(2)Median and average close price\nMedian ClosePrice:  ${sold['ClosePrice'].median():,.0f}")
print(f"Average ClosePrice: ${sold['ClosePrice'].mean():,.0f}")

# (3)
print("(3)DAYS ON MARKET DISTRIBUTION")
dom = sold["DaysOnMarket"].dropna()
print(f"Min:    {dom.min():.0f} days")
print(f"Max:    {dom.max():.0f} days")
print(f"Mean:   {dom.mean():.1f} days")
print(f"Median: {dom.median():.0f} days")
print(f"10th percentile: {dom.quantile(0.10):.0f} days")
print(f"25th percentile: {dom.quantile(0.25):.0f} days")
print(f"75th percentile: {dom.quantile(0.75):.0f} days")
print(f"90th percentile: {dom.quantile(0.90):.0f} days")

# (4) 
print("(4)SOLD ABOVE VS. BELOW LIST PRICE")
valid = sold.dropna(subset=["ClosePrice", "ListPrice"])
above = (valid["ClosePrice"] > valid["ListPrice"]).sum()
below = (valid["ClosePrice"] < valid["ListPrice"]).sum()
at    = (valid["ClosePrice"] == valid["ListPrice"]).sum()
total = len(valid)
print(f"Sold ABOVE list price: {above:,} ({above/total*100:.1f}%)")
print(f"Sold AT list price:    {at:,}    ({at/total*100:.1f}%)")
print(f"Sold BELOW list price: {below:,} ({below/total*100:.1f}%)")

# (5)
print("(5)DATE CONSISTENCY ISSUES")
sold["CloseDate"] = pd.to_datetime(sold["CloseDate"], errors="coerce")
sold["ListingContractDate"] = pd.to_datetime(sold["ListingContractDate"], errors="coerce")
sold["PurchaseContractDate"] = pd.to_datetime(sold["PurchaseContractDate"], errors="coerce")

close_before_listing = (sold["CloseDate"] < sold["ListingContractDate"]).sum()
close_before_purchase = (sold["CloseDate"] < sold["PurchaseContractDate"]).sum()
purchase_before_listing = (sold["PurchaseContractDate"] < sold["ListingContractDate"]).sum()

print(f"CloseDate before ListingContractDate:   {close_before_listing:,} records")
print(f"CloseDate before PurchaseContractDate:  {close_before_purchase:,} records")
print(f"PurchaseDate before ListingDate:        {purchase_before_listing:,} records")

# (6)
print("(6)TOP 10 COUNTIES BY MEDIAN CLOSE PRICE")
county_median = (
    sold.groupby("CountyOrParish")["ClosePrice"]
    .median()
    .sort_values(ascending=False)
    .head(10)
)
print(county_median.apply(lambda x: f"${x:,.0f}").to_string())

# Drop >90% missing columns, keep only dashboard-relevant fields
cols_to_keep = [
    # Transaction
    "CloseDate", "ClosePrice", "ListPrice", "OriginalListPrice",
    "DaysOnMarket", "LivingArea", "BedroomsTotal", "BathroomsTotalInteger",
    "YearBuilt", "LotSizeAcres",
    # Geography
    "CountyOrParish", "MLSAreaMajor", "PostalCode", "City",
    "StateOrProvince", "UnparsedAddress", "Latitude", "Longitude",
    # Agent / Brokerage
    "ListAgentFirstName", "ListAgentLastName", "ListAgentEmail",
    "ListOfficeName", "BuyerAgentFirstName", "BuyerAgentLastName", "BuyerOfficeName",
    # Classification
    "PropertyType", "PropertySubType", "ListingKey",
    # Dates
    "ListingContractDate", "PurchaseContractDate", "ContractStatusChangeDate"
]

sold_cols = [c for c in cols_to_keep if c in sold.columns]
listings_cols = [c for c in cols_to_keep if c in listings.columns]

sold_clean = sold[sold_cols].copy()
listings_clean = listings[listings_cols].copy()

print(f"Sold:     {sold.shape[1]} columns → {sold_clean.shape[1]} columns kept")
print(f"Listings: {listings.shape[1]} columns → {listings_clean.shape[1]} columns kept")

# Save
sold_clean.to_csv("data/sold_eda.csv", index=False)
listings_clean.to_csv("data/listings_eda.csv", index=False)

