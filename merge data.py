import pandas as pd
import glob
import os

# Listing
listing_files = sorted(glob.glob("data/Listing/CRMLSListing*.csv"))
print(f"Found {len(listing_files)} Listing files")

listings_raw = pd.concat(
    [pd.read_csv(f, low_memory=False) for f in listing_files],
    ignore_index=True
)
print(f"Listings row count before filter: {len(listings_raw):,}")

# sold
sold_files = sorted(glob.glob("data/Sold/CRMLSSold*.csv"))
print(f"Found {len(sold_files)} Sold files")

sold_raw = pd.concat(
    [pd.read_csv(f, low_memory=False) for f in sold_files],
    ignore_index=True
)
print(f"Sold row count before filter: {len(sold_raw):,}")

# Filter to Residential only
listings = listings_raw[listings_raw["PropertyType"] == "Residential"].copy()
sold      = sold_raw[sold_raw["PropertyType"] == "Residential"].copy()

print(f"Listings after Residential filter: {len(listings):,}  (removed {len(listings_raw)-len(listings):,} rows)")
print(f"Sold after Residential filter:     {len(sold):,}  (removed {len(sold_raw)-len(sold):,} rows)")

listings.to_csv("data/combined_listings.csv", index=False)
sold.to_csv("data/combined_sold.csv", index=False)