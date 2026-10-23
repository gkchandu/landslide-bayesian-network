import pandas as pd

PATH = "data/raw/Global_Landslide_Catalog_Export.csv"

df = pd.read_csv(PATH)

print("Shape:", df.shape)
print("\nColumns:\n", df.columns.tolist())
print("\nMissing values:\n", df.isna().sum())

country_cols = [c for c in df.columns if "country" in c.lower()]
print("\nCountry-like columns:", country_cols)

if country_cols:
    col = country_cols[0]
    print("\nTop countries:\n", df[col].value_counts().head(10))
    india = df[df[col].astype(str).str.contains("India", case=False, na=False)]
    print("\nIndia records:", india.shape[0])
    print("\nSample India rows:\n", india.head(3).T)

for key in ["trigger", "category", "size", "fatality"]:
    for c in [c for c in df.columns if key in c.lower()]:
        print(f"\n--- {c} ---")
        print(df[c].value_counts(dropna=False).head(15))