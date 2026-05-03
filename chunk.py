"""
This is chunk prep that you only need to run once: "python chunk.py". Run this before embed.py.
It reads the Superstore CSV file and converts it to chunks.
There are 5 chunks types: basic transactions, monthly, categories, regions and rankings.
Rankings has year, region profit rank and category profit rank.
Produces chunks.json that will be used by embed.py.
"""
import pandas as pd
import json

df = pd.read_csv("data/Sample - Superstore.csv", encoding="latin-1")
df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Ship Date"] = pd.to_datetime(df["Ship Date"])
df["Year"] = df["Order Date"].dt.year

print(f"{len(df)} rows, {df.shape[1]} columns")
print(df.dtypes)

# transaction chunks (one per row)
def transaction_to_text(row):
    return (
        f"On {row['Order Date'].strftime('%B %d, %Y')}, customer {row['Customer Name']} "
        f"({row['Segment']} segment) in {row['City']}, {row['State']} ({row['Region']} region) "
        f"ordered {row['Quantity']} unit(s) of '{row['Product Name']}' "
        f"(Category: {row['Category']}, Sub-Category: {row['Sub-Category']}). "
        f"Sales: ${row['Sales']:.2f}, Discount: {row['Discount']*100:.0f}%, "
        f"Profit: ${row['Profit']:.2f}. Ship Mode: {row['Ship Mode']}."
    )

transaction_chunks = df.apply(transaction_to_text, axis=1).tolist()
transaction_tags = df.apply(lambda r: {
    "type": "transactions",
    "year": str(r["Year"]),
    "region": r["Region"],
    "category": r["Category"],
    "sub_category": r["Sub-Category"],
}, axis=1).tolist()

# monthly rollups
df["YearMonth"] = df["Order Date"].dt.to_period("M")
monthly = df.groupby(["YearMonth", "Year"]).agg(
    total_sales=("Sales", "sum"),
    total_profit=("Profit", "sum"),
    num_orders=("Order ID", "nunique"),
    avg_discount=("Discount", "mean"),
).reset_index()

def monthly_to_text(row):
    margin = (row["total_profit"] / row["total_sales"] * 100) if row["total_sales"] else 0
    return (
        f"In {row['YearMonth']}, the store had {row['num_orders']} orders with "
        f"total sales of ${row['total_sales']:,.2f} and total profit of "
        f"${row['total_profit']:,.2f} (profit margin: {margin:.1f}%). "
        f"Average discount given: {row['avg_discount']*100:.1f}%."
    )

monthly_chunks = monthly.apply(monthly_to_text, axis=1).tolist()
monthly_tags = monthly.apply(lambda r: {"type": "monthly", "year": str(r["Year"])}, axis=1).tolist()

# category/subcategory rollups
cat_perf = df.groupby(["Category", "Sub-Category"]).agg(
    total_sales=("Sales", "sum"),
    total_profit=("Profit", "sum"),
    num_orders=("Order ID", "nunique"),
    avg_discount=("Discount", "mean"),
).reset_index()

def category_to_text(row):
    margin = (row["total_profit"] / row["total_sales"] * 100) if row["total_sales"] else 0
    return (
        f"Category '{row['Category']}' / Sub-Category '{row['Sub-Category']}': "
        f"total sales ${row['total_sales']:,.2f}, profit ${row['total_profit']:,.2f} "
        f"(margin {margin:.1f}%), across {row['num_orders']} orders. "
        f"Average discount: {row['avg_discount']*100:.1f}%."
    )

category_chunks = cat_perf.apply(category_to_text, axis=1).tolist()
category_tags = cat_perf.apply(lambda r: {
    "type": "categories",
    "category": r["Category"],
    "sub_category": r["Sub-Category"],
}, axis=1).tolist()

# region rollups
region_perf = df.groupby("Region").agg(
    total_sales=("Sales", "sum"),
    total_profit=("Profit", "sum"),
    num_orders=("Order ID", "nunique"),
).reset_index()

def region_to_text(row):
    margin = (row["total_profit"] / row["total_sales"] * 100) if row["total_sales"] else 0
    return (
        f"Region '{row['Region']}': total sales ${row['total_sales']:,.2f}, "
        f"profit ${row['total_profit']:,.2f} (margin {margin:.1f}%), "
        f"{row['num_orders']} unique orders."
    )

region_chunks = region_perf.apply(region_to_text, axis=1).tolist()
region_tags = region_perf.apply(lambda r: {"type": "regions", "region": r["Region"]}, axis=1).tolist()

# yearly
yearly = df.groupby("Year").agg(
    total_sales=("Sales", "sum"),
    total_profit=("Profit", "sum"),
    num_orders=("Order ID", "nunique"),
).reset_index().sort_values("Year")

yearly_text = "Yearly performance:\n"
for _, r in yearly.iterrows():
    margin = r["total_profit"] / r["total_sales"] * 100
    yearly_text += (
        f"  {int(r['Year'])}: sales ${r['total_sales']:,.2f}, "
        f"profit ${r['total_profit']:,.2f} (margin {margin:.1f}%), "
        f"{r['num_orders']} orders.\n"
    )

# region profit rank
region_ranked = region_perf.sort_values("total_profit", ascending=False).reset_index(drop=True)
region_text = "Regions ranked by profit (best to worst):\n"
for rank, (_, r) in enumerate(region_ranked.iterrows(), 1):
    margin = r["total_profit"] / r["total_sales"] * 100
    region_text += (
        f"  #{rank} {r['Region']}: profit ${r['total_profit']:,.2f} "
        f"(margin {margin:.1f}%), sales ${r['total_sales']:,.2f}.\n"
    )

# category profit rank
cat_total = df.groupby("Category").agg(
    total_sales=("Sales", "sum"),
    total_profit=("Profit", "sum"),
).reset_index().sort_values("total_profit", ascending=False)

cat_text = "Product categories ranked by profit (best to worst):\n"
for rank, (_, r) in enumerate(cat_total.iterrows(), 1):
    margin = r["total_profit"] / r["total_sales"] * 100
    cat_text += (
        f"  #{rank} {r['Category']}: profit ${r['total_profit']:,.2f} "
        f"(margin {margin:.1f}%), sales ${r['total_sales']:,.2f}.\n"
    )

ranking_chunks = [yearly_text, region_text, cat_text]
ranking_tags = [
    {"type": "rankings", "topic": "yearly_trend"},
    {"type": "rankings", "topic": "region_ranking"},
    {"type": "rankings", "topic": "category_ranking"},
]

all_chunks = {
    "transactions": {"chunks": transaction_chunks, "tags": transaction_tags},
    "monthly": {"chunks": monthly_chunks, "tags": monthly_tags},
    "categories": {"chunks": category_chunks, "tags": category_tags},
    "regions": {"chunks": region_chunks, "tags": region_tags},
    "rankings": {"chunks": ranking_chunks, "tags": ranking_tags},
}

with open("chunks.json", "w") as f:
    json.dump(all_chunks, f)

print("Total chunks saved:")
for k, v in all_chunks.items():
    print(f"{k}: {len(v['chunks'])}")
