import os
import sqlite3
import pandas as pd
import numpy as np

# Ensure directories exist
os.makedirs("data/clean", exist_ok=True)
os.makedirs("sql", exist_ok=True)

print("Loading raw data...")
cust_df = pd.read_csv("data/raw/customers.csv")
prod_df = pd.read_csv("data/raw/products.csv")
orders_df = pd.read_csv("data/raw/orders.csv")

# ---------------------------------------------------------
# 1. CLEAN CUSTOMERS
# ---------------------------------------------------------
print("Cleaning customers...")
cust_df['city'] = cust_df['city'].astype(str).str.strip().str.title()
cust_df['gender'] = cust_df['gender'].astype(str).str.strip().str.lower()
cust_df['gender'] = cust_df['gender'].map({'male': 'male', 'female': 'female'}).fillna('unknown')

cust_df['age'] = pd.to_numeric(cust_df['age'], errors='coerce')
cust_df.loc[(cust_df['age'] < 10) | (cust_df['age'] > 100), 'age'] = np.nan
city_medians = cust_df.groupby('city')['age'].transform('median')
cust_df['age'] = cust_df['age'].fillna(city_medians).round().astype(int)
cust_df['signup_date'] = pd.to_datetime(cust_df['signup_date']).dt.strftime('%Y-%m-%d')
cust_df.to_csv("data/clean/customers.csv", index=False)

# ---------------------------------------------------------
# 2. CLEAN PRODUCTS
# ---------------------------------------------------------
prod_df.to_csv("data/clean/products.csv", index=False)

# ---------------------------------------------------------
# 3. CLEAN ORDERS
# ---------------------------------------------------------
print("Cleaning orders...")
orders_df['order_ts'] = pd.to_datetime(orders_df['order_ts'])
orders_df = orders_df.drop_duplicates()
orders_df = orders_df.sort_values('order_ts').groupby('order_id', as_index=False).last()

orders_df['order_amount_num'] = pd.to_numeric(orders_df['order_amount'], errors='coerce')
orders_merged = orders_df.merge(prod_df, on='product_id', how='left')
calc_amount = (orders_merged['price_inr'] * orders_merged['quantity'] * (1 - orders_merged['discount_pct'] / 100)).round(2)
orders_df['order_amount'] = orders_df['order_amount_num'].fillna(calc_amount)

orders_df['delivery_minutes'] = pd.to_numeric(orders_df['delivery_minutes'], errors='coerce')
orders_df.loc[orders_df['delivery_minutes'] < 0, 'delivery_minutes'] = np.nan

cat_medians = orders_df.merge(prod_df, on='product_id', how='left').groupby('category')['delivery_minutes'].transform('median')
is_completed = orders_df['status'] == 'completed'
orders_df.loc[is_completed & orders_df['delivery_minutes'].isna(), 'delivery_minutes'] = cat_medians.round()

orders_df['order_ts'] = orders_df['order_ts'].dt.strftime('%Y-%m-%d %H:%M:%S')
if 'order_amount_num' in orders_df.columns:
    orders_df = orders_df.drop(columns=['order_amount_num'])
orders_df.to_csv("data/clean/orders.csv", index=False)

# ---------------------------------------------------------
# 4. STORE IN SQLITE DATABASE
# ---------------------------------------------------------
print("Storing clean tables in SQLite database (data/instamart.db)...")
conn = sqlite3.connect("data/instamart.db")
cust_df.to_sql("customers", conn, if_exists="replace", index=False)
prod_df.to_sql("products", conn, if_exists="replace", index=False)
orders_df.to_sql("orders", conn, if_exists="replace", index=False)
conn.commit()
conn.close()

print("Part A data cleaning and SQLite storage complete!")