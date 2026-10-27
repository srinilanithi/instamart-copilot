import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import joblib

os.makedirs("outputs", exist_ok=True)

print("Loading cleaned data...")
cust_df = pd.read_csv("data/clean/customers.csv")
orders_df = pd.read_csv("data/clean/orders.csv")

orders_df['order_ts'] = pd.to_datetime(orders_df['order_ts'])
anchor_date = orders_df['order_ts'].max()  # End of dataset window

# Filter completed orders for RFM calculation
completed_orders = orders_df[orders_df['status'] == 'completed']

# 1. Feature Engineering (RFM + Tenure)
rfm = completed_orders.groupby('customer_id').agg(
    recency=('order_ts', lambda x: (anchor_date - x.max()).days),
    frequency=('order_id', 'count'),
    monetary=('order_amount', 'sum'),
    avg_order_value=('order_amount', 'mean')
).reset_index()

cust_df = cust_df.merge(rfm, on='customer_id', how='left').fillna({
    'recency': (anchor_date - pd.to_datetime(cust_df['signup_date'])).dt.days,
    'frequency': 0,
    'monetary': 0.0,
    'avg_order_value': 0.0
})

cust_df['tenure'] = (anchor_date - pd.to_datetime(cust_df['signup_date'])).dt.days

# 2. Customer Segmentation (K-Means Clustering)
feature_cols = ['recency', 'frequency', 'monetary', 'tenure']
X_scaled = StandardScaler().fit_transform(cust_df[feature_cols])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cust_df['segment'] = kmeans.fit_predict(X_scaled)
cust_df.to_csv("outputs/customer_segments.csv", index=False)
print("Saved customer segments to outputs/customer_segments.csv")

# 3. Lapse (Churn) Prediction Model
# Define target: Lapsed = 1 if no order in the last 60 days, else 0
cust_df['is_lapsed'] = (cust_df['recency'] > 60).astype(int)

X = cust_df[feature_cols]
y = cust_df['is_lapsed']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save trained model
joblib.dump(model, "outputs/lapse_model.joblib")
print("Saved lapse model to outputs/lapse_model.joblib")

# 4. Generate Retention List
cust_df['lapse_probability'] = model.predict_proba(X)[:, 1]
retention_list = cust_df[cust_df['lapse_probability'] > 0.5][['customer_id', 'city', 'recency', 'frequency', 'lapse_probability']]
retention_list.to_csv("outputs/retention_list.csv", index=False)
print(f"Generated retention list with {len(retention_list)} customers at outputs/retention_list.csv")