import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

# ==========================

# PAGE CONFIG

# ==========================

st.set_page_config(
page_title="Food Waste Reduction Dashboard",
page_icon="🍽️",
layout="wide"
)

st.title("🍽️ Restaurant & Supermarket Production Recommendation System")

# ==========================

# LOAD DATA

# ==========================

df = pd.read_csv("restaurant_supermarket_food_waste_dataset_5000.csv")

st.sidebar.header("Dataset Overview")

st.sidebar.write("Rows:", df.shape[0])
st.sidebar.write("Columns:", df.shape[1])

# ==========================

# DATA PREPROCESSING

# ==========================

le_product = LabelEncoder()
le_category = LabelEncoder()

df["product_name"] = le_product.fit_transform(df["product_name"])
df["category"] = le_category.fit_transform(df["category"])

# ==========================

# MODEL

# ==========================

X = df.drop(
columns=[
"date",
"qty_sold",
"waste_qty"
]
)

y = df["qty_sold"]

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42
)

model = XGBRegressor(
n_estimators=200,
max_depth=5,
learning_rate=0.1,
random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# ==========================

# KPI SECTION

# ==========================

st.header("📊 Key Metrics")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
"Total Products Sold",
int(df["qty_sold"].sum())
)

c2.metric(
"Total Waste",
int(df["waste_qty"].sum())
)

c3.metric(
"Model MAE",
round(mae, 2)
)

c4.metric(
"R² Score",
round(r2, 2)
)

# ==========================

# SALES ANALYSIS

# ==========================

st.header("📈 Product Sales")

sales = (
df.groupby("product_name")["qty_sold"]
.sum()
.sort_values(ascending=False)
)

st.bar_chart(sales)

# ==========================

# WASTE ANALYSIS

# ==========================

st.header("🗑️ Product Waste")

waste = (
df.groupby("product_name")["waste_qty"]
.sum()
.sort_values(ascending=False)
)

st.bar_chart(waste)

# ==========================

# FEATURE IMPORTANCE

# ==========================

st.header("🎯 Feature Importance")

importance = pd.DataFrame({
"Feature": X.columns,
"Importance": model.feature_importances_
})

importance = importance.sort_values(
by="Importance",
ascending=False
)

st.dataframe(importance)

st.bar_chart(
importance.set_index("Feature")
)

# ==========================

# PRODUCTION RECOMMENDATION

# ==========================

st.header("🏭 Production Recommendation")

predicted_demand = st.number_input(
"Predicted Demand",
value=100
)

current_inventory = st.number_input(
"Current Inventory",
value=20
)

safety_stock = st.number_input(
"Safety Stock",
value=10
)

recommendation = (
predicted_demand
+ safety_stock
- current_inventory
)

st.success(
f"Recommended Production Quantity: {max(0, recommendation)}"
)

# ==========================

# DATA PREVIEW

# ==========================

st.header("📄 Dataset Preview")

st.dataframe(df.head(20))
