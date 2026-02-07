# ======================================================
# TRAINING MODEL CUSTOMER CHURN PREDICTION
# ======================================================

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler


# ======================================================
# 1. LOAD DATASET
# ======================================================

print("✅ Memuat dataset...")

df = pd.read_csv("data/churn.csv")

print("✅ Dataset berhasil dimuat!")


# ======================================================
# 2. DATA PREPARATION
# ======================================================

# Mapping target variable
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
df["Churn"] = df["Churn"].astype(int)

# Menghapus kolom ID
df.drop(columns=["customerID"], inplace=True)

print("✅ Target variable berhasil dipetakan!")


# ======================================================
# 3. PEMBAGIAN DATASET
# ======================================================

X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("✅ Dataset berhasil dibagi menjadi data train dan test!")


# ======================================================
# 4. HANDLING MISSING VALUE
# (menggunakan kolom sebelum feature selection)
# ======================================================

num_cols_impute = X_train.select_dtypes(include=["int64", "float64"]).columns
cat_cols_impute = X_train.select_dtypes(include="object").columns

# Imputasi numerik (median dari train)
X_train[num_cols_impute] = X_train[num_cols_impute].fillna(
    X_train[num_cols_impute].median()
)
X_test[num_cols_impute] = X_test[num_cols_impute].fillna(
    X_train[num_cols_impute].median()
)

# Imputasi kategorikal (mode dari train)
X_train[cat_cols_impute] = X_train[cat_cols_impute].fillna(
    X_train[cat_cols_impute].mode().iloc[0]
)
X_test[cat_cols_impute] = X_test[cat_cols_impute].fillna(
    X_train[cat_cols_impute].mode().iloc[0]
)

print("✅ Missing value berhasil ditangani!")


# ======================================================
# 5. FEATURE SELECTION
# ======================================================

# Menghapus fitur dengan multicollinearity tinggi
if "TotalCharges" in X_train.columns:
    X_train.drop(columns=["TotalCharges"], inplace=True)
    X_test.drop(columns=["TotalCharges"], inplace=True)

print("✅ Feature selection selesai!")


# ======================================================
# 6. IDENTIFIKASI FITUR NUMERIK & KATEGORIKAL
# (setelah feature selection)
# ======================================================

num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X_train.select_dtypes(include="object").columns


# ======================================================
# 7. FEATURE SCALING
# ======================================================

scaler = StandardScaler()

X_train_num = scaler.fit_transform(X_train[num_cols])
X_test_num = scaler.transform(X_test[num_cols])

print("✅ Scaling fitur numerik berhasil!")


# ======================================================
# 8. FEATURE ENCODING
# ======================================================

encoder = OneHotEncoder(
    drop="first",
    sparse_output=False,
    handle_unknown="ignore"
)

X_train_cat = encoder.fit_transform(X_train[cat_cols])
X_test_cat = encoder.transform(X_test[cat_cols])

print("✅ Encoding fitur kategorikal berhasil!")


# ======================================================
# 9. FEATURE COMBINATION
# ======================================================

X_train_final = np.hstack([X_train_num, X_train_cat])
X_test_final = np.hstack([X_test_num, X_test_cat])

print("✅ Fitur numerik dan kategorikal berhasil digabungkan!")


# ======================================================
# 10. HANDLING IMBALANCED DATA
# ======================================================

ros = RandomOverSampler(random_state=42)

X_train_resampled, y_train_resampled = ros.fit_resample(
    X_train_final,
    y_train
)

print("✅ Class imbalance berhasil ditangani!")


# ======================================================
# 11. MODEL TRAINING
# ======================================================

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_resampled, y_train_resampled)

print("✅ Model Logistic Regression berhasil dilatih!")


# ======================================================
# 12. SAVE MODEL & PREPROCESSING OBJECT
# ======================================================

os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/churn_model.joblib")
joblib.dump(scaler, "model/scaler.joblib")
joblib.dump(encoder, "model/encoder.joblib")

print("✅ Model dan preprocessing object berhasil disimpan!")
print("✅ Proses training model selesai.")
