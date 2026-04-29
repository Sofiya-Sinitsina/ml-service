import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

import numpy as np

np.random.seed(28)


df = pd.read_csv(r"D:\Downloads\synthetic_crop_yield_437.csv")

y = df["yield_tons_per_hectare"]
X = df.drop(columns=["yield_tons_per_hectare"])

numeric_features = X.select_dtypes(include=["float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=100,
    random_state=7))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pipeline.fit(X_train, y_train)

preds = pipeline.predict(X_test)

print("MAE:", mean_absolute_error(y_test, preds))

joblib.dump(pipeline, "modelRF4.pkl")