import streamlit as st
import pandas as pd
import joblib
import os
from pathlib import Path

# Get the directory of the current script and construct paths
app_dir = Path(__file__).parent
project_root = app_dir.parent
model_path = project_root / "models" / "house_price_model.pkl"
preprocessor_path = project_root / "models" / "preprocessor.pkl"

# Load model and preprocessor
model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)

# Load cleaned data to get column defaults
data_path = project_root / "data" / "preprocessed" / "ames_house_prices_cleaned.csv"
df_template = pd.read_csv(data_path, keep_default_na=False, na_values=[''])
df_template = df_template.drop(columns=["Id", "Segment", "SalePrice"], errors='ignore')

st.title("🏠 House Price Prediction App")

st.write("Enter house details below:")

# -------------------------
# User Inputs
# -------------------------
OverallQual = st.slider("Overall Quality", 1, 10, 5)
GrLivArea = st.number_input("Living Area (sq ft)", 500, 10000, 1500)
TotalBsmtSF = st.number_input("Basement Area", 0, 5000, 800)
GarageCars = st.slider("Garage Cars", 0, 4, 1)
GarageArea = st.number_input("Garage Area", 0, 2000, 400)
LotArea = st.number_input("Lot Area", 1000, 50000, 8000)
YearBuilt = st.slider("Year Built", 1872, 2010, 2000)

# Example categorical (simplified)
Neighborhood = st.selectbox("Neighborhood", ["NAmes", "CollgCr", "OldTown", "Edwards"])

# -------------------------
# Predict Button
# -------------------------
if st.button("Predict Price"):
    
    # Create a complete input dataframe using template defaults
    input_data = df_template.iloc[[0]].copy()
    
    # Set ordinal features to 'TA' (3) as safe default - most common quality rating
    ordinal_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'KitchenQual',
                    'FireplaceQu', 'GarageQual', 'GarageCond', 'HeatingQC']
    for col in ordinal_cols:
        if col in input_data.columns:
            input_data[col] = 'TA'  # Average/Typical
    
    if 'BsmtExposure' in input_data.columns:
        input_data['BsmtExposure'] = 'Av'  # Average exposure
    
    # Update with user inputs
    input_data["OverallQual"] = OverallQual
    input_data["GrLivArea"] = GrLivArea
    input_data["TotalBsmtSF"] = TotalBsmtSF
    input_data["GarageCars"] = GarageCars
    input_data["GarageArea"] = GarageArea
    input_data["LotArea"] = LotArea
    input_data["YearBuilt"] = YearBuilt
    input_data["Neighborhood"] = Neighborhood
    
    # Calculate engineered features (same as in feature engineering notebook)
    input_data["TotalBathrooms"] = (input_data["FullBath"] + 
                                     0.5 * input_data["HalfBath"] + 
                                     input_data["BsmtFullBath"] + 
                                     0.5 * input_data["BsmtHalfBath"])
    
    input_data["HouseAge"] = input_data["YrSold"] - input_data["YearBuilt"]
    input_data["YearsSinceRemodel"] = input_data["YrSold"] - input_data["YearRemodAdd"]
    input_data["GarageAge"] = input_data["YrSold"] - input_data["GarageYrBlt"]
    input_data["TotalLivingArea"] = (input_data["TotalBsmtSF"] + 
                                      input_data["1stFlrSF"] + 
                                      input_data["2ndFlrSF"])

    # Preprocess (the preprocessor has OrdinalEncoder that will handle string-to-number conversion)
    input_processed = preprocessor.transform(input_data)

    # Predict
    prediction = model.predict(input_processed)

    st.success(f"🏠 Predicted House Price: ${int(prediction[0]):,}")