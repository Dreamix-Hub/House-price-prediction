import pandas as pd
from sklearn.model_selection import train_test_split

# Load your feature engineered dataset
df = pd.read_csv("../data/final/housing_feature_engineered.csv")

# 1. Define target variable (y)
y = df["SalePrice"]

# 2. Define features (X)
X = df.drop("SalePrice", axis=1)

# 3. Train-Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
