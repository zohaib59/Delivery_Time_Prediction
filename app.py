
# 📦 Imports
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# 📂 Load Dataset
os.chdir(r"C:\Users\zohaib khan\OneDrive\Desktop\USE ME\dump\zk")
data = pd.read_csv("delivery.csv")
pd.set_option('display.max_columns', None)

# 🧼 Label Encoding
for col in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

# 🎯 Feature-Target Split
X = data.drop("delivery_time_minutes", axis=1).astype(np.float32)
y = data["delivery_time_minutes"].astype(np.float32)

# 🔀 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔁 Scaling
scaler = StandardScaler()
X_train_scaled_df = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test_scaled_df = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
full_scaled_df = pd.DataFrame(scaler.transform(X), columns=X.columns)

# 📏 MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100

# 🧮 Enhanced Evaluation
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    def show_metrics(y_true, y_pred, label):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        std_dev = y_true.std()
        print(f"\n📊 {model_name} [{label} Set]")
        print(f"R²                     : {r2:.4f}")
        print(f"RMSE                   : {rmse:.2f}")
        print(f"MAE                    : {mae:.2f}")
        print(f"MAPE                   : {mape:.2f}%")
        print(f"Target Std Dev         : {std_dev:.2f}")
        print(f"RMSE / StdDev Ratio    : {rmse / std_dev:.2f}")
        print("✅ Good Fit" if rmse < std_dev else "⚠️ RMSE exceeds variability — may need tuning")

    show_metrics(y_train, y_train_pred, "Train")
    show_metrics(y_test, y_test_pred, "Test")

    residuals = y_test - y_test_pred
    sns.histplot(residuals, bins=50, kde=True)
    plt.title(f"Residuals: {model_name} [Test]")
    plt.grid()
    plt.xlabel("Residual")
    plt.show()

    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=y_test_pred, y=residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f"Residuals vs Predictions: {model_name} [Test]")
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.grid()
    plt.show()

# ⚡ Final Model List
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(),
    "XGBoost": XGBRegressor(n_jobs=-1),
    "LightGBM": LGBMRegressor(n_jobs=-1)
}

# 🧪 CV & SHAP with Clean Logging
results = []
kf = KFold(n_splits=3, shuffle=True, random_state=42)

for name, model in models.items():
    try:
        pipe = Pipeline([('regressor', model)])
        
        # Train & evaluate
        evaluate_model(pipe.named_steps['regressor'], X_train_scaled_df, y_train, X_test_scaled_df, y_test, name)
        
        # Cross-validation score
        cv_score = cross_val_score(pipe, X_train_scaled_df, y_train, cv=kf, scoring='r2').mean()
        results.append((name, cv_score))

        # SHAP explanation for baseline model
        if name == "Linear Regression":
            data["Predicted_Processing_Time"] = pipe.predict(full_scaled_df)
            data["Residual"] = data["processing_time_days"] - data["Predicted_Processing_Time"]
            
            explainer = shap.Explainer(pipe.named_steps['regressor'], X_train_scaled_df)
            shap_values = explainer(X_test_scaled_df.iloc[:100])
            shap.plots.beeswarm(shap_values, max_display=10)

    except Exception:
        print(f"{name} failed during evaluation or SHAP rendering.")

# 💾 Save
data.to_excel("model_predictions.xlsx", index=False)
results_df.to_excel("model_scores.xlsx", index=False)
print("\n✅ Exported: model_predictions.xlsx & model_scores.xlsx")

# 🔍 Hyperparameter Tuning
param_grids = {
    'XGBRegressor': {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
    },
    'LGBMRegressor': {
        'n_estimators': [100, 200],
        'max_depth': [5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [31, 50]
    },
    'GradientBoostingRegressor': {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
    }
}

models_to_tune = {
    'XGBRegressor': XGBRegressor(objective='reg:squarederror', n_jobs=-1),
    'LGBMRegressor': LGBMRegressor(objective='regression'),
    'GradientBoostingRegressor': GradientBoostingRegressor()
}

best_models = {}
for name, model in models_to_tune.items():
    print(f"\n🔍 Tuning {name}...")
    grid = GridSearchCV(estimator=model,
                        param_grid=param_grids[name],
                        cv=3,
                        scoring='r2',
                        verbose=1,
                        n_jobs=-1)
    grid.fit(X_train_scaled_df, y_train)
    best_models[name] = grid.best_estimator_
    print(f"✅ Best Params for {name}: {grid.best_params_}")
    print(f"📈 Best CV R² Score: {grid.best_score_:.4f}")

# 🔎 Evaluate Tuned Models
for name, model in best_models.items():
    evaluate_model(model, X_train_scaled_df, y_train, X_test_scaled_df, y_test, f"{name} [Tuned]")

