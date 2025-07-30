# Delivery Time Prediction – ML Regression Benchmark
Just wrapped up a real-world ML pipeline for **forecasting delivery time (minutes)** using a dataset packed with logistics intelligence: traffic scores, 
fuel consumption, route IDs, driver ratings, hub locations, and more.
This project benchmarks multiple regression models on a real-world logistics dataset to predict delivery time in minutes (`delivery_time_minutes`). 
Built with explainability, diagnostic visuals, and hyperparameter tuning, it's ideal for production-scale forecasting in delivery and fleet scenarios.

🚚 Predicting Delivery Time with ML – Regression Benchmarking Across 8 Models 📦

📊 Benchmarked Models:
- Linear, Ridge, Lasso
- Tree-based: RF, GB, XGBoost, LightGBM

🧪 Evaluation Metrics:
- RMSE, MAE, R², MAPE
- Residual analysis and StdDev ratios
- SHAP integrated for transparency

This pipeline is plug-and-play for delivery ops, fleet modeling, or API-driven scheduling systems.

#MachineLearning #Regression #LogisticsAI #FleetOps #XGBoost #LightGBM #PredictiveModeling #MLOps #DataScienceForBusiness

## ⚙️ Key Features

- Label encoding for categorical logistics variables
- Feature scaling via `StandardScaler`
- Enhanced evaluation: RMSE, R², MAE, MAPE, StdDev ratios
- Residual visualization (distribution & scatter)
- Hyperparameter tuning for XGBoost, LightGBM, and Gradient Boosting
- SHAP explainability integration (for baseline models)
