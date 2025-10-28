# **California Housing Price Prediction**
Capstone Project Module 3 — Machine Learning Regression

## Project Introduction
This project builds a supervised regression model to predict median house values across districts in California using demographic and geographical data from the 1990 U.S. Census.  
California’s housing market fluctuates due to differences in income, location, and population density.  
A data-driven predictive model enables:
- Real estate firms to estimate optimal listing prices.  
- Banks and lenders to assess property collateral values.  
- Policy makers to evaluate housing affordability and inequality.

## Objectives
- Develop a regression model that predicts `median_house_value` accurately.  
- Identify socio-economic and spatial factors driving housing price variations.  
- Provide an objective, data-based valuation framework.

## Dataset Overview
The dataset used is the California Housing Dataset from the 1990 U.S. Census, consisting of over 14,000 observations.  
Each row represents a housing block group, the smallest geographical unit in the census.

| Column | Description |
|--------|-------------|
| `longitude`, `latitude` | District coordinates |
| `housing_median_age` | Median age of houses |
| `total_rooms`, `total_bedrooms` | Total rooms and bedrooms |
| `population`, `households` | Demographic data |
| `median_income` | Median income (in tens of thousands USD) |
| `ocean_proximity` | Distance from the ocean (categorical) |
| `median_house_value` | Target variable — median home value (USD) |

Data quality summary:
- Missing values handled via median imputation (`total_bedrooms`).  
- Top-coded `median_house_value` at 500,001 USD (~4.7% of data).  
- Created three ratio features: `rooms_per_household`, `bedrooms_per_room`, and `population_per_household`.

## Modeling Workflow
The modeling process follows the full machine learning pipeline.

1. **Data Cleaning and Feature Engineering**  
   - Missing values imputed with median for robustness.  
   - Ratio features engineered to capture density and size relationships.  
   - Numerical features standardized via `StandardScaler`.  
   - Categorical features encoded using `OneHotEncoder`.  
   - Combined pipeline constructed using `ColumnTransformer` for reproducibility.

2. **Model Development**  
   Models evaluated:
   - Linear Reg
   - Ridge and Lasso Regression (regularized)  
   - Random Forest Regressor  
   - XGBoost Regressor (final model)

3. **Model Evaluation**  
   Metrics used:
   - RMSE (Root Mean Squared Error)  
   - MAE (Mean Absolute Error)  
   - R² (Coefficient of Determination)

   Results summary:

   | Model | RMSE | R² | Notes |
   |--------|------|----|-------|
   | Linear Regression | ~68,000 | 0.63 | Baseline |
   | Ridge / Lasso | ~60,000 | 0.70 | Improved regularization |
   | Random Forest | ~47,000 | 0.80 | Non-linear capture |
   | XGBoost | 45,900 | 0.84 | Best overall performance |

   The XGBoost model achieved the best balance of low RMSE and high R², explaining about 84% of housing price variability.

4. **Model Saving**  
   The trained pipeline (preprocessor + model) was saved for deployment:
   ```python
   from joblib import dump
   dump(best_model, "best_model.pkl")
