# Optiver Market Volatility Prediction

## Project Background
This project originates from a financial data science coursework inspired by a real-world market-making problem at Optiver.

The objective is to predict short-term market volatility using high-frequency order book and trade data, with an emphasis on interpretability and stability for trading applications.

Key constraints of the project include:
- Processing ultra-high-frequency financial data
- Selecting appropriate time-series modelling frameworks
- Evaluating models using multiple performance metrics
- Prioritising interpretability and stability over purely maximising accuracy

## Project Context
This repository is based on a university coursework project originally hosted in a private school GitHub account that is not publicly accessible.

To make my work reviewable by recruiters, I migrated the relevant parts of the project
to this public repository. The project was completed as part of a group assignment.

All restructuring, documentation, and analysis in this repository reflect
my personal contribution and understanding of the work.

## Problem Statement
Accurate volatility prediction plays an important role in risk management,
pricing, and trading strategy design.

This project focuses on building interpretable and robust volatility prediction models
using high-frequency financial data, rather than purely optimising raw predictive accuracy.

## Data
- High-frequency order book and trade data
- Data cleaning and preprocessing to handle noise and outliers
- Feature engineering including:
  - Log returns
  - Realised volatility
  - Rolling window statistics

## Methodology
The analysis follows a standard data science workflow:
1. Exploratory Data Analysis (EDA) to understand distributional properties
2. Feature engineering based on financial time-series characteristics
3. Model development and comparison, including:
   - Baseline statistical models (e.g. ARIMA)
   - Feature-based regression models
4. Model evaluation using appropriate time-series validation strategies

## Results
- Compared multiple models on predictive accuracy and stability
- Baseline time-series models provided strong interpretability
- Feature-based models showed improved performance in volatile periods
- Results highlight trade-offs between responsiveness and overfitting

## Key Takeaways
- High-frequency financial data requires careful preprocessing
- Model interpretability and stability are critical in financial applications
- Time-series cross-validation is essential to avoid data leakage

## Modelling Decisions and Limitations
- Model choices and evaluation metrics were guided by the project’s emphasis on interpretability and robustness for trading applications.
- Some design decisions, such as the choice of SMAPE and extensive feature engineering, involved trade-offs between interpretability, performance, and computational cost.
- Potential limitations include sensitivity to feature construction and challenges in validating models on ultra-high-frequency time-series data.
These considerations informed the final model selection and provided directions for future improvement.

## Tools & Technologies
Python, Pandas, NumPy, Statsmodels, Jupyter Notebook

## My Contribution
This project was completed as part of a group coursework.
My primary contributions focused on time-series modelling and financial interpretation, including:
- Designing and implementing an ARIMA-based volatility prediction model
- Conducting stationarity analysis and parameter selection using ACF/PACF and statistical tests
- Implementing rolling-window forecasting to adapt to evolving market dynamics
- Computing realised volatility from order book data using weighted average price (WAP)
- Contributing to the project introduction, literature review, and financial context of volatility modelling
