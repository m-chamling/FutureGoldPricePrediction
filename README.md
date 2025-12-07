# Gold Price Prediction

This project predicts the **next-day closing price of Gold Futures (GC=F)** using historical data from Yahoo Finance and a Random Forest model.

## Overview
- Data source: Yahoo Finance  
- Features: Closing prices from the last 3 days  
- Model: Random Forest Regressor  
- Output: Next-day price prediction  

## Results
- RMSE ≈ 865  
- MAE ≈ 674  
- Example prediction: **2356.40 USD**

## How to Run
```bash
pip install yfinance pandas numpy scikit-learn matplotlib
jupyter notebook
