# Trading ML Strategy

This repository contains an implementation of a trading strategy powered by machine learning models. The strategy is backtested using historical stock data, leveraging Python packages like Backtrader, Pandas, and Scikit-learn. Detailed performance reports and analyses are generated to compare model effectiveness on different stocks.

---

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Key Features](#key-features)
- [Results](#results)

---

## Overview

The project implements a backtesting framework and machine learning-based trading strategy. The repository supports:

1. Downloading stock market data.
2. Generating features for machine learning models.
3. Comparing the performance of ML models in financial trading.
4. Backtesting trading strategies and producing detailed reports.


# Trading Machine Learning Reports

Access the reports for the Logistic Regression and MLP Classifier models below. Click on the respective links to view the detailed analysis.

| **Logistic Regression Reports**                                    | **MLP Classifier Reports**                                    |
|--------------------------------------------------------------------|----------------------------------------------------------------|
| [FTV_LogisticRegression](https://dyllon-preston.github.io/trading-ML/FTV_LogisticRegression.html) | [FTV_MLPClassifier](https://dyllon-preston.github.io/trading-ML/FTV_MLPClassifier.html) |
| [CL_LogisticRegression](https://dyllon-preston.github.io/trading-ML/CL_LogisticRegression.html) | [CL_MLPClassifier](https://dyllon-preston.github.io/trading-ML/CL_MLPClassifier.html) |
| [FE_LogisticRegression](https://dyllon-preston.github.io/trading-ML/FE_LogisticRegression.html) | [FE_MLPClassifier](https://dyllon-preston.github.io/trading-ML/FE_MLPClassifier.html) |
| [HST_LogisticRegression](https://dyllon-preston.github.io/trading-ML/HST_LogisticRegression.html) | [HST_MLPClassifier](https://dyllon-preston.github.io/trading-ML/HST_MLPClassifier.html) |
| [IQV_LogisticRegression](https://dyllon-preston.github.io/trading-ML/IQV_LogisticRegression.html) | [IQV_MLPClassifier](https://dyllon-preston.github.io/trading-ML/IQV_MLPClassifier.html) |
| [MOS_LogisticRegression](https://dyllon-preston.github.io/trading-ML/MOS_LogisticRegression.html) | [MOS_MLPClassifier](https://dyllon-preston.github.io/trading-ML/MOS_MLPClassifier.html) |
| [MU_LogisticRegression](https://dyllon-preston.github.io/trading-ML/MU_LogisticRegression.html) | [MU_MLPClassifier](https://dyllon-preston.github.io/trading-ML/MU_MLPClassifier.html) |
| [PPG_LogisticRegression](https://dyllon-preston.github.io/trading-ML/PPG_LogisticRegression.html) | [PPG_MLPClassifier](https://dyllon-preston.github.io/trading-ML/PPG_MLPClassifier.html) |
| [TMUS_LogisticRegression](https://dyllon-preston.github.io/trading-ML/TMUS_LogisticRegression.html) | [TMUS_MLPClassifier](https://dyllon-preston.github.io/trading-ML/TMUS_MLPClassifier.html) |
| [UPS_LogisticRegression](https://dyllon-preston.github.io/trading-ML/UPS_LogisticRegression.html) | [UPS_MLPClassifier](https://dyllon-preston.github.io/trading-ML/UPS_MLPClassifier.html) |

---

## Requirements

The project was developed using **Python 3.7.x**. Below are the versions of the key packages used:

| Package          | Version    |
|------------------|------------|
| Python           | 3.7.0      |
| Numpy            | 1.21.5     |
| Pandas           | 1.3.5      |
| Scikit-learn     | 1.0.2      |
| Backtrader       | 1.9.76.123 |
| Pyfolio          | 0.9.2      |
| YFinance         | 0.2.28     |
| quantstats       | 0.0.62     |


---

## Installation

Follow the steps below to set up the project environment:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Dyllon-Preston/trading-ML
   cd trading-ML


2. **Set up a python Enviroment Using Conda**

```
conda create -n trading_ml_env python=3.7
conda activate trading_ml_env
```

3. **Install Dependencies**

## Project Structure

The repository is organized as follows:

trading-ml-strategy/
│
├── main.py                  # Runs tests for data importing and generates reports
├── test.py                  # Unit tests for utilities and models
├── utility.py               # Utilities for stock data, backtesting, and feature generation
├── generate_reports.py      # Module to run backtests and generate performance reports
├── model_comparison.ipynb   # Jupyter notebook for comparing ML model performance
├── README.md                # Project documentation (this file)

## Usage

1. Running Backtests and Generating Reports
Use the `main.py` file to run tests, download stock data, run the backtesting strategy, and generate reports.

```
python main.py
```

2. Model Comparison
Use the model_comparison.ipynb notebook to experiment with and compare different machine learning models for financial data.

## Key Features

- Machine Learning-Based Trading: Implements strategies powered by scikit-learn models.
- Flexible Feature Generation: Utilities to compute key financial indicators and prepare model inputs.
- Backtesting Framework: Leverages Backtrader for historical simulation of trading strategies.
- Performance Analysis: Generates detailed reports, including cumulative returns, Sharpe ratios, and drawdowns using quantstats.
- Stock Data Management: Supports downloading and preprocessing data from Yahoo Finance.

## Results

Detailed performance reports are saved in the output directory. Metrics include:

Cumulative Returns: Performance of the strategy over time.
Sharpe Ratio: Risk-adjusted returns.
Drawdowns: Maximum losses from peak to trough.
For more details, refer to the generate_reports.py output.
