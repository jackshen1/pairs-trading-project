# Pairs Trading Strategy (Statistical Arbitrage)

This project implements a cointegration-based statistical arbitrage strategy to identify and exploit mean-reverting relationships between pairs of stocks.

It includes both a **static regression strategy** (using a fixed relationship from a training period) and a **dynamic rolling regression strategy** (recalculating over a moving window). These are applied across multiple stock markets — including Indian, UK, and US equities — to study generalizability and robustness.

### 📈 What It Does

- Uses **linear regression** on a training set to model pair relationships.
- Tests for **cointegration** using the Engle-Granger test.
- Constructs a **spread** and calculates its **z-score** to trigger trades.
- Implements a **backtest** with realistic trade execution (including transaction costs).
- Applies a **maximum drawdown stop condition** to halt trading if performance degrades.
- Calculates and reports:
  - **Cumulative return**
  - **Sharpe ratio**
  - **Daily volatility**
  - **Max drawdown**

### 🧪 Analysis and Insights

- **Static model** worked well for a period on certain Indian stock pairs (e.g., HDFCBANK and ICICIBANK), but eventually broke down — visualized through spread divergence.
- **Dynamic rolling regression** consistently failed to generalize across most stock pairs tested, providing a useful caution against overfitting.
- **Sharpe Ratio Example**:  
  On one static pair:  
  - Mean Daily Return: `0.000806`  
  - Volatility: `0.018322`  
  - Sharpe Ratio: `0.70`

### ⚙️ Technical Highlights

- Backtesting built in **Python** using `pandas`, `numpy`, `matplotlib`, `statsmodels`, and `yfinance`.
- Strategy logic is modularized into `strategy.py`.
- Batch testing of pairs and performance logging handled via `run_pairs.ipynb`.
- Deeper case study and diagnostics in `static_model.ipynb`.

### 📁 Outputs

- A `.csv` file logging performance of all tested stock pairs.
- Cumulative return plots saved for every pair.
- Exploratory visuals: price history, regression fit, spread evolution, and drawdown effects.

---
