import pandas as pd
import numpy as np
import statsmodels.api as sm

def run_static_strategy(pick1, pick2, data, split_date="2020-03-01", cost=0.0005):
    # generate data
    train_data = data[data.index < split_date]
    test_data = data[data.index >= split_date]

    X = sm.add_constant(train_data[pick1])
    Y = train_data[pick2]
    model = sm.OLS(Y, X).fit()
    intercept, slope = model.params

    # spread + z-score
    spread_train = train_data[pick2] - (intercept + slope * train_data[pick1])
    spread_mean = spread_train.mean()
    spread_std = spread_train.std()

    spread_test = test_data[pick2] - (intercept + slope * test_data[pick1])
    zscore_test = (spread_test - spread_mean) / spread_std
    zscore_test = zscore_test.dropna()

    returns1 = test_data[pick1].pct_change().fillna(0).loc[zscore_test.index]
    returns2 = test_data[pick2].pct_change().fillna(0).loc[zscore_test.index]

    position = 0
    entry_day = -10000
    positions = []
    pnl = []
    prev_position = 0

    z_vals = zscore_test.values
    r1_vals = returns1.values
    r2_vals = returns2.values
    index = zscore_test.index

    # trading logic
    for i in range(len(z_vals)):
        z = z_vals[i]
        if position == 0:
            if z < -1:
                position = 1
                entry_day = i
            elif z > 1:
                position = -1
                entry_day = i
        elif position > 0 and z > -0.5 and i - entry_day > 2:
            position = 0
        elif position < 0 and z < 0.5 and i - entry_day > 2:
            position = 0

        positions.append(position)

        if i == 0:
            pnl.append(0)
        else:
            daily_return = r2_vals[i] - slope * r1_vals[i]
            cost_applied = cost * abs(position - prev_position) if position != prev_position else 0
            pnl.append(position * daily_return - cost_applied)

        prev_position = position

    results = pd.DataFrame({
        "Z-Score": zscore_test,
        "Position": positions,
        "Strategy Return": pnl
    }, index=index)

    results["Cumulative Return"] = (1 + results["Strategy Return"]).cumprod()

    return results


def run_dynamic_strategy(pick1, pick2, data, split_date="2020-03-01", regression_window=120, zscore_window=30, cost=0.0005):
    spread = []
    slopes = []
    intercepts = []
    dates = []

    # generating regression at each point
    for i in range(regression_window, len(data)):
        x_window = data[pick1].iloc[i - regression_window:i]
        y_window = data[pick2].iloc[i - regression_window:i]

        slope, intercept = np.polyfit(x_window, y_window, 1)
        spread_val = data[pick2].iloc[i] - (intercept + slope * data[pick1].iloc[i])

        spread.append(spread_val)
        slopes.append(slope)
        intercepts.append(intercept)
        dates.append(data.index[i])

    spread_series = pd.Series(spread, index=dates)

    zscore = (spread_series - spread_series.rolling(zscore_window).mean()) / \
             spread_series.rolling(zscore_window).std()
    zscore = zscore.dropna()

    aligned_data = data.loc[zscore.index]
    returns1 = aligned_data[pick1].pct_change().fillna(0)
    returns2 = aligned_data[pick2].pct_change().fillna(0)
    slopes_series = pd.Series(slopes, index=dates).loc[zscore.index]

    common_index = zscore.index.intersection(spread_series.index)\
                                 .intersection(returns1.index)\
                                 .intersection(returns2.index)\
                                 .intersection(slopes_series.index)

    zscore = zscore.loc[common_index]
    spread_series = spread_series.loc[common_index]
    returns1 = returns1.loc[common_index]
    returns2 = returns2.loc[common_index]
    slopes_series = slopes_series.loc[common_index]

    test_mask = zscore.index >= split_date
    zscore_test = zscore[test_mask]
    spread_test = spread_series[test_mask]
    returns1 = returns1[test_mask]
    returns2 = returns2[test_mask]
    slopes_series = slopes_series[test_mask]

    position = 0
    entry_day = -10000
    positions = []
    pnl = []
    prev_position = 0

    z_vals = zscore_test.values
    r1_vals = returns1.values
    r2_vals = returns2.values
    slopes_vals = slopes_series.values
    index = zscore_test.index

    # trading logic
    for i in range(len(z_vals)):
        z = z_vals[i]
        slope_now = slopes_vals[i]

        if position == 0:
            if z < -1:
                position = 1
                entry_day = i
            elif z > 1:
                position = -1
                entry_day = i
        elif position > 0 and z > -0.5 and i - entry_day > 2:
            position = 0
        elif position < 0 and z < 0.5 and i - entry_day > 2:
            position = 0

        positions.append(position)

        if i == 0:
            pnl.append(0)
        else:
            daily_return = r2_vals[i] - slope_now * r1_vals[i]
            cost_applied = cost * abs(position - prev_position) if position != prev_position else 0
            pnl.append(position * daily_return - cost_applied)

        prev_position = position

    results = pd.DataFrame({
        "Z-Score": zscore_test,
        "Position": positions,
        "Strategy Return": pnl
    }, index=index)

    results["Cumulative Return"] = (1 + results["Strategy Return"]).cumprod()

    return results

