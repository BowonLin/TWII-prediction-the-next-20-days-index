from datetime import datetime
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from quantile_forest import RandomForestQuantileRegressor
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ==========================================
# 1. API Credentials & Parameters
# ==========================================
FINMIND_TOKEN = ""
URL = ""

START_DATE = "2020-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")


def get_finmind_data(dataset, data_id):
    parameter = {
        "dataset": dataset,
        "data_id": data_id,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "token": FINMIND_TOKEN,
    }
    resp = requests.get(URL, params=parameter)
    data = resp.json()
    return pd.DataFrame(data["data"])


print(f"1. Fetching raw market data ({START_DATE} ~ {END_DATE})...")

# A. TAIEX Index
df_taiex_raw = get_finmind_data("TaiwanStockPrice", "TAIEX")
df_taiex = df_taiex_raw[["date", "close", "min"]].copy()
df_taiex.columns = ["date", "TAIEX_Close", "TAIEX_Low"]
df_taiex["date"] = pd.to_datetime(df_taiex["date"]).dt.strftime("%Y-%m-%d")

# B. TX Futures (Main Contract)
df_tx_raw = get_finmind_data("TaiwanFuturesDaily", "TX")
df_tx_main = (
    df_tx_raw.sort_values("volume", ascending=False)
    .groupby("date")
    .first()
    .reset_index()
)
df_tx = df_tx_main[["date", "close", "min", "volume"]].copy()
df_tx.columns = ["date", "TX_Close", "TX_Low", "TX_Volume"]
df_tx["date"] = pd.to_datetime(df_tx["date"]).dt.strftime("%Y-%m-%d")

# C. Philadelphia Semiconductor Index (^SOX)
sox_data = yf.download("^SOX", start=START_DATE, end=END_DATE, progress=False)
if isinstance(sox_data.columns, pd.MultiIndex):
    sox_series = sox_data["Close"]["^SOX"]
else:
    sox_series = sox_data["Close"]

df_sox = pd.DataFrame(
    {
        "date": pd.to_datetime(sox_series.index)
        .tz_localize(None)
        .strftime("%Y-%m-%d"),
        "SOX_Close": sox_series.values,
    }
)

# ==========================================
# 2. Medium-Term Feature Engineering
# ==========================================
print("\n2. Engineering Features for 20-Day Horizon...")


def calculate_shannon_entropy(series, bins=5):
    counts, _ = np.histogram(series, bins=bins)
    probs = counts / np.sum(counts)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


df_pred = pd.merge(df_sox, df_tx, on="date", how="outer").sort_values("date")
df_pred[["SOX_Close", "TX_Close", "TX_Low", "TX_Volume"]] = df_pred[
    ["SOX_Close", "TX_Close", "TX_Low", "TX_Volume"]
].ffill()

# Shannon Entropy & Volatility
df_pred["TX_Vol_Pct"] = df_pred["TX_Volume"].pct_change().fillna(0)
df_pred["TX_Entropy"] = (
    df_pred["TX_Vol_Pct"]
    .rolling(window=20)
    .apply(lambda x: calculate_shannon_entropy(x, bins=5), raw=True)
)

df_pred["TX_Ret"] = np.log(df_pred["TX_Close"]).diff()
df_pred["TX_Volat_20"] = df_pred["TX_Ret"].rolling(window=20).std()
df_pred["TX_Volat_60"] = df_pred["TX_Ret"].rolling(window=60).std()

# Shift Predictors by 1 Day
df_pred_shifted = df_pred[
    [
        "date",
        "SOX_Close",
        "TX_Close",
        "TX_Low",
        "TX_Entropy",
        "TX_Volat_20",
        "TX_Volat_60",
    ]
].copy()

df_pred_shifted.columns = [
    "date",
    "SOX_Close_Tminus1",
    "TX_Close_Tminus1",
    "TX_Low_Tminus1",
    "Entropy_Tminus1",
    "Volat20_Tminus1",
    "Volat60_Tminus1",
]

df_pred_shifted[
    [
        "SOX_Close_Tminus1",
        "TX_Close_Tminus1",
        "TX_Low_Tminus1",
        "Entropy_Tminus1",
        "Volat20_Tminus1",
        "Volat60_Tminus1",
    ]
] = df_pred_shifted[
    [
        "SOX_Close_Tminus1",
        "TX_Close_Tminus1",
        "TX_Low_Tminus1",
        "Entropy_Tminus1",
        "Volat20_Tminus1",
        "Volat60_Tminus1",
    ]
].shift(1)

df_merged = pd.merge(df_taiex, df_pred_shifted, on="date", how="inner").dropna()
df_final = df_merged.set_index("date").astype(float)

# Moving Average Bias
df_final["TAIEX_MA20_Bias"] = (
    df_final["TAIEX_Close"] - df_final["TAIEX_Close"].rolling(20).mean()
) / df_final["TAIEX_Close"].rolling(20).mean()

# ==========================================
# 3. Log Transforms & Target (20-Day Cumulative Minimum Low)
# ==========================================
df_log = np.log(
    df_final[
        [
            "TAIEX_Close",
            "SOX_Close_Tminus1",
            "TX_Close_Tminus1",
            "TX_Low_Tminus1",
        ]
    ]
)
df_log.columns = [f"log_{col}" for col in df_log.columns]

df_ret = df_log.diff()
df_ret.columns = [f"ret_{col.replace('log_', '')}" for col in df_log.columns]

# Target Y_20: Cumulative Minimum Low over the next 20 trading days
df_final["TAIEX_Min_Low_20D"] = (
    df_final["TAIEX_Low"].iloc[::-1].rolling(window=20).min().iloc[::-1]
)
df_ret["ret_TAIEX_Min20"] = np.log(df_final["TAIEX_Min_Low_20D"]) - np.log(
    df_final["TAIEX_Close"]
)

df_all = pd.concat(
    [df_final, df_log, df_ret], axis=1
).dropna(subset=["TAIEX_Min_Low_20D", "ret_TAIEX_Min20", "ret_TAIEX_Close"])

# Cointegration Residuals
coint_df = df_all[["log_TAIEX_Close", "log_SOX_Close_Tminus1"]]
johansen_test = coint_johansen(coint_df, det_order=0, k_ar_diff=1)
beta = johansen_test.evec[:, 0]

df_all["ECT_Tminus1"] = np.real(coint_df.dot(beta).shift(1))
df_all["Basis_Tminus1"] = (
    df_all["log_TAIEX_Close"] - df_all["log_TX_Close_Tminus1"]
)

df_model = df_all.dropna()

# ==========================================
# 4. 80% Train / 20% Out-of-Sample Evaluation
# ==========================================
print("\n=== 3. Training Quantile Random Forest (20-Day Horizon) ===")

X_cols = [
    "ret_SOX_Close_Tminus1",
    "ret_TX_Close_Tminus1",
    "ret_TX_Low_Tminus1",
    "ECT_Tminus1",
    "Basis_Tminus1",
    "Entropy_Tminus1",
    "Volat20_Tminus1",
    "Volat60_Tminus1",
    "TAIEX_MA20_Bias",
]

X = df_model[X_cols].astype(float)
y = df_model["ret_TAIEX_Min20"].astype(float)

train_size = int(len(df_model) * 0.8)

X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

qrf = RandomForestQuantileRegressor(
    n_estimators=300, max_depth=6, min_samples_leaf=3, random_state=42
)
qrf.fit(X_train, y_train)

# Out-of-Sample Evaluation on Test Set
preds_ret = qrf.predict(X_test, quantiles=[0.05, 0.50, 0.95])

taiex_close_test = df_model["TAIEX_Close"].iloc[train_size:].values
actual_min20_test = df_model["TAIEX_Min_Low_20D"].iloc[train_size:].values

pred_min20_q05 = taiex_close_test * np.exp(preds_ret[:, 0])
pred_min20_q50 = taiex_close_test * np.exp(preds_ret[:, 1])
pred_min20_q95 = taiex_close_test * np.exp(preds_ret[:, 2])

rmse = np.sqrt(mean_squared_error(actual_min20_test, pred_min20_q50))
mae = mean_absolute_error(actual_min20_test, pred_min20_q50)
mape = np.mean(np.abs((actual_min20_test - pred_min20_q50) / actual_min20_test)) * 100
coverage = np.mean(actual_min20_test >= pred_min20_q05) * 100

print("\n==========================================")
print(" 20-DAY CUMULATIVE LOW EVALUATION METRICS ")
print("==========================================")
print(f"1. Out-of-Sample RMSE : {rmse:.2f} points")
print(f"2. Out-of-Sample MAE  : {mae:.2f} points")
print(f"3. Out-of-Sample MAPE : {mape:.2f}%")
print(f"4. Q05 Safety Net Coverage Rate : {coverage:.2f}%")
print("==========================================\n")

# ==========================================
# 5. PREDICT FUTURE 20-DAY LOW (Real-time Prediction)
# ==========================================
print("=== 4. REAL-TIME PREDICTION FOR NEXT 20 TRADING DAYS ===")

# Use the latest available row (today: 2026-08-05)
latest_features = X.iloc[[-1]]
latest_close = df_model["TAIEX_Close"].iloc[-1]
latest_date = df_model.index[-1]

latest_preds_ret = qrf.predict(latest_features, quantiles=[0.05, 0.50, 0.95])[0]

future_q05_low = latest_close * np.exp(latest_preds_ret[0])
future_q50_low = latest_close * np.exp(latest_preds_ret[1])
future_q95_low = latest_close * np.exp(latest_preds_ret[2])

print(f"\n[Latest Market Close Date]: {latest_date}")
print(f"[TAIEX Current Close Price]: {latest_close:.2f} points\n")

print(f"--- FORECAST FOR NEXT 20 TRADING DAYS (Aug 06 ~ Early Sept 2026) ---")
print(
    f"► Expected Minimum Low (Q50 Median)     : {future_q50_low:.2f} pts"
    f" (Pullback: {(future_q50_low/latest_close - 1)*100:.2f}%)"
)
print(
    f"► Extreme Support Lower Bound (Q05 Safety) : {future_q05_low:.2f} pts"
    f" (Max Pullback: {(future_q05_low/latest_close - 1)*100:.2f}%)"
)
print(
    f"► Shallow Pullback Bound (Q95 Upper)      : {future_q95_low:.2f} pts"
    f" (Shallow Pullback: {(future_q95_low/latest_close - 1)*100:.2f}%)"
)
print(
    f"\n💡 Summary: The model forecasts TAIEX 20-day minimum low to land between"
    f" {future_q05_low:.0f} ~ {future_q50_low:.0f} points."
)

# ==========================================
# 6. Visualization
# ==========================================
results_df = pd.DataFrame(
    {
        "Actual_Min_20D": actual_min20_test,
        "Pred_Min_Q05": pred_min20_q05,
        "Pred_Min_Q50": pred_min20_q50,
        "Pred_Min_Q95": pred_min20_q95,
    },
    index=X_test.index,
)

plt.figure(figsize=(14, 7), dpi=120)
plt.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "default"
)

plt.plot(
    results_df.index,
    results_df["Actual_Min_20D"],
    label="Actual 20-Day Cumulative Minimum Low",
    color="#1f77b4",
    linewidth=2,
    alpha=0.9,
)

plt.plot(
    results_df.index,
    results_df["Pred_Min_Q05"],
    label="Pred 20D Low Q05 (Extreme Support Lower Bound)",
    color="#d62728",
    linestyle="--",
    linewidth=1.8,
    alpha=0.85,
)

plt.plot(
    results_df.index,
    results_df["Pred_Min_Q50"],
    label="Pred 20D Low Q50 (Median Forecast)",
    color="#2ca02c",
    linestyle=":",
    linewidth=1.5,
    alpha=0.7,
)

plt.fill_between(
    results_df.index,
    results_df["Pred_Min_Q05"],
    results_df["Pred_Min_Q95"],
    color="#d62728",
    alpha=0.12,
    label="90% Prediction Band (Q05-Q95)",
)

plt.title(
    f"TAIEX 20-Day Cumulative Minimum Low Prediction | Out-of-Sample MAPE:"
    f" {mape:.2f}% | Q05 Coverage: {coverage:.2f}%",
    fontsize=13,
    fontweight="bold",
    pad=15,
)
plt.xlabel("Trade Date", fontsize=12)
plt.ylabel("Index Points", fontsize=12)

plt.xticks(
    range(0, len(results_df), max(1, len(results_df) // 10)),
    results_df.index[::max(1, len(results_df) // 10)],
    rotation=30,
    ha="right",
)

plt.legend(loc="upper left", frameon=True, framealpha=0.9)
plt.tight_layout()
plt.show()
