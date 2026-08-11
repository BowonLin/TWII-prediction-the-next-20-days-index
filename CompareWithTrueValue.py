import pandas as pd

# ==========================================
# 1. 設定基準與預測點位 (從上次模型輸出帶入)
# ==========================================
base_date = "2026-07-08"
base_close = 45734.41  # 2026-07-08 加權指數收盤價

pred_q95_low = 45568.27  # Q95 淺幅拉回上軌
pred_q50_low = 44602.12  # Q50 中位數預估低點
pred_q05_low = 42084.88  # Q05 極限防禦下軌

# ==========================================
# 2. 2026-07-09 ~ 2026-08-05 的實際加權指數每日最低價 (TAIEX Low)
# ==========================================
actual_data = {
    "2026-07-09": 45610.12,
    "2026-07-10": 45580.45,
    "2026-07-13": 45420.30,
    "2026-07-14": 45210.88,
    "2026-07-15": 45100.25,
    "2026-07-16": 44980.60,
    "2026-07-17": 44850.15,
    "2026-07-20": 44710.90,
    "2026-07-21": 44650.33,
    "2026-07-22": 44520.10,
    "2026-07-23": 44340.76,
    "2026-07-24": 43607.40,
    "2026-07-27": 42969.48,
    "2026-07-28": 41565.00,
    "2026-07-29": 39384.85,  # 這 20 個交易日的全期最低點 (黑天鵝暴跌日)
    "2026-07-30": 39404.65,
    "2026-07-31": 41610.41,
    "2026-08-03": 42780.42,
    "2026-08-04": 42895.81,
    "2026-08-05": 43809.83,
}

df_actual = pd.DataFrame(
    list(actual_data.items()), columns=["Date", "TAIEX_Low"]
)
df_actual.set_index("Date", inplace=True)

# ==========================================
# 3. 算出區間極值與誤差
# ==========================================
actual_min_low = df_actual["TAIEX_Low"].min()
actual_min_date = df_actual["TAIEX_Low"].idxmin()

# 誤差計算
q50_error_pts = actual_min_low - pred_q50_low
q50_error_pct = (q50_error_pts / pred_q50_low) * 100

actual_pullback_pct = (actual_min_low / base_close - 1) * 100
pred_q50_pullback_pct = (pred_q50_low / base_close - 1) * 100
pred_q05_pullback_pct = (pred_q05_low / base_close - 1) * 100

is_q05_held = actual_min_low >= pred_q05_low

# ==========================================
# 4. 格式化印出比較報告
# ==========================================
print("==========================================================================")
print("  TAIEX FORECAST VS REALITY EVALUATION (2026-07-09 ~ 2026-08-05)")
print("==========================================================================")
print(f"1. 基準日 (Base Date: {base_date}) 加權指數收盤價 : {base_close:.2f} 點\n")

print("2. 模型對未來 20 天的預測區間 (Forecasted Range):")
print(f"   ► Q95 淺幅拉回上軌 (Shallow Bound) : {pred_q95_low:.2f} 點")
print(
    f"   ► Q50 中位數預期低點 (Median Forecast): {pred_q50_low:.2f} 點"
    f" (預估拉回 {pred_q50_pullback_pct:.2f}%)"
)
print(
    f"   ► Q05 極限防禦下軌 (Extreme Support) : {pred_q05_low:.2f} 點 (預估最大拉回"
    f" {pred_q05_pullback_pct:.2f}%)\n"
)

print("3. 這 20 個交易日的真實市場表現 (Actual Reality):")
print(
    f"   ► 實際發生的最低點 (Actual Min Low) : {actual_min_low:.2f} 點 (發生於"
    f" {actual_min_date})"
)
print(f"   ► 實際最大拉回幅度 (Actual Pullback) : {actual_pullback_pct:.2f}%\n")

print("4. 預測誤差分析與防禦檢定 (Error Analysis):")
print(f"   ► Q50 中位數預測點位偏差 (Points Error) : {q50_error_pts:+.2f} 點")
print(f"   ► Q50 中位數預測絕對百分比誤差 (APE) : {abs(q50_error_pct):.2f}%")
if is_q05_held:
    print(
        f"   ► Q05 防禦邊界檢定結果 : ✅ 成功守住！(實際最低點 {actual_min_low:.2f}"
        f" > Q05 下軌 {pred_q05_low:.2f})"
    )
else:
    print(
        f"   ► Q05 防禦邊界檢定結果 : ❌ 跌破下軌 (實際最低點 {actual_min_low:.2f}"
        f" < Q05 下軌 {pred_q05_low:.2f})"
    )
print("==========================================================================")
