```markdown
# 📈 TAIEX 20-Day Intraday Low Prediction Model
> 結合計量經濟學 (Johansen Cointegration) 與分位數隨機森林 (QRF) 的台股加權指數波段低點預測模型

![Python Version](https://img.shields.org/badge/Python-3.10%2B-blue)
![License](https://img.shields.org/badge/License-MIT-green)

本專案旨在預測未來 20 個交易日內台股加權指數（TAIEX）的波段最低點與極限防禦下軌。透過跨時區資料對齊，整合美股費城半導體指數（SOX）與台指期夜盤（TX）資訊，並結合資訊理論中的夏農熵（Shannon Entropy）與計量協整合殘差，解決傳統機器學習缺乏長期點位錨定力及無法捕捉尾端恐慌風險的盲區。

---

## ✨ 專案核心特色 (Key Features)

- 跨時區資料對齊 (Time Alignment)： 使用 $T-1$ 日的美股 SOX 與台指期夜盤數據預測 $T$ 日開盤後的走勢，完全杜絕資料洩漏 (Data Leakage)。
- 夏農熵 (Shannon Entropy) 恐慌濾網： 引進資訊理論中的夏農熵 ($H = -\sum p_i \log_2 p_i$) 捕捉交易量異常變動，動態調整恐慌發酵時的下軌防禦區間。
- Johansen 協整合錨定 (ECT)： 檢定台股與美股的長期均衡關係，導出誤差修正項 (Error Correction Term)，提供機器學習長期的均值回歸 (Mean Reversion) 拉力。
- 分位數隨機森林 (Quantile Random Forest, QRF)： 不僅預測合理低點中位數 ($Q_{50}$)，更能產出 95% 統計信心度下的極限防禦下軌 ($Q_{05}$)。

---

## 📊 回測表現 (Out-of-Sample Results)

經 20% 未知樣本外測試（Out-of-Sample Test），模型在指數 >40,000 點的歷史高檔區間展現極佳的穩定度：

| 評估指標 (Error Metrics) | 評估結果 | 說明 |
| :--- | :--- | :--- |
| MAPE (平均絕對百分比誤差) | 2.13% | 一個月累積最低點預測偏差控制在 2.1% 左右 |
| MAE (平均絕對點位誤差) | 685.20 點 | 高檔大盤下平均點位偏差約 685 點 |
| $Q_{05}$ 安全防禦覆蓋率 | 95.03% | 95.03% 的交易日實際最低價均未跌破預測下軌 (符合理論 95% 防禦邊界) |

---

## 🛠️ 環境安裝與套件需求 (Installation)

請確保你的 Python 版本為 `3.10` 以上，並透過以下命令安裝專案所需套件：

```bash
pip install numpy pandas requests yfinance statsmodels scikit-garden quantile-forest matplotlib

```

---

## 🔑 關鍵設定：API Token 設定說明

本專案使用 [FinMind Trade API](https://finmindtrade.com/) 擷取台股加權指數與台指期合約資料。

為了維護個人資訊安全，程式碼中的 `FINMIND_TOKEN` 欄位已留空。在使用本程式碼前，請先註冊 FinMind 並取得你的個人 API Token：

```python
# -------------------------------------------------------------------
# ⚠️ 請在下方的 FINMIND_TOKEN 填入你自己的 FinMind API Token
# -------------------------------------------------------------------
FINMIND_TOKEN = "YOUR_FINMIND_API_TOKEN_HERE"  # <--- 在此填入你的 Token
URL = "[https://api.finmindtrade.com/api/v4/data](https://api.finmindtrade.com/api/v4/data)"

```

> 💡 如何取得 FinMind Token？
> 1. 前往 [FinMind 官網](https://finmindtrade.com/) 註冊免費帳號。
> 2. 登入後至個人後台即可複製你的專屬 API Token。
> 
> 

---

## 🚀 快速開始 (Quick Start)

1. 複製本專案庫 (Clone Repository)：
```bash
git clone [https://github.com/your-username/taiex-20d-low-prediction.git](https://github.com/your-username/taiex-20d-low-prediction.git)
cd taiex-20d-low-prediction

```


2. 設定 API Token：
開啟程式碼主檔案（如 `main.py`），填入你的 FinMind Token。
3. 執行預測腳本：
```bash
python main.py

```


4. 輸出範例：
執行後系統會自動對齊資料、進行 Johansen 協整合檢定、訓練分位數隨機森林模型，並輸出未來 20 個交易日的預測區間與 Matplotlib 視覺化圖表。

---

## 💡 實盤策略落地應用

* $Q_{50}$ (中位數預測)： 作為未來一個月內常態回檔、分批建倉或加碼的第一波合理買點。
* $Q_{05}$ (極限防禦下軌)： 95% 信心水準下的極限強支撐。適合做黑天鵝恐慌時的抄底點位，或作為賣出月選擇權 Put (Sell Put) 的履約價安全邊界。

---

## 📜 授權條款 (License)

本專案採用 [MIT License](https://www.google.com/search?q=LICENSE) 授權。

---

## 與真實資料的對比（2026 年 7 月 9 日至 2026 年 8 月 5 日（共 20 個交易日）這段期間的台股加權指數波段最低點）
==========================================================================
  TAIEX FORECAST VS REALITY EVALUATION (2026-07-09 ~ 2026-08-05)
==========================================================================
1. 基準日 (Base Date: 2026-07-08) 加權指數收盤價 : 45734.41 點

2. 模型對未來 20 天的預測區間 (Forecasted Range):
   ► Q95 淺幅拉回上軌 (Shallow Bound) : 45568.27 點
   ► Q50 中位數預期低點 (Median Forecast): 44602.12 點 (預估拉回 -2.48%)
   ► Q05 極限防禦下軌 (Extreme Support) : 42084.88 點 (預估最大拉回 -7.98%)

3. 這 20 個交易日的真實市場表現 (Actual Reality):
   ► 實際發生的最低點 (Actual Min Low) : 39384.85 點 (發生於 2026-07-29)
   ► 實際最大拉回幅度 (Actual Pullback) : -13.88%

4. 預測誤差分析與防禦檢定 (Error Analysis):
   ► Q50 中位數預測點位偏差 (Points Error) : -5217.27 點
   ► Q50 中位數預測絕對百分比誤差 (APE) : 11.70%
   ► Q05 防禦邊界檢定結果 : ❌ 跌破下軌 (實際最低點 39384.85 < Q05 下軌 42084.88)
==========================================================================
