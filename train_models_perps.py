# Filename: train_models_perps.py
# ---------------------------------
# This script trains the ML models needed for the perpetuals Meta-Strategy using the provided 12H data.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import xgboost as xgb

print("Starting the Perpetuals ML Model Factory...")

# --- Configuration & Data Loading ---
ASSETS = ['BTC', 'ETH', 'SOL', 'ADA', 'XRP']
PRICE_FILES = {asset: f'{asset}-PERP_12H.csv' for asset in ASSETS}

# Load all price data
price_data = {}
for asset, file in PRICE_FILES.items():
    try:
        price_data[asset] = pd.read_csv(file)
    except FileNotFoundError:
        print(f"Error: Could not find {file}. Please ensure all CSV files are in the same folder.")
        exit()

# Load Gold data for correlation feature
try:
    xau_data = pd.read_csv('XAU-USD_1D.csv', index_col='timestamp', parse_dates=True)
except FileNotFoundError:
    print("Warning: Could not find XAU-USD_1D.csv. Gold correlation feature will be skipped.")
    xau_data = None

# Use BTC as the primary market indicator
btc_data = price_data['BTC'].copy()
btc_data['timestamp'] = pd.to_datetime(btc_data['timestamp'], unit='s')
btc_data.set_index('timestamp', inplace=True)

# Resample to daily for longer-term feature calculation
btc_daily = btc_data['close'].resample('1D').last().to_frame()
btc_daily['high'] = btc_data['high'].resample('1D').max()
btc_daily['low'] = btc_data['low'].resample('1D').min()
btc_daily.interpolate(method='linear', inplace=True)

# --- Feature Engineering ---
print("Engineering features for the models...")
features = pd.DataFrame(index=btc_data.index)

# B4 Crash Forecaster Features (signals of panic, calculated on daily data)
daily_features = pd.DataFrame(index=btc_daily.index)
daily_features['ATR_14'] = (btc_daily['high'] - btc_daily['low']).rolling(14).mean()
daily_features['ATR_Volatility_Spike'] = daily_features['ATR_14'].pct_change(30)
daily_features['Momentum_90d'] = btc_daily['close'].pct_change(90)
daily_features['ROC_30d'] = btc_daily['close'].pct_change(30) # NEW FEATURE: Rate of Change

# NEW FEATURE: Bitcoin-Gold Correlation
if xau_data is not None:
    daily_gold_close = xau_data['close'].reindex(btc_daily.index).ffill()
    daily_features['BTC_Gold_Corr_60d'] = btc_daily['close'].pct_change().rolling(60).corr(daily_gold_close.pct_change())

# C6 Breakout Forecaster Features (signals of chop ending, on 12H data)
def calculate_adx(high, low, close, n=14):
    plus_dm = high.diff(); minus_dm = low.diff().mul(-1)
    plus_dm[plus_dm < 0] = 0; plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < 0] = 0; minus_dm[minus_dm < plus_dm] = 0
    tr = pd.DataFrame({'h-l': high-low, 'h-pc':abs(high-close.shift()), 'l-pc':abs(low-close.shift())}).max(axis=1)
    atr_val = tr.ewm(alpha=1/n, adjust=False).mean()
    plus_di = (plus_dm.ewm(alpha=1/n, adjust=False).mean() / atr_val) * 100
    minus_di = (minus_dm.ewm(alpha=1/n, adjust=False).mean() / atr_val) * 100
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.ewm(alpha=1/n, adjust=False).mean()

features['ADX_28'] = calculate_adx(btc_data['high'], btc_data['low'], btc_data['close'], n=28) # 14-day equivalent
features['ATR_12H'] = (btc_data['high'] - btc_data['low']).rolling(28).mean()
features['Volatility_Compression'] = features['ATR_12H'] / features['ATR_12H'].rolling(200).mean() # 100-day equivalent

# Merge daily features back to 12H index
features = features.merge(daily_features, how='left', left_index=True, right_index=True)
features.ffill(inplace=True)
# REMOVED: features.dropna(inplace=True) -> Will be handled at the end.

# --- Labeling the Data ---
print("Creating labels for future outcomes...")
labels = pd.DataFrame(index=features.index)

# B4 Label: Did a >30% crash happen in the next 30 days? (Uses daily data)
rolling_max = btc_daily['close'].rolling(30).max().shift(-30)
future_min = btc_daily['close'].rolling(30).min().shift(-30)
daily_labels = pd.DataFrame(index=btc_daily.index)
daily_labels['is_crash'] = (future_min / rolling_max - 1) < -0.3
labels = labels.merge(daily_labels, how='left', left_index=True, right_index=True)
labels.ffill(inplace=True)

# C6 Label: Did a >15% breakout happen in the next 20 periods (10 days)?
rolling_range_high = btc_data['close'].rolling(40).max() # 20-day range
rolling_range_low = btc_data['close'].rolling(40).min() # 20-day range
future_high = btc_data['close'].rolling(20).max().shift(-20) # 10-day future
future_low = btc_data['close'].rolling(20).min().shift(-20)
labels['is_breakout'] = (future_high > rolling_range_high * 1.15) | (future_low < rolling_range_low * 0.85)

# --- Data Alignment & Final Cleaning ---
# Combine features and labels first
combined = features.join(labels)

# --- START: ROBUST FEATURE HANDLING ---
# If a feature column is all NaN, it means the data was unavailable. Warn and drop it.
if 'BTC_Gold_Corr_60d' in combined.columns and combined['BTC_Gold_Corr_60d'].isnull().all():
    print("Warning: BTC-Gold correlation feature failed to compute. Dropping this feature for the current run.")
    combined.drop(columns=['BTC_Gold_Corr_60d'], inplace=True)
    # Also remove it from the feature list for the backtester's ML helper
    if 'BTC_Gold_Corr_60d' in features.columns:
        features.drop(columns=['BTC_Gold_Corr_60d'], inplace=True)
# --- END: ROBUST FEATURE HANDLING ---

# Now, drop all rows with any NaNs from the combined dataframe
combined.dropna(inplace=True)

# Separate them back out, perfectly aligned and cleaned
aligned_features = combined[features.columns]
aligned_labels = combined[labels.columns]

# Convert labels to int type for the models
aligned_labels['is_crash'] = aligned_labels['is_crash'].astype(int)
aligned_labels['is_breakout'] = aligned_labels['is_breakout'].astype(int)

# --- Train, Evaluate, and Save Models ---
# B4 Crash Forecaster
print("\n--- Training B4 Crash Forecaster Model (XGBoost) ---")
y_crash = aligned_labels['is_crash'].dropna()
X_crash = aligned_features.loc[y_crash.index]
X_train, X_test, y_train, y_test = train_test_split(X_crash, y_crash, test_size=0.2, random_state=42, shuffle=False)

scale_pos_weight_crash = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1
b4_model = xgb.XGBClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8,
    colsample_bytree=0.8, use_label_encoder=False, eval_metric='logloss',
    scale_pos_weight=scale_pos_weight_crash, random_state=42
)
b4_model.fit(X_train, y_train)
print("B4 Model Evaluation:\n", classification_report(y_test, b4_model.predict(X_test)))
joblib.dump(b4_model, "b4_crash_model_perps.joblib")
print("B4 model saved to b4_crash_model_perps.joblib")

# C6 Breakout Forecaster
print("\n--- Training C6 Breakout Forecaster Model (XGBoost) ---")
y_breakout = aligned_labels['is_breakout'].dropna()
X_breakout = aligned_features.loc[y_breakout.index]
X_train, X_test, y_train, y_test = train_test_split(X_breakout, y_breakout, test_size=0.2, random_state=42, shuffle=False)

scale_pos_weight_breakout = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1
c6_model = xgb.XGBClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8,
    colsample_bytree=0.8, use_label_encoder=False, eval_metric='logloss',
    scale_pos_weight=scale_pos_weight_breakout, random_state=42
)
c6_model.fit(X_train, y_train)
print("C6 Model Evaluation:\n", classification_report(y_test, c6_model.predict(X_test)))
joblib.dump(c6_model, "c6_breakout_model_perps.joblib")
print("C6 model saved to c6_breakout_model_perps.joblib")

print("\nModel Factory process complete.")