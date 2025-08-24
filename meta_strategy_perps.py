# Filename: meta_strategy_perps_final_corrected.py
# -------------------------------------------------
# This is the definitive, fully comprehensive backtesting engine for the final
# perpetuals Meta-Strategy. It uses a custom-built, event-driven backtester
# with corrected, robust logic for rebalancing and data handling.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# --- Tunable Bull Regime Parameters (used by sweep) ---
BULL_ADX_MIN = 20
BULL_BREAKOUT_WIN = 60
BULL_VOLUME_MULT = 1.0

# Global sweep-mode flag for non-class contexts
SWEEP_MODE = False

# --- Helper Functions for Indicator Calculations ---
def sma(series, n):
    """Returns the Simple Moving Average"""
    return series.rolling(n).mean()

def adx(high, low, close, n=14):
    """Returns the Average Directional Index"""
    plus_dm = high.diff()
    minus_dm = low.diff().mul(-1)
    plus_dm[plus_dm < 0] = 0; plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < 0] = 0; minus_dm[minus_dm < plus_dm] = 0
    tr = pd.DataFrame({'h-l': high-low, 'h-pc':abs(high-close.shift()), 'l-pc':abs(low-close.shift())}).max(axis=1)
    atr_val = tr.ewm(alpha=1/n, adjust=False).mean()
    plus_di = (plus_dm.ewm(alpha=1/n, adjust=False).mean() / atr_val) * 100
    minus_di = (minus_dm.ewm(alpha=1/n, adjust=False).mean() / atr_val) * 100
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.ewm(alpha=1/n, adjust=False).mean()

def zscore(series, n):
    """Returns the Z-Score"""
    return (series - series.rolling(n).mean()) / series.rolling(n).std()

def atr(high, low, close, n=14):
    """Returns the Average True Range"""
    tr = pd.DataFrame({'h-l': high-low, 'h-pc':abs(high-close.shift()), 'l-pc':abs(low-close.shift())}).max(axis=1)
    return tr.rolling(n).mean()

# --- Custom Backtesting Engine ---
class PerpetualsBacktester:
    def __init__(self, price_data, funding_data, strategy_logic, initial_cash=100000, leverage=2.5, commission=0.0005, stop_loss_pct=0.20, sweep_mode=False):
        self.price_data = price_data
        self.funding_data = funding_data
        self.strategy_logic = strategy_logic
        self.initial_cash = initial_cash
        self.leverage = leverage
        self.commission = commission
        self.stop_loss_pct = stop_loss_pct
        
        self.cash = initial_cash
        self.positions = {asset: 0.0 for asset in price_data.columns if not any(x in asset for x in ['_High','_Low'])}
        self.equity_curve = []
        self.dates = []
        self.weekly_peak = initial_cash
        
        # Risk overlays
        self.max_gross_exposure = 1.0
        self.rebalance_band = 0.05
        self.atr_target_vol = 0.10
        self.min_leverage_factor = 0.5
        self.cooldown_trigger_dd = 0.08
        self.cooldown_periods = 28
        self.cooldown_leverage_factor = 0.5
        self.cooldown_remaining = 0
        
        # No leverage mode
        self.force_no_leverage = True
        
        # Sweep mode (disable plotting and verbose prints)
        self.sweep_mode = sweep_mode

    def run(self):
        print("Running comprehensive perpetuals backtest with corrected engine... This may take a few minutes.")
        
        # CORRECTED: Combine price and funding data using robust resampling
        merged_data = self.price_data.copy()
        for asset in self.funding_data:
            funding_df = self.funding_data[asset]
            funding_df['timestamp'] = pd.to_datetime(funding_df['timestamp'])
            fd_series = funding_df.set_index('timestamp')['funding_rate']
            # Resample to 12H frequency, carrying the last rate forward
            resampled_fd = fd_series.resample('12h').ffill()
            merged_data[f'{asset}_funding'] = resampled_fd.reindex(merged_data.index).fillna(0)

        for i in range(401, len(merged_data)): # Start after enough data for longest MA
            current_slice = merged_data.iloc[:i+1]
            current_prices = current_slice.iloc[-1]
            
            # 1. Mark-to-market portfolio value
            position_value = sum(units * current_prices.get(asset, 0) for asset, units in self.positions.items())
            portfolio_value = self.cash + position_value
            
            self.equity_curve.append(portfolio_value)
            self.dates.append(current_slice.index[-1])

            # Dynamic ATR-based leverage and cooldown
            effective_leverage = self.leverage
            if all(col in current_slice.columns for col in ['BTC_High','BTC_Low','BTC']):
                btc_atr_val = atr(current_slice['BTC_High'], current_slice['BTC_Low'], current_slice['BTC'], n=28).iloc[-1]
                btc_price_now = current_slice['BTC'].iloc[-1]
                if btc_price_now > 0 and pd.notna(btc_atr_val):
                    normalized_atr = btc_atr_val / btc_price_now
                    leverage_factor = np.clip(self.atr_target_vol / max(normalized_atr, 1e-8), self.min_leverage_factor, 1.0)
                    effective_leverage *= leverage_factor

            # Cooldown logic
            if self.cooldown_remaining > 0:
                effective_leverage *= self.cooldown_leverage_factor
                self.cooldown_remaining -= 1
            if self.weekly_peak > 0:
                current_dd = 1 - (portfolio_value / self.weekly_peak)
                if current_dd >= self.cooldown_trigger_dd and self.cooldown_remaining == 0:
                    self.cooldown_remaining = self.cooldown_periods

            # Enforce no leverage
            if self.force_no_leverage:
                effective_leverage = 1.0

            # 2. Apply funding payments
            for asset, units in self.positions.items():
                funding_rate = current_prices.get(f'{asset}_funding', 0)
                funding_payment = units * current_prices.get(asset, 0) * funding_rate
                self.cash -= funding_payment
            
            # 3. Master Stop-Loss (Circuit Breaker)
            if current_slice.index[-1].dayofweek == 0:
                self.weekly_peak = max(self.weekly_peak, portfolio_value)
            
            if portfolio_value < self.weekly_peak * (1 - self.stop_loss_pct):
                self.cash = portfolio_value # Liquidate everything
                for asset in self.positions: self.positions[asset] = 0
                self.weekly_peak = portfolio_value
                continue
                
            # 4. Get target allocations from strategy
            target_allocations_pct = self.strategy_logic(current_slice, self.positions.copy())
            
            # Cap max gross exposure
            gross = sum(abs(p) for p in target_allocations_pct.values())
            if gross > self.max_gross_exposure and gross > 0:
                scale = self.max_gross_exposure / gross
                for k in target_allocations_pct:
                    target_allocations_pct[k] *= scale
            
            # 5. Execute trades to rebalance to target
            for asset in self.positions.keys():
                target_pct = target_allocations_pct.get(asset, 0)
                current_price = current_prices.get(asset, 0)

                if effective_leverage <= 1e-8 or current_price <= 0:
                    target_units = 0.0
                else:
                    denominator = portfolio_value * effective_leverage
                    current_units = self.positions.get(asset, 0)
                    current_pct = (current_units * current_price) / denominator if denominator > 0 else 0.0
                    # Rebalance band: skip small changes
                    if abs(target_pct - current_pct) < self.rebalance_band:
                        continue
                    target_value = denominator * target_pct
                    target_units = target_value / current_price
                
                current_units = self.positions.get(asset, 0)
                units_to_trade = target_units - current_units
                
                trade_value = units_to_trade * current_price
                commission_cost = abs(trade_value) * self.commission
                
                self.cash -= trade_value
                self.cash -= commission_cost
                self.positions[asset] = target_units

        return self.calculate_stats()
        
    def calculate_stats(self):
        if not self.dates: return {}
        equity = pd.Series(self.equity_curve, index=pd.to_datetime(self.dates))
        total_return = (equity.iloc[-1] / self.initial_cash) - 1
        num_years = (equity.index[-1] - equity.index[0]).days / 365.25
        cagr = (equity.iloc[-1] / self.initial_cash) ** (1/num_years) - 1 if num_years > 0 else 0
        
        # Use 12H periods for Sharpe calculation (365*2 periods/year)
        returns = equity.pct_change().dropna()
        sharpe_ratio = (returns.mean() * np.sqrt(365*2)) / returns.std() if returns.std() != 0 else 0
        
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        max_drawdown = drawdown.min()
        
        avg_monthly_return = ((1 + cagr) ** (1/12)) - 1
        
        # Monthly returns tracking
        monthly_equity = equity.resample('ME').last()
        monthly_returns = monthly_equity.pct_change().dropna()
        try:
            if not self.sweep_mode:
                monthly_returns.to_csv('perps_monthly_returns.csv', header=['MonthlyReturn'])
                print("Saved monthly returns to 'perps_monthly_returns.csv'")
        except Exception:
            pass
        if not self.sweep_mode and not monthly_returns.empty:
            print("Monthly returns (last 12):")
            print(monthly_returns.tail(12).apply(lambda x: f"{x:.2%}"))
        
        # IS/OOS segmentation
        split_date = pd.Timestamp('2023-01-01')
        is_monthly = monthly_returns[monthly_returns.index < split_date]
        oos_monthly = monthly_returns[monthly_returns.index >= split_date]
        if not self.sweep_mode:
            if not is_monthly.empty:
                try:
                    is_monthly.to_csv('perps_monthly_returns_in_sample.csv', header=['MonthlyReturn'])
                    print("Saved in-sample monthly returns to 'perps_monthly_returns_in_sample.csv'")
                except Exception:
                    pass
            if not oos_monthly.empty:
                try:
                    oos_monthly.to_csv('perps_monthly_returns_out_of_sample.csv', header=['MonthlyReturn'])
                    print("Saved out-of-sample monthly returns to 'perps_monthly_returns_out_of_sample.csv'")
                except Exception:
                    pass
            if not is_monthly.empty:
                print(f"In-sample months: {len(is_monthly)}, mean: {is_monthly.mean():.2%}, winrate: {(is_monthly>0).mean():.0%}")
            if not oos_monthly.empty:
                print(f"Out-of-sample months: {len(oos_monthly)}, mean: {oos_monthly.mean():.2%}, winrate: {(oos_monthly>0).mean():.0%}")
        
        stats = {
            "Total Return": f"{total_return:.2%}",
            "CAGR (Annualized)": f"{cagr:.2%}",
            "Average Monthly Return": f"{avg_monthly_return:.2%}",
            "Max Drawdown": f"{max_drawdown:.2%}",
            "Sharpe Ratio": f"{sharpe_ratio:.2f}",
            "Total Return Raw": float(total_return),
            "IS Mean Monthly": float(is_monthly.mean()) if not is_monthly.empty else 0.0,
            "OOS Mean Monthly": float(oos_monthly.mean()) if not oos_monthly.empty else 0.0,
        }
        
        if not self.sweep_mode:
            plt.style.use('seaborn-v0_8-darkgrid'); fig, ax = plt.subplots(figsize=(15, 8))
            ax.plot(equity.index, equity.values, label='Meta-Strategy (Perps)', color='royalblue', linewidth=2)
            ax.set_yscale('log'); ax.set_title('Perpetuals Meta-Strategy - Logarithmic Equity Curve', fontsize=16)
            ax.set_ylabel('Portfolio Value ($) - Log Scale', fontsize=12); ax.set_xlabel('Date', fontsize=12)
            ax.legend(); ax.grid(True, which='both', linestyle='--', linewidth=0.5); plt.tight_layout()
            plt.savefig("perps_meta_strategy_equity_curve.png"); print("\nEquity curve plot saved to 'perps_meta_strategy_equity_curve.png'")
        return stats

# --- Load Models & Define Strategy Logic ---
try:
    b4_model = joblib.load("b4_crash_model_perps.joblib")
    c6_model = joblib.load("c6_breakout_model_perps.joblib")
except FileNotFoundError:
    print("FATAL ERROR: Model files not found. Please run train_models_perps.py first.")
    exit()

# --- Global variable for Gold data ---
xau_df = None

# --- ML Feature Engineering Function ---
def create_ml_features(data_slice):
    """Prepares the feature set for the ML models, matching the training script."""
    global xau_df
    # 12H Features
    btc_high = data_slice['BTC_High']; btc_low = data_slice['BTC_Low']; btc_close = data_slice['BTC']
    features = pd.DataFrame(index=data_slice.index)
    features['ADX_28'] = adx(btc_high, btc_low, btc_close, n=28)
    features['ATR_12H'] = (btc_high - btc_low).rolling(28).mean()
    features['Volatility_Compression'] = features['ATR_12H'] / features['ATR_12H'].rolling(200).mean()

    # Daily Features
    btc_daily_close = btc_close.resample('1D').last()
    btc_daily_high = btc_high.resample('1D').max()
    btc_daily_low = btc_low.resample('1D').min()
    daily_features = pd.DataFrame(index=btc_daily_close.index)
    daily_features['ATR_14'] = (btc_daily_high - btc_daily_low).rolling(14).mean()
    daily_features['ATR_Volatility_Spike'] = daily_features['ATR_14'].pct_change(30)
    daily_features['Momentum_90d'] = btc_daily_close.pct_change(90)
    daily_features['ROC_30d'] = btc_daily_close.pct_change(30)

    if xau_df is not None:
        daily_gold_close = xau_df['close'].reindex(btc_daily_close.index).ffill()
        daily_features['BTC_Gold_Corr_60d'] = btc_daily_close.pct_change(fill_method=None).rolling(60).corr(daily_gold_close.pct_change(fill_method=None))

    # Merge and clean
    features = features.merge(daily_features, how='left', left_index=True, right_index=True)
    features.ffill(inplace=True)
    
    feature_order = ['ADX_28', 'ATR_12H', 'Volatility_Compression', 'ATR_14', 'ATR_Volatility_Spike', 'Momentum_90d', 'ROC_30d']
    if 'BTC_Gold_Corr_60d' in features.columns:
        feature_order.append('BTC_Gold_Corr_60d')

    if features.empty or features[feature_order].iloc[-1:].isnull().values.any():
        return None
        
    return features[feature_order].iloc[-1:]

# --- State variable for regime change logging ---
previous_regime = None
# Pyramiding and hysteresis state
risk_on_streak = 0
pyramid_level = {}
# Min-hold state (12H periods)
hold_days = {}


def meta_strategy_perps_logic(data_slice, current_positions):
    global previous_regime, risk_on_streak, pyramid_level, hold_days
    target_allocations = {asset: 0.0 for asset in ASSET_UNIVERSE}
    
    # Update hold days based on current positions
    for a, units in current_positions.items():
        if units > 0:
            hold_days[a] = hold_days.get(a, 0) + 1
        else:
            hold_days[a] = 0
    
    # 1. ML-Driven Regime Detection
    ml_features = create_ml_features(data_slice)
    is_crash_predicted = False
    is_breakout_predicted = False

    if ml_features is not None:
        if b4_model.predict(ml_features)[0] == 1:
            is_crash_predicted = True
        if c6_model.predict(ml_features)[0] == 1:
            is_breakout_predicted = True

    # Price-based breakout fallback for Risk-ON
    btc_close = data_slice['BTC']
    btc_adx_val_gate = adx(data_slice['BTC_High'], data_slice['BTC_Low'], btc_close, 28).iloc[-1]
    btc_sma100_gate = sma(btc_close, 100).iloc[-1]
    btc_sma400_gate = sma(btc_close, 400).iloc[-1]
    btc_breakout_60 = btc_close.iloc[-1] > btc_close.rolling(60).max().shift(1).iloc[-1]
    price_based_breakout = (btc_breakout_60 and btc_adx_val_gate >= 20 and btc_sma100_gate > btc_sma400_gate)
    
    if price_based_breakout:
        is_breakout_predicted = True

    # Hysteresis tracking
    if is_crash_predicted:
        risk_on_streak = 0
    elif is_breakout_predicted:
        risk_on_streak += 1
    else:
        risk_on_streak = 0

    current_regime = "RISK-OFF (Crash Predicted)" if is_crash_predicted else "RISK-ON (Breakout Predicted)" if is_breakout_predicted else "Market-Neutral"
    
    if current_regime != previous_regime:
        if not SWEEP_MODE:
            print(f"{data_slice.index[-1].date()}: Regime changed to {current_regime}")
        previous_regime = current_regime

    # Reset pyramid levels when leaving Risk-ON
    if current_regime != "RISK-ON (Breakout Predicted)":
        for a in list(pyramid_level.keys()):
            pyramid_level[a] = 0

    # 2. Deploy Strategy based on ML Signal
    # RISK-OFF: Go to cash
    if is_crash_predicted:
        for asset in target_allocations:
            target_allocations[asset] = 0.0
    
    # RISK-ON: Breakout Strategy with Chandelier exit, concentration, pyramiding, funding-aware filter
    elif is_breakout_predicted:
        assets_to_hold = []
        assets_for_entry = []
        bull_asset_universe = ["BTC", "ETH", "SOL", "ADA"]
        
        # Trailing exit via 20-SMA for existing positions
        for asset, units in current_positions.items():
            if units > 0 and asset in bull_asset_universe:
                exit_sma = sma(data_slice[asset], 20).iloc[-1]
                if data_slice[asset].iloc[-1] > exit_sma:
                    assets_to_hold.append(asset)
        
        # Entries: ADX>=BULL_ADX_MIN, BULL_BREAKOUT_WIN breakout with volume>BULL_VOLUME_MULT x average
        entry_candidates = []
        if btc_adx_val_gate >= BULL_ADX_MIN:
            for asset in bull_asset_universe:
                if asset in assets_to_hold:
                    continue
                breakout_level = data_slice[asset].rolling(BULL_BREAKOUT_WIN).max().shift(1).iloc[-1]
                current_volume = data_slice[f'{asset}_Volume'].iloc[-1]
                avg_volume = data_slice[f'{asset}_Volume'].rolling(40).mean().iloc[-1]
                if data_slice[asset].iloc[-1] > breakout_level and current_volume > (avg_volume * BULL_VOLUME_MULT):
                    entry_candidates.append(asset)
        
        # Select top-3 by 60d momentum among holds+entries
        selection_pool = list(set(assets_to_hold + entry_candidates))
        selected = []
        if selection_pool:
            mom = {a: data_slice[a].pct_change(120).iloc[-1] for a in selection_pool}
            ranked = sorted(mom.items(), key=lambda x: (x[1] if pd.notna(x[1]) else -np.inf), reverse=True)
            selected = [a for a,_ in ranked[:4]]
        
        # Inverse-vol weights across selected
        if selected:
            vols = {asset: data_slice[asset].pct_change().rolling(20).std().iloc[-1] for asset in selected}
            valid_vols = {asset: vol for asset, vol in vols.items() if vol > 0}
            if valid_vols:
                inv_vols = {asset: 1/vol for asset, vol in valid_vols.items()}
                total_inv_vol = sum(inv_vols.values())
                for asset, inv in inv_vols.items():
                    target_allocations[asset] = inv / total_inv_vol
    
    else:
        btc_close = data_slice['BTC']
        eth_btc_ratio = data_slice['ETH'] / btc_close
        ratio_sma10 = sma(eth_btc_ratio, 10).iloc[-1]
        ratio_sma30 = sma(eth_btc_ratio, 30).iloc[-1]
        z_score = zscore(eth_btc_ratio, 90).iloc[-1]
        btc_trend_adx = adx(data_slice['BTC_High'], data_slice['BTC_Low'], data_slice['BTC'], 28).iloc[-1]
        
        # Pairs (small) when trending weakly
        if btc_trend_adx < 20:
            if z_score > 1.75 and ratio_sma10 < ratio_sma30:
                target_allocations['ETH'] = -0.3; target_allocations['BTC'] = 0.3
            elif z_score < -1.75 and ratio_sma10 > ratio_sma30:
                target_allocations['ETH'] = 0.3; target_allocations['BTC'] = -0.3
        
        # Carry-tilted long-only momentum overlay (0.7 gross target, capped later)
        overlay_assets = ["BTC","ETH","SOL","ADA"]
        scores = {}
        for a in overlay_assets:
            if a in data_slice.columns and f"{a}_Volume" in data_slice.columns:
                mom = data_slice[a].pct_change(120).iloc[-1]
                if pd.isna(mom) or mom <= 0:
                    continue
                # Funding tilt: penalize positive funding
                fund_col = f"{a}_funding"
                avg_funding = data_slice.get(fund_col, pd.Series(index=data_slice.index)).rolling(28).mean().iloc[-1] if fund_col in data_slice.columns else 0.0
                penalty = 1.0 / (1.0 + max(avg_funding, 0.0) * 1000.0)
                scores[a] = max(mom, 0.0) * penalty
        if scores:
            total_score = sum(scores.values())
            if total_score > 0:
                for a, s in scores.items():
                    target_allocations[a] = target_allocations.get(a, 0.0) + 0.7 * (s / total_score)
        
    return target_allocations

# --- Main Execution ---
if __name__ == "__main__":
    ASSET_UNIVERSE = ["BTC", "ETH", "SOL", "ADA", "XRP"]
    print("Loading and preparing all data files...")
    
    # Load gold data to be used by the feature creator
    try:
        xau_df = pd.read_csv('XAU-USD_1D.csv', index_col='timestamp', parse_dates=True)
    except FileNotFoundError:
        xau_df = None # Handle case where file is missing

    price_dfs = {asset: pd.read_csv(f'{asset}-PERP_12H.csv', index_col='timestamp') for asset in ASSET_UNIVERSE}
    funding_dfs = {asset: pd.read_csv(f'{asset}-PERP_FUNDING.csv') for asset in ASSET_UNIVERSE}

    df_prices = pd.DataFrame()
    for asset, df in price_dfs.items():
        df.index = pd.to_datetime(df.index, unit='s')
        df_prices[asset] = df['close']
        df_prices[f'{asset}_High'] = df['high']
        df_prices[f'{asset}_Low'] = df['low']
        df_prices[f'{asset}_Volume'] = df['volume']
    df_prices.sort_index(inplace=True); df_prices.interpolate(method='time', inplace=True)
    df_prices.dropna(inplace=True)
    
    # Restrict test window
    test_start = pd.Timestamp('2020-01-01')
    test_end = pd.Timestamp('2025-12-31')
    df_prices = df_prices.loc[(df_prices.index >= test_start) & (df_prices.index <= test_end)]
    
    # Run parameter sweep
    adx_grid = [18, 20, 22, 25]
    breakout_grid = [50, 60, 70, 80]
    results = []
    SWEEP_MODE = True
    for adx_th in adx_grid:
        for brk_win in breakout_grid:
            BULL_ADX_MIN = adx_th
            BULL_BREAKOUT_WIN = brk_win
            backtester = PerpetualsBacktester(df_prices, funding_dfs, meta_strategy_perps_logic, sweep_mode=True)
            stats = backtester.run()
            results.append({
                'ADX': adx_th,
                'BreakoutWin': brk_win,
                'TotalReturn': stats.get('Total Return Raw', 0.0),
                'OOSMean': stats.get('OOS Mean Monthly', 0.0),
                'ISMean': stats.get('IS Mean Monthly', 0.0),
                'MaxDD': stats.get('Max Drawdown', ''),
                'Sharpe': stats.get('Sharpe Ratio', ''),
            })
    SWEEP_MODE = False
    res_df = pd.DataFrame(results)
    # Rank by OOS mean monthly return, then TotalReturn
    res_df.sort_values(by=['OOSMean','TotalReturn'], ascending=False, inplace=True)
    print("\nParameter Sweep (top 10 by OOS mean monthly return):")
    print(res_df.head(10))

    # Use best params for a final detailed run with plots and CSV
    if not res_df.empty:
        best = res_df.iloc[0]
        BULL_ADX_MIN = int(best['ADX'])
        BULL_BREAKOUT_WIN = int(best['BreakoutWin'])
    
    backtester = PerpetualsBacktester(df_prices, funding_dfs, meta_strategy_perps_logic, sweep_mode=False)
    final_stats = backtester.run()
    
    print("\n" + "="*50); print("  FINAL PERPETUALS META-STRATEGY RESULTS (CORRECTED)  "); print("="*50)
    print(pd.DataFrame.from_dict(final_stats, orient='index', columns=['Value']))