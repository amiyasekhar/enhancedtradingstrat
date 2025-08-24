#!/usr/bin/env python3
"""
Historical downloader for:
- Binance USDT-M perpetuals (BTC, ETH, SOL, ADA, XRP): 12h OHLCV + funding rates
- Gold XAU/USD (daily) via yfinance

Outputs:
- ASSET-PERP_12H.csv  with columns: timestamp, open, high, low, close, volume
- ASSET-PERP_FUNDING.csv with columns: timestamp, funding_rate
- XAU-USD_1D.csv with columns: timestamp, open, high, low, close, volume

Date range attempted: 2012-01-01 to 2025-08-24 (gracefully falls back to available data)
"""

import os
import time
import math
import traceback
from datetime import datetime, timedelta, timezone

import ccxt
import pandas as pd
import yfinance as yf

# ==========================
# User-configurable settings
# ==========================

# Optional Binance API Keys (not required for public market data/funding history, but supported)
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")   # or paste directly: "your_key"
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")  # or paste directly: "your_secret"

# Rate limiting & retries
RATE_LIMIT_SLEEP = 0.25  # seconds between API calls to be nice to the exchange
MAX_RETRIES = 3          # per paginated call

# Date range (attempted)
START_DATE_STR = "2012-01-01"
END_DATE_STR   = "2025-08-24"

# Assets
CRYPTO_ASSETS = ["BTC", "ETH", "SOL", "ADA", "XRP"]  # USDT-M perpetuals
TIMEFRAME = "12h"                                    # 12-hour candles for crypto
GOLD_TICKER = "XAUUSD=X"                             # yfinance ticker for XAU/USD spot
GOLD_OUTFILE = "XAU-USD_1D.csv"

# ==========================
# Helpers
# ==========================

def to_ms(dt: datetime) -> int:
    """UTC datetime -> milliseconds since epoch."""
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)

def ms_to_s(ms: int) -> int:
    """Milliseconds -> integer seconds."""
    return int(ms // 1000)

def ms_to_datestr(ms: int) -> str:
    """Milliseconds -> 'YYYY-MM-DD' in UTC."""
    return datetime.utcfromtimestamp(ms / 1000.0).strftime("%Y-%m-%d")

def parse_timeframe_ms(exchange: ccxt.Exchange, timeframe: str) -> int:
    """Return timeframe length in ms using ccxt's parse_timeframe."""
    return int(exchange.parse_timeframe(timeframe) * 1000)

def safe_sleep(seconds: float):
    try:
        time.sleep(seconds)
    except Exception:
        pass

# ==========================
# CCXT Exchange Setup
# ==========================

def init_binance_usdm():
    """
    Initialize Binance USDT-M futures exchange (perpetuals).
    ccxt.binanceusdm defaults to USDT-M futures.
    """
    exchange = ccxt.binanceusdm({
        "apiKey": BINANCE_API_KEY,
        "secret": BINANCE_API_SECRET,
        "enableRateLimit": True,
        # 'rateLimit' is managed by ccxt; we still add our own sleeps to be conservative
        "options": {
            "adjustForTimeDifference": True,
        },
        "timeout": 30000,
    })
    return exchange

# ==========================
# Fetchers (Crypto OHLCV)
# ==========================

def fetch_ohlcv_all(exchange: ccxt.Exchange, symbol: str, timeframe: str,
                    since_ms: int, until_ms: int, limit: int = 1500) -> pd.DataFrame:
    """
    Paginated fetch of OHLCV from 'since_ms' up to 'until_ms' (inclusive attempt).
    Returns a DataFrame with columns: timestamp(ms), open, high, low, close, volume.
    """
    tf_ms = parse_timeframe_ms(exchange, timeframe)
    all_rows = []
    cursor = since_ms
    first_batch = True

    while True:
        # Stop if we have moved beyond until
        if cursor > until_ms + tf_ms:
            break

        attempt = 0
        last_err = None
        while attempt < MAX_RETRIES:
            try:
                # Binance returns candles starting at 'cursor', up to 'limit'
                batch = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=limit)
                break
            except Exception as e:
                last_err = e
                attempt += 1
                safe_sleep(RATE_LIMIT_SLEEP * (attempt + 1))
        if attempt == MAX_RETRIES and last_err is not None:
            print(f"[WARN] fetch_ohlcv_all: retries exhausted for {symbol} @ {ms_to_datestr(cursor)}; {type(last_err).__name__}: {last_err}")
            # try to move cursor forward by one interval to avoid infinite loop
            cursor += tf_ms
            continue

        safe_sleep(RATE_LIMIT_SLEEP)

        if not batch:
            # No more data
            break

        # Record earliest on first batch to warn if start is later than requested
        if first_batch:
            first_batch = False
            earliest_ms = batch[0][0]
            if earliest_ms > since_ms + tf_ms:
                print(f"[INFO] {symbol}: data not found before {ms_to_datestr(earliest_ms)}; fetching available data from that date.")

        all_rows.extend(batch)
        last_ts = batch[-1][0]
        if last_ts == cursor:
            # safety to avoid a stuck loop
            cursor += tf_ms
        else:
            cursor = last_ts + tf_ms

        # Stop if we've covered past 'until_ms'
        if last_ts >= until_ms:
            break

    if not all_rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(all_rows, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
    # Convert to seconds and keep required columns/order
    df["timestamp"] = df["timestamp_ms"].apply(ms_to_s)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    # Deduplicate/sort
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    # Clip to until_ms (start of candle <= until)
    until_s = ms_to_s(until_ms)
    df = df[df["timestamp"] <= until_s].reset_index(drop=True)
    return df

def fetch_price_data(exchange: ccxt.Exchange, asset: str, start_dt: datetime, end_dt: datetime):
    """
    Fetch 12h OHLCV for ASSET/USDT perpetual and write ASSET-PERP_12H.csv
    """
    # Resolve the correct USDT-M perpetual symbol (handles formats like BTC/USDT or BTC/USDT:USDT)
    def resolve_usdm_symbol(ex: ccxt.Exchange, base_asset: str) -> str:
        markets = ex.load_markets()
        # First, prefer explicit matches that might exist
        for candidate in [f"{base_asset}/USDT", f"{base_asset}/USDT:USDT", f"{base_asset}USDT"]:
            if candidate in markets:
                m = markets[candidate]
                # Ensure it's a swap/perpetual
                if m.get("swap") or m.get("contract"):
                    return candidate

        # Otherwise, scan markets for a USDT-quoted swap
        candidates = []
        for sym, m in markets.items():
            try:
                if m.get("base") == base_asset and m.get("quote") == "USDT" and (m.get("swap") or m.get("contract")):
                    candidates.append((sym, m))
            except Exception:
                continue
        if not candidates:
            raise ValueError(f"Symbol {base_asset}/USDT not found on Binance USDT-M futures.")
        # Prefer linear contracts if available
        for sym, m in candidates:
            if m.get("linear") is True:
                return sym
        return candidates[0][0]

    symbol = resolve_usdm_symbol(exchange, asset)
    out_file = f"{asset}-PERP_12H.csv"

    since_ms = to_ms(start_dt)
    until_ms = to_ms(end_dt)

    print(f"[START] {asset} 12h OHLCV ({symbol}) from {START_DATE_STR} to {END_DATE_STR}")
    df = fetch_ohlcv_all(exchange, symbol, TIMEFRAME, since_ms, until_ms, limit=1500)

    if df.empty:
        print(f"[WARN] No OHLCV data returned for {asset} (perpetual).")
    else:
        # Ensure numeric types
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.to_csv(out_file, index=False)
        print(f"[OK] Wrote {out_file} with {len(df)} rows.")

# ==========================
# Fetchers (Funding Rates)
# ==========================

def fetch_funding_rate_history(exchange: ccxt.Exchange, symbol: str, since_ms: int, until_ms: int, batch_limit: int = 1000) -> pd.DataFrame:
    """
    Paginated funding rate history using ccxt's unified fetchFundingRateHistory.
    Returns DataFrame with columns: timestamp(ms), fundingRate
    """
    all_items = []
    cursor = since_ms
    first_batch = True

    while True:
        if cursor > until_ms + 8 * 60 * 60 * 1000:  # move beyond by ~one funding interval
            break

        attempt = 0
        last_err = None
        this_batch = None
        while attempt < MAX_RETRIES:
            try:
                # Some exchanges support 'limit', ccxt normalizes fields.
                this_batch = exchange.fetch_funding_rate_history(symbol, since=cursor, limit=batch_limit)
                break
            except Exception as e:
                last_err = e
                attempt += 1
                safe_sleep(RATE_LIMIT_SLEEP * (attempt + 1))
        if attempt == MAX_RETRIES and last_err is not None:
            print(f"[WARN] funding: retries exhausted for {symbol} @ {ms_to_datestr(cursor)}; {type(last_err).__name__}: {last_err}")
            cursor += 8 * 60 * 60 * 1000  # skip one funding interval to avoid lock
            continue

        safe_sleep(RATE_LIMIT_SLEEP)

        if not this_batch:
            break

        # First-batch warning if we started later than requested
        if first_batch:
            first_batch = False
            earliest_ms = this_batch[0].get("timestamp")
            if earliest_ms and earliest_ms > since_ms + 1:
                print(f"[INFO] {symbol}: funding data not found before {ms_to_datestr(earliest_ms)}; fetching available data from that date.")

        all_items.extend(this_batch)
        last_ts = this_batch[-1].get("timestamp")
        if last_ts is None:
            # Defensive: step cursor by 8h
            cursor += 8 * 60 * 60 * 1000
        else:
            # Move just beyond last timestamp
            cursor = last_ts + 1

        if last_ts and last_ts >= until_ms:
            break

    if not all_items:
        return pd.DataFrame(columns=["timestamp", "funding_rate"])

    # Normalize to DataFrame
    rows = []
    for it in all_items:
        ts = it.get("timestamp")
        rate = it.get("fundingRate")
        if ts is None or rate is None:
            continue
        rows.append((ms_to_s(int(ts)), float(rate)))

    df = pd.DataFrame(rows, columns=["timestamp", "funding_rate"])
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    # Clip to until
    until_s = ms_to_s(until_ms)
    df = df[df["timestamp"] <= until_s].reset_index(drop=True)
    return df

def fetch_funding_data(exchange: ccxt.Exchange, asset: str, start_dt: datetime, end_dt: datetime):
    """
    Fetch funding rate history for ASSET/USDT perpetual and write ASSET-PERP_FUNDING.csv
    """
    # Reuse the same resolver logic as OHLCV
    def resolve_usdm_symbol(ex: ccxt.Exchange, base_asset: str) -> str:
        markets = ex.load_markets()
        for candidate in [f"{base_asset}/USDT", f"{base_asset}/USDT:USDT", f"{base_asset}USDT"]:
            if candidate in markets:
                m = markets[candidate]
                if m.get("swap") or m.get("contract"):
                    return candidate
        candidates = []
        for sym, m in markets.items():
            try:
                if m.get("base") == base_asset and m.get("quote") == "USDT" and (m.get("swap") or m.get("contract")):
                    candidates.append((sym, m))
            except Exception:
                continue
        if not candidates:
            raise ValueError(f"Symbol {base_asset}/USDT not found on Binance USDT-M futures.")
        for sym, m in candidates:
            if m.get("linear") is True:
                return sym
        return candidates[0][0]

    symbol = resolve_usdm_symbol(exchange, asset)
    out_file = f"{asset}-PERP_FUNDING.csv"

    since_ms = to_ms(start_dt)
    until_ms = to_ms(end_dt)

    print(f"[START] {asset} funding ({symbol}) from {START_DATE_STR} to {END_DATE_STR}")
    df = fetch_funding_rate_history(exchange, symbol, since_ms, until_ms, batch_limit=1000)

    if df.empty:
        print(f"[WARN] No funding data returned for {asset} (perpetual).")
    else:
        # Ensure numeric type for rate
        df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
        df.to_csv(out_file, index=False)
        print(f"[OK] Wrote {out_file} with {len(df)} rows.")

# ==========================
# Gold via yfinance (Daily)
# ==========================

def fetch_gold_daily(start_dt: datetime, end_dt: datetime, ticker: str = GOLD_TICKER, out_file: str = GOLD_OUTFILE):
    """
    Fetch XAU/USD daily OHLCV via yfinance and write XAU-USD_1D.csv
    - yfinance's 'end' is exclusive; we pass end+1day to include END_DATE_STR.
    - Many FX-like tickers have no volume; we fill missing with 0 to match required columns.
    """
    # yfinance requires strings and end-exclusive behavior
    yf_start = start_dt.strftime("%Y-%m-%d")
    yf_end = (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")

    tried_tickers = []
    data = None
    for t in [ticker, "GC=F", "XAU=X"]:
        if t in tried_tickers:
            continue
        tried_tickers.append(t)
        print(f"[START] Gold daily (yfinance {t}) from {START_DATE_STR} to {END_DATE_STR}")
        data = yf.download(
            t,
            start=yf_start,
            end=yf_end,
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="column",
        )
        if data is not None and not data.empty:
            ticker = t
            break
        else:
            print(f"[INFO] No data for ticker {t}, trying next fallback...")

    if data is None or data.empty:
        print("[WARN] No gold data returned from yfinance for all attempted tickers.")
        empty = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        empty.to_csv(out_file, index=False)
        print(f"[OK] Wrote empty {out_file}.")
        return

    # Normalize column names and shapes
    # yfinance columns usually: ['Open','High','Low','Close','Adj Close','Volume']
    df = data.copy()
    # If a Series was returned, convert to DataFrame named 'close'
    if isinstance(df, pd.Series):
        df = df.to_frame(name="close")
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(level) for level in col if str(level) != ""]).strip() for col in df.columns]
    # Some tickers return timezone-aware index; convert to UTC date -> epoch seconds at 00:00:00 UTC
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)

    # Flexible column discovery: exact or substring matches
    lower_cols = {str(c).lower(): c for c in df.columns}

    def find_col(targets):
        for target in targets:
            # exact lowercase match
            if target in lower_cols:
                return lower_cols[target]
            # substring match across all columns
            for lc_name, orig in lower_cols.items():
                if target in lc_name:
                    return orig
        return None

    col_close = find_col(["close", "adj close", "adj_close"])  # prefer close, fallback to adj close
    if col_close is None:
        # as a last resort, take the first numeric column
        numeric_candidates = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_candidates:
            col_close = numeric_candidates[0]
        else:
            raise ValueError("yfinance data missing required column: close")

    col_open = find_col(["open"])
    col_high = find_col(["high"])
    col_low  = find_col(["low"])
    col_vol  = find_col(["volume", "vol"])

    # Build standardized DataFrame, synthesizing from close when missing
    def to_series(obj):
        if obj is None:
            return None
        s = obj
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return pd.to_numeric(s, errors="coerce")

    close_s = to_series(df[col_close])
    open_s  = to_series(df[col_open]) if col_open is not None else close_s
    high_s  = to_series(df[col_high]) if col_high is not None else close_s
    low_s   = to_series(df[col_low])  if col_low  is not None else close_s

    if col_vol is None:
        volume_series = pd.Series(0, index=df.index)
    else:
        vol_obj = df[col_vol]
        if isinstance(vol_obj, pd.DataFrame):
            volume_series = pd.to_numeric(vol_obj.iloc[:, 0], errors="coerce").fillna(0)
        else:
            volume_series = pd.to_numeric(vol_obj, errors="coerce").fillna(0)

    df = pd.DataFrame({
        "open": open_s,
        "high": high_s,
        "low": low_s,
        "close": close_s,
        "volume": volume_series,
    }, index=df.index)

    # Handle volume possibly being absent or a DataFrame-like object
    if "volume" not in df.columns:
        volume_series = pd.Series(0, index=df.index)
    else:
        vol_obj = df["volume"]
        if isinstance(vol_obj, pd.DataFrame):
            # Take first column if multiple present
            volume_series = pd.to_numeric(vol_obj.iloc[:, 0], errors="coerce").fillna(0)
        else:
            volume_series = pd.to_numeric(vol_obj, errors="coerce").fillna(0)

    df = pd.DataFrame({
        "open": pd.to_numeric(df["open"], errors="coerce"),
        "high": pd.to_numeric(df["high"], errors="coerce"),
        "low": pd.to_numeric(df["low"], errors="coerce"),
        "close": pd.to_numeric(df["close"], errors="coerce"),
        "volume": volume_series,
    }, index=df.index)

    # Create timestamp column (start of day in UTC)
    df["timestamp"] = (df.index.tz_localize(timezone.utc).view("int64") // 10**9).astype(int)

    df = df.reset_index(drop=True)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Clip strictly to end_dt (in case the +1 day brought extra)
    until_s = int(end_dt.replace(tzinfo=timezone.utc).timestamp())
    df = df[df["timestamp"] <= until_s].reset_index(drop=True)

    df.to_csv(out_file, index=False)
    print(f"[OK] Wrote {out_file} with {len(df)} rows.")

# ==========================
# Main
# ==========================

def main():
    start_dt = datetime.strptime(START_DATE_STR, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(END_DATE_STR, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    # Init Binance USDT-M futures
    try:
        ex = init_binance_usdm()
    except Exception as e:
        print("[FATAL] Could not initialize Binance USDT-M exchange:", repr(e))
        return

    # Crypto data for each asset
    for asset in CRYPTO_ASSETS:
        try:
            fetch_price_data(ex, asset, start_dt, end_dt)
        except Exception as e:
            print(f"[ERROR] {asset} OHLCV failed: {type(e).__name__}: {e}")
            traceback.print_exc()

        # short pause between asset types
        safe_sleep(RATE_LIMIT_SLEEP * 2)

        try:
            fetch_funding_data(ex, asset, start_dt, end_dt)
        except Exception as e:
            print(f"[ERROR] {asset} funding failed: {type(e).__name__}: {e}")
            traceback.print_exc()

        # longer pause before next asset
        safe_sleep(RATE_LIMIT_SLEEP * 3)

    # Gold daily
    try:
        fetch_gold_daily(start_dt, end_dt, ticker=GOLD_TICKER, out_file=GOLD_OUTFILE)
    except Exception as e:
        print(f"[ERROR] Gold fetch failed: {type(e).__name__}: {e}")
        traceback.print_exc()

    print("\n[DONE] All tasks attempted. CSVs saved to current directory.")

if __name__ == "__main__":
    main()
