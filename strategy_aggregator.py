import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# =============================================================
# Configuration
# =============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
MODULE_ROOT = PROJECT_ROOT / "module"
OUTPUT_DIR = PROJECT_ROOT / "module"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Strategy directory name -> canonical key and weight
STRATEGY_MAP: Dict[str, Dict[str, object]] = {
    "LiquiditySweep": {"key": "vol_spike", "weight": 0.35},
    "BodyImbalance": {"key": "body_imbalance", "weight": 0.25},
    "order block": {"key": "order_block", "weight": 0.25},
    "stock_burner_backtest": {"key": "stock_burner", "weight": 0.15},
}

# Timeframe folders to minutes / label and HTF/LTF classification
TIMEFRAME_MAP: Dict[str, Dict[str, object]] = {
    "3": {"label": "3m", "minutes": 3, "class": "LTF"},
    "5": {"label": "5m", "minutes": 5, "class": "LTF"},
    "15": {"label": "15m", "minutes": 15, "class": "LTF"},
    "60": {"label": "60m", "minutes": 60, "class": "HTF"},
    "120": {"label": "120m", "minutes": 120, "class": "HTF"},
    "180": {"label": "180m", "minutes": 180, "class": "HTF"},
    "240": {"label": "240m", "minutes": 240, "class": "HTF"},
    "day": {"label": "1D", "minutes": 60 * 24, "class": "HTF"},
    "D": {"label": "1D", "minutes": 60 * 24, "class": "HTF"},
}

# Timeframe weight by label (directional strength)
TIMEFRAME_WEIGHT: Dict[str, float] = {
    "3m": 0.5,
    "5m": 0.6,
    "15m": 0.7,
    "60m": 1.0,
    "120m": 1.2,
    "180m": 1.3,
    "240m": 1.4,
    "1D": 1.6,
}

# NIFTY 50 base symbols
NIFTY_50_BASE: List[str] = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "KOTAKBANK", "LT", "ITC", "SBIN", "HINDUNILVR",
    "AXISBANK", "BAJFINANCE", "ASIANPAINT", "MARUTI", "SUNPHARMA", "TITAN", "WIPRO", "ULTRACEMCO", "NESTLEIND",
    "POWERGRID", "BAJAJFINSV", "TECHM", "NTPC", "GRASIM", "JSWSTEEL", "HCLTECH", "TATAMOTORS", "DRREDDY", "CIPLA",
    "ONGC", "HDFCLIFE", "DIVISLAB", "HEROMOTOCO", "BRITANNIA", "BPCL", "COALINDIA", "ADANIENT", "ADANIPORTS",
    "INDUSINDBK", "BAJAJ-AUTO", "SHREECEM", "SBILIFE", "EICHERMOT", "TATACONSUM", "HINDALCO", "APOLLOHOSP",
    "ICICIPRULI", "TATASTEEL", "M&M", "BHARTIARTL",
]
NIFTY_50_SET = set(NIFTY_50_BASE)


# =============================================================
# Utility functions
# =============================================================

def normalise_symbol(raw: str) -> str:
    if not isinstance(raw, str):
        return ""
    s = raw.strip().upper()
    if ":" in s:
        s = s.split(":", 1)[1]
    if s.endswith("-EQ"):
        s = s[:-3]
    return s.strip()


def classify_strategy(path: Path) -> Tuple[str, float]:
    for dir_name, cfg in STRATEGY_MAP.items():
        if dir_name.lower() in str(path).lower().replace("\\", "/"):
            return cfg["key"], float(cfg["weight"])
    return "unknown", 0.0


def infer_timeframe(path: Path) -> Tuple[str, int, str, float]:
    parts = [p.name for p in path.parents]
    for part in parts:
        if part in TIMEFRAME_MAP:
            meta = TIMEFRAME_MAP[part]
            label = str(meta["label"])
            minutes = int(meta["minutes"])
            tf_class = str(meta["class"])
            tf_weight = TIMEFRAME_WEIGHT.get(label, 1.0)
            return label, minutes, tf_class, tf_weight
    # Fallback from filename like XYZ-15-LiquiditySweep.csv
    name = path.name
    for key, meta in TIMEFRAME_MAP.items():
        if f"-{key}-" in name:
            label = str(meta["label"])
            minutes = int(meta["minutes"])
            tf_class = str(meta["class"])
            tf_weight = TIMEFRAME_WEIGHT.get(label, 1.0)
            return label, minutes, tf_class, tf_weight
    return "UNKNOWN", 0, "UNKNOWN", 1.0


def map_signal_to_value(sig: str) -> int:
    if not isinstance(sig, str):
        return 0
    s = sig.strip().upper()
    if s in {"BUY", "LONG"}:
        return 1
    if s in {"SELL", "SHORT"}:
        return -1
    return 0


# =============================================================
# Loading and validation
# =============================================================

def discover_csv_files() -> List[Path]:
    if not MODULE_ROOT.exists():
        return []
    files: List[Path] = []
    for root, _, filenames in os.walk(MODULE_ROOT):
        for fn in filenames:
            if fn.lower().endswith((".csv", ".xlsx")):
                files.append(Path(root) / fn)
    return files


def load_and_annotate(csv_path: Path, audit: Dict) -> pd.DataFrame:
    strategy_key, strategy_weight = classify_strategy(csv_path)
    tf_label, tf_minutes, tf_class, tf_weight = infer_timeframe(csv_path)

    try:
        if csv_path.suffix.lower() == ".xlsx":
            df = pd.read_excel(csv_path)
        else:
            df = pd.read_csv(csv_path)
    except Exception as e:
        audit.setdefault("load_errors", []).append({
            "file": str(csv_path),
            "error": str(e),
        })
        return pd.DataFrame()

    # Normalise column names to expected lower-case form
    df.columns = [c.strip().lower() for c in df.columns]

    # Map expected columns from known patterns
    col_map = {}
    # Direct matches
    for target in ["symbol", "timestamp", "price", "signal", "score", "volume"]:
        if target in df.columns:
            col_map[target] = target
            continue
        # fallback heuristics
        if target == "symbol":
            # try to infer symbol from filename
            inferred = csv_path.stem.split("-")[0].upper()
            df["symbol"] = inferred
            col_map["symbol"] = "symbol"
        elif target == "timestamp":
            for cand in ["datetime", "entry time", "date"]:
                if cand in df.columns:
                    col_map[cand] = target
                    break
        elif target == "price":
            for cand in ["entry price"]:
                if cand in df.columns:
                    col_map[cand] = target
                    break
        elif target == "signal":
            for cand in ["direction"]:
                if cand in df.columns:
                    col_map[cand] = target
                    break
        elif target == "score":
            # If score column missing, default to 1.0 per row
            df["score"] = 1.0
            col_map["score"] = "score"
        elif target == "volume":
            # If volume column missing, default to 1
            df["volume"] = 1
            col_map["volume"] = "volume"

    # Rename columns to canonical names
    rename_dict = {v: k for k, v in col_map.items()}
    df = df.rename(columns=rename_dict)

    # Final fill for any still-missing columns (after rename)
    if "symbol" not in df.columns:
        inferred = csv_path.stem.split("-")[0].upper()
        df["symbol"] = inferred
    if "timestamp" not in df.columns:
        # try to create a fake timestamp from entry time if present
        if "entry time" in df.columns:
            df["timestamp"] = pd.to_datetime(df["entry time"], errors="coerce")
        else:
            df["timestamp"] = pd.NaT
    if "price" not in df.columns:
        if "entry price" in df.columns:
            df["price"] = pd.to_numeric(df["entry price"], errors="coerce")
        else:
            df["price"] = pd.NA
    if "signal" not in df.columns:
        if "direction" in df.columns:
            df["signal"] = df["direction"]
        else:
            df["signal"] = "HOLD"
    if "score" not in df.columns:
        df["score"] = 1.0
    if "volume" not in df.columns:
        df["volume"] = 1

    expected_cols = {"symbol", "timestamp", "price", "signal", "score", "volume"}
    missing = expected_cols - set(df.columns)
    if missing:
        audit.setdefault("schema_errors", []).append({
            "file": str(csv_path),
            "missing_columns": sorted(list(missing)),
        })
        return pd.DataFrame()

    df = df.copy()
    df["symbol_raw"] = df["symbol"]
    df["symbol"] = df["symbol"].apply(normalise_symbol)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df["strategy"] = strategy_key
    df["strategy_weight"] = strategy_weight
    df["timeframe"] = tf_label
    df["timeframe_minutes"] = tf_minutes
    df["timeframe_class"] = tf_class
    df["timeframe_weight"] = tf_weight
    df["source_file"] = str(csv_path)

    return df


def validate_data(df: pd.DataFrame, audit: Dict) -> pd.DataFrame:
    if df.empty:
        return df

    total_rows = len(df)
    audit_local = {
        "total_rows": int(total_rows),
        "rejected_invalid_symbol": 0,
        "rejected_invalid_timestamp": 0,
        "rejected_missing_fields": 0,
        "rejected_price_volume": 0,
        "rejected_price_jumps": 0,
    }

    mask_valid_symbol = df["symbol"].isin(NIFTY_50_SET)
    audit_local["rejected_invalid_symbol"] = int((~mask_valid_symbol).sum())

    mask_valid_ts = df["timestamp"].notna()
    audit_local["rejected_invalid_timestamp"] = int((~mask_valid_ts).sum())

    mask_fields = df[["signal", "score", "price", "volume"]].notna().all(axis=1)
    audit_local["rejected_missing_fields"] = int((~mask_fields).sum())

    mask_price_vol = (df["price"] > 0) & (df["volume"] > 0)
    audit_local["rejected_price_volume"] = int((~mask_price_vol).sum())

    mask_basic = mask_valid_symbol & mask_valid_ts & mask_fields & mask_price_vol
    df_valid = df[mask_basic].copy()

    df_valid = df_valid.sort_values(["symbol", "timeframe", "timestamp"]).reset_index(drop=True)
    mask_jump = pd.Series(False, index=df_valid.index)
    for (sym, tf), grp in df_valid.groupby(["symbol", "timeframe"], sort=False):
        if len(grp) < 2:
            continue
        prev_price = grp["price"].shift(1)
        rel_change = (grp["price"] / prev_price - 1).abs()
        jump_idx = grp.index[rel_change > 0.20]
        mask_jump.loc[jump_idx] = True

    audit_local["rejected_price_jumps"] = int(mask_jump.sum())

    df_valid = df_valid[~mask_jump].copy()

    key = "global"
    agg = audit.setdefault("validation", {}).setdefault(key, {})
    for k, v in audit_local.items():
        agg[k] = agg.get(k, 0) + int(v)

    return df_valid


# =============================================================
# Aggregation and decision logic
# =============================================================

def aggregate_signals(df: pd.DataFrame, audit: Dict) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[
            "symbol", "master_signal", "conviction_score", "strategies_triggered", "timeframes_triggered",
            "structural_reason", "entry_price", "stop_loss_level", "target_level", "signal_timestamp",
            "signal_timeframe", "chart_anchor_price", "timestamp_generated"
        ])

    df = df.copy()
    df["signal_value"] = df["signal"].apply(map_signal_to_value)
    df["score"] = pd.to_numeric(df["score"], errors="coerce").clip(lower=0.0, upper=1.0).fillna(0.0)

    df["row_weight"] = df["signal_value"] * df["score"] * df["strategy_weight"] * df["timeframe_weight"]

    df["session_date"] = df["timestamp"].dt.date

    summary_rows: List[Dict[str, object]] = []

    for symbol, sym_df in df.groupby("symbol", sort=False):
        sym_df = sym_df.sort_values("timestamp")
        last_date = sym_df["session_date"].max()
        day_df = sym_df[sym_df["session_date"] == last_date]
        if day_df.empty:
            continue

        htf_df = day_df[day_df["timeframe_class"] == "HTF"]
        ltf_df = day_df[day_df["timeframe_class"] == "LTF"]

        def dir_from_df(sub: pd.DataFrame) -> Tuple[str, float, float, float]:
            if sub.empty:
                return "NEUTRAL", 0.0, 0.0, 0.0
            bullish = sub.loc[sub["signal_value"] > 0, "row_weight"].sum()
            bearish = -sub.loc[sub["signal_value"] < 0, "row_weight"].sum()
            total = bullish + bearish
            if total <= 0:
                return "NEUTRAL", 0.0, bullish, bearish
            net = bullish - bearish
            strength = abs(net) / total
            if net > 0:
                return "BUY", strength, bullish, bearish
            elif net < 0:
                return "SELL", strength, bullish, bearish
            else:
                return "NEUTRAL", 0.0, bullish, bearish

        htf_dir, htf_strength, htf_bull, htf_bear = dir_from_df(htf_df)
        ltf_dir, ltf_strength, _, _ = dir_from_df(ltf_df)

        # Strategies and timeframes that actually triggered (non-zero weight)
        strat_weights = day_df.groupby("strategy")["row_weight"].sum().to_dict()
        triggered_strats = [s for s, w in strat_weights.items() if w != 0 and s != "unknown"]

        tf_weights = day_df.groupby("timeframe")["row_weight"].sum().to_dict()
        triggered_tfs = sorted([tf for tf, w in tf_weights.items() if w != 0])

        # Decision: BUY / SELL / NO-TRADE
        if htf_dir in {"BUY", "SELL"} and htf_strength >= 0.35 and len(triggered_strats) >= 1:
            if htf_dir == "BUY" and ltf_dir != "SELL":
                master_signal = "BUY"
            elif htf_dir == "SELL" and ltf_dir != "BUY":
                master_signal = "SELL"
            else:
                master_signal = "NO-TRADE"
        else:
            master_signal = "NO-TRADE"

        # Conviction score (0–100)
        conviction = round(htf_strength * 100, 0)

        # Structural reason: short, surgical
        reason_parts = []
        if master_signal == "BUY":
            if "vol_spike" in triggered_strats:
                reason_parts.append("Liquidity sweep")
            if "body_imbalance" in triggered_strats:
                reason_parts.append("bullish imbalance")
            if "order_block" in triggered_strats:
                reason_parts.append("OB retest")
            if "stock_burner" in triggered_strats:
                reason_parts.append("trend pullback")
        elif master_signal == "SELL":
            if "vol_spike" in triggered_strats:
                reason_parts.append("Liquidity sweep")
            if "body_imbalance" in triggered_strats:
                reason_parts.append("bearish imbalance")
            if "order_block" in triggered_strats:
                reason_parts.append("OB break")
            if "stock_burner" in triggered_strats:
                reason_parts.append("trend exhaustion")
        structural_reason = " + ".join(reason_parts) if reason_parts else "Weak alignment"

        # Entry price: latest valid price (prefer LTF, fallback HTF)
        entry_df = ltf_df if not ltf_df.empty else htf_df
        entry_price = entry_df.sort_values("timestamp").iloc[-1]["price"] if not entry_df.empty else None

        # Stop loss / target placeholders (since we lack structural data)
        stop_loss_level = None
        target_level = None
        if master_signal == "BUY" and entry_price is not None:
            stop_loss_level = round(entry_price * 0.985, 2)  # crude 1.5% SL
            target_level = round(entry_price * 1.015, 2)    # crude 1.5% target
        elif master_signal == "SELL" and entry_price is not None:
            stop_loss_level = round(entry_price * 1.015, 2)
            target_level = round(entry_price * 0.985, 2)

        # Chart verification fields
        signal_timestamp = None
        signal_timeframe = None
        chart_anchor_price = None
        if not day_df.empty:
            # Pick row with highest absolute row_weight (most influential)
            anchor_row = day_df.loc[day_df["row_weight"].abs().idxmax()]
            signal_timestamp = pd.to_datetime(anchor_row["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
            signal_timeframe = anchor_row["timeframe"]
            chart_anchor_price = anchor_row["price"]

        # Timestamp generated
        timestamp_generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        summary_rows.append({
            "symbol": symbol,
            "master_signal": master_signal,
            "conviction_score": conviction,
            "strategies_triggered": f"{len(triggered_strats)} | {', '.join(triggered_strats)}",
            "timeframes_triggered": f"{len(triggered_tfs)} | {', '.join(triggered_tfs)}",
            "structural_reason": structural_reason,
            "entry_price": entry_price,
            "stop_loss_level": stop_loss_level,
            "target_level": target_level,
            "signal_timestamp": signal_timestamp,
            "signal_timeframe": signal_timeframe,
            "chart_anchor_price": chart_anchor_price,
            "timestamp_generated": timestamp_generated,
        })

    audit.setdefault("output", {})["total_symbols"] = len(summary_rows)

    return pd.DataFrame(summary_rows)


# =============================================================
# Orchestration
# =============================================================

def run_aggregation():
    audit: Dict[str, object] = {
        "start_time": datetime.now().isoformat(),
        "input_files": {},
        "output": {},
    }

    csv_files = discover_csv_files()
    audit["input_files"]["count"] = len(csv_files)
    audit["input_files"]["paths"] = [str(p) for p in csv_files]

    all_rows: List[pd.DataFrame] = []
    for p in csv_files:
        df_raw = load_and_annotate(p, audit)
        if df_raw.empty:
            continue
        audit_key = classify_strategy(p)[0] or "unknown"
        vdf = validate_data(df_raw, audit)
        if vdf.empty:
            continue
        audit.setdefault("processing", {})[audit_key] = f"Processed {len(vdf)} rows"
        all_rows.append(vdf)

    if all_rows:
        big_df = pd.concat(all_rows, ignore_index=True)
    else:
        big_df = pd.DataFrame()

    final_df = aggregate_signals(big_df, audit)

    csv_path = OUTPUT_DIR / f"final_scan_execution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    final_df.to_csv(csv_path, index=False)

    audit.setdefault("output", {})["csv"] = str(csv_path)
    audit["end_time"] = datetime.now().isoformat()
    audit_path = OUTPUT_DIR / "final_scan_audit.json"
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2)

    print(f" Done. CSV written to: {csv_path}")
    print(f"Audit log: {audit_path}")


if __name__ == "__main__":
    run_aggregation()
