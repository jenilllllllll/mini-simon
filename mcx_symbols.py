import logging
from datetime import datetime, date
from typing import Any, Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

MCX_COM_URL = "https://public.fyers.in/sym_details/MCX_COM.csv"

_COMMODITY_BASE_UNDERLYING: Dict[str, str] = {
    "GOLD": "GOLD",
    "SILVER": "SILVER",
    "CRUDEOIL": "CRUDEOIL",
    "COPPER": "COPPER",
    "NATURALGAS": "NATURALGAS",
    "ZINC": "ZINC",
    "LEAD": "LEAD",
    "ALUMINIUM": "ALUMINIUM",
    "NICKEL": "NICKEL",
}


def _is_standard_contract_symbol(base_name: str, symbol_code: str) -> bool:
    bn = (base_name or "").upper()
    sym = str(symbol_code or "").upper()
    if not sym.startswith("MCX:"):
        return False

    # Exclude mini/micro variants. This project only wants standard contracts.
    if bn == "GOLD":
        return "GOLDM" not in sym and "GOLDGUINEA" not in sym
    if bn == "SILVER":
        return "SILVERMIC" not in sym and "SILVERM" not in sym
    if bn == "CRUDEOIL":
        return "CRUDEOILM" not in sym

    return True


def _load_mcx_symbol_master() -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(MCX_COM_URL, header=None)
    except Exception as e:
        logger.error("Error loading MCX symbol master from %s: %s", MCX_COM_URL, e)
        return None

    if df is None or df.empty:
        return None
    return df


def get_mcx_contract_meta(symbol: str) -> Tuple[Optional[str], Optional[date]]:
    """Return (symbol, expiry_date) for a full MCX instrument like 'MCX:CRUDEOIL26MARFUT'."""

    sym = str(symbol or "").strip()
    if not sym:
        return None, None

    df = _load_mcx_symbol_master()
    if df is None:
        return sym, None

    if 9 not in df.columns or 8 not in df.columns:
        return sym, None

    try:
        rows = df[df[9] == sym]
    except Exception:
        return sym, None

    if rows is None or rows.empty:
        return sym, None

    raw_exp = rows.iloc[0][8]
    return sym, _parse_expiry_from_name(raw_exp)


def _parse_expiry_from_name(raw: Any) -> Optional[date]:
    try:
        if raw is None:
            return None

        if isinstance(raw, (int, float)):
            if raw <= 0:
                return None
            return datetime.fromtimestamp(float(raw)).date()

        s = str(raw).strip()
        if not s:
            return None

        if s.isdigit():
            val = float(s)
            if val <= 0:
                return None
            return datetime.fromtimestamp(val).date()

        parts = s.split()
        if len(parts) >= 4:
            day, mon, yy = parts[-4], parts[-3], parts[-2]
            return datetime.strptime(f"{day} {mon} {yy}", "%d %b %y").date()

        return None
    except Exception:
        return None


def get_current_commodity_symbol(base_name: str) -> Optional[str]:
    bn = (base_name or "").upper()
    underlying = _COMMODITY_BASE_UNDERLYING.get(bn)
    if underlying is None:
        return None

    df = _load_mcx_symbol_master()
    if df is None:
        return None

    try:
        df = df[df[13] == underlying]
    except Exception:
        return None

    if df.empty:
        return None

    # Filter out mini/micro variants for the contracts we care about.
    try:
        if 9 in df.columns:
            df = df[df[9].apply(lambda s: _is_standard_contract_symbol(bn, s))]
    except Exception:
        pass

    if df.empty:
        return None

    try:
        if 16 in df.columns:
            df = df[df[16] == "XX"]
    except Exception:
        pass

    if df.empty:
        return None

    if 8 not in df.columns:
        logger.warning("MCX_COM.csv structure unexpected; missing expiry timestamp column")
        return None

    df = df.copy()
    df["expiry_date"] = df[8].apply(_parse_expiry_from_name)
    df = df[df["expiry_date"].notnull()]
    if df.empty:
        return None

    today = date.today()
    df = df[df["expiry_date"] > today]
    if df.empty:
        return None

    df_sorted = df.sort_values("expiry_date")
    current_row = df_sorted.iloc[0]
    current_expiry = current_row["expiry_date"]
    chosen_row = current_row

    delta_days = (current_expiry - today).days
    if delta_days <= 2:
        later = df_sorted[df_sorted["expiry_date"] > current_expiry]
        if not later.empty:
            chosen_row = later.iloc[0]

    symbol = str(chosen_row[9])
    if not (symbol.startswith("MCX:") and symbol.endswith("FUT")):
        logger.warning("Resolved MCX symbol %s for %s has unexpected format", symbol, bn)
    return symbol


def build_commodity_symbol_mapping() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for base in _COMMODITY_BASE_UNDERLYING.keys():
        sym = get_current_commodity_symbol(base)
        if sym:
            mapping[base] = sym
    return mapping
