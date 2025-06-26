# mini-simon

## ✅ Final Build Checklist for Going Live

| Step | Module | Status | Seq |
| --- | --- | --- | --- |
| ✅ | Historical Backtest Engine | Done |  |
| ✅ | Volume Spike + Liquidity Sweep | Done |  |
| ✅ | Body Imbalance after Sweep | Done |  |
| ✅ | Multi-Timeframe Bias | Add |  |
| 🔄 | Order Block Detection | Add |  |
| 🔄 | Liquidity Pool Confirmation | Add |  |
| 🔄 | Smart Money Divergence (Optional) | Consider |  |
| 🔄 | Live Signal Engine (Next Step) | Build |  |
| 🔄 | Telegram/Console/Web Output | Plan |  |

### Modules:

- Detect:
    - **Liquidity Sweep** (wick break + volume spike)
    - **Order Block Validity** (price returns to OB after sweep)
    - **Volume Imbalance** (spike vs. average)
    - **Structure Breaks (CHoCH / BOS)**
    - **Premium/Discount Zones** (based on FVG or previous swing)

## 🔍 Here's What You Need on Top of OHLCV:

| Feature | Why It’s Needed |
| --- | --- |
| ✅ **Wick Size / Body Ratio** | Detect strong rejections, manipulation candles |
| ✅ **Swing High / Low Detection** | Key to identifying liquidity levels |
| ✅ **Volume Spike Detection** | Signal for institutional activity |
| ✅ **VWAP / Rolling VWAP** | Fair value zones for smart entries |
| ✅ **Session Labels** (Opening Range, London Open, etc.) | For time-based manipulation patterns |
| ✅ **Previous Candle Bias** | Check displacement candles or imbalance logic |
| ✅ **Order Block Candidates** | Wick-to-body logic + confirmation via mitigation |
| ✅ **Liquidity Pool Zone Markers** | To detect sweeps and inducement zones |
|  |  |