import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime

st.set_page_config(page_title="Revisions & Valuation — XLSX Workflow",
                   page_icon="📈", layout="wide")

# ---------- optional price fetch (guard yfinance) ----------
try:
    import yfinance as yf
    YF_OK = True
except Exception:
    YF_OK = False

# ---------- Exchange → Yahoo suffix helpers ----------
EXCHANGE_SUFFIX = {
    # USA
    "NASDAQ": "", "NYSE": "", "NYQ": "", "NMS": "", "NMQ": "",
    # UK
    "LSE": ".L", "XLON": ".L",
    # Germany
    "XETRA": ".DE", "ETR": ".DE", "FWB": ".F",
    # Netherlands
    "AMS": ".AS", "AEX": ".AS",
    # France
    "EPA": ".PA", "PAR": ".PA",
    # Spain
    "BME": ".MC",
    # Switzerland
    "SWX": ".SW", "SIX": ".SW", "VTX": ".SW",
    # Italy
    "MIL": ".MI", "Borsa Italiana": ".MI",
    # Canada
    "TSX": ".TO", "TSE": ".TO",
    # Australia
    "ASX": ".AX",
    # Hong Kong
    "HKEX": ".HK",
    # Nordics
    "OMX": ".ST", "STO": ".ST",
    "CPH": ".CO", "OSL": ".OL", "HEL": ".HE",
    # Ireland
    "ISE": ".IR",
}

def infer_yf_symbol(symbol: str, exchange: str) -> str:
    if not isinstance(symbol, str) or not symbol.strip():
        return ""
    suff = EXCHANGE_SUFFIX.get(str(exchange).strip(), "")
    return f"{symbol.strip()}{suff}"

@st.cache_data(ttl=600, show_spinner=False)
def fetch_price(yf_symbol: str):
    if not (YF_OK and yf_symbol):
        return None
    try:
        t = yf.Ticker(yf_symbol)
        price = t.fast_info.get("lastPrice")
        if price is None or (isinstance(price, float) and np.isnan(price)):
            hist = t.history(period="5d", interval="1d")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
        return float(price) if price is not None else None
    except Exception:
        return None

def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["EPS_TTM","EPS_Fwd_FY_Old","EPS_Fwd_FY_New",
                "EPS_Next_Q_Old","EPS_Next_Q_New","FiveY_Median_PE","Price"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out["Revision_%"] = (out["EPS_Fwd_FY_New"] - out["EPS_Fwd_FY_Old"]) / out["EPS_Fwd_FY_Old"]
    out["Revision_Next_Q_%"] = (out["EPS_Next_Q_New"] - out["EPS_Next_Q_Old"]) / out["EPS_Next_Q_Old"]
    out["Forward_PE"] = out["Price"] / out["EPS_Fwd_FY_New"]
    out["PE_Deviation"] = out["Forward_PE"] - out["FiveY_Median_PE"]
    out["PE_Discount_%"] = (out["FiveY_Median_PE"] - out["Forward_PE"]) / out["FiveY_Median_PE"]

    def flag_row(r):
        try:
            if pd.notna(r["Revision_%"]) and pd.notna(r["Forward_PE"]) and pd.notna(r["FiveY_Median_PE"]):
                if (r["Revision_%"] > 0) and (r["Forward_PE"] <= 0.95 * r["FiveY_Median_PE"]):
                    return "Attractive"
                if (r["Revision_%"] < 0) or (r["Forward_PE"] > 1.10 * r["FiveY_Median_PE"]):
                    return "Caution"
                return "Neutral"
            return ""
        except Exception:
            return ""
    out["Flag"] = out.apply(flag_row, axis=1)
    return out

def to_excel_bytes(df_dict: dict) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for sheet, data in df_dict.items():
            (data if isinstance(data, pd.DataFrame) else pd.DataFrame()
             ).to_excel(writer, index=False, sheet_name=sheet[:31])
    return output.getvalue()

# ---------- UI ----------
st.title("📈 Revisions & Valuation — Upload XLSX and Analyze")
st.success("App loaded ✔️  Upload your XLSX to begin.")
st.write("Expected columns: **Symbol, Exchange, Name, Sector, Industry, Country, Asset_Type, Notes**. "
         "App derives **YF_Symbol**; you fill **EPS forward** & **5Y median P/E**; then it computes "
         "**Revision %**, **Forward P/E**, **PE discount** and a **Flag**.")

with st.sidebar:
    st.header("Options")
    fetch_prices = st.checkbox("Fetch live prices (yfinance)", value=False,
                               help="Keep off on first run to avoid slow external calls.")
    if not YF_OK:
        st.caption("yfinance not available; price fetch disabled.")
    only_attractive = st.checkbox("Show only Attractive", value=False)
    sort_by = st.selectbox("Sort by",
                           ["Revision_% desc", "PE_Discount_% desc", "Forward_PE asc", "Symbol asc"], index=0)
    add_timestamp = st.checkbox("Timestamp on export filename", value=True)

up = st.file_uploader("Upload your portfolio XLSX", type=["xlsx"])
if up is None:
    st.info("Upload an .xlsx to continue.")
    st.stop()

# ---------- Read XLSX ----------
try:
    base = pd.read_excel(up)
except Exception as e:
    st.error(f"Failed to read XLSX: {e}")
    st.stop()

required_cols = ["Symbol","Exchange","Name","Sector","Industry","Country","Asset_Type","Notes"]
missing = [c for c in required_cols if c not in base.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# Derive YF_Symbol and add analysis columns if needed
base["YF_Symbol"] = [infer_yf_symbol(s, ex) for s, ex in zip(base["Symbol"], base["Exchange"])]
for col in ["EPS_TTM","EPS_Fwd_FY_Old","EPS_Fwd_FY_New",
            "EPS_Next_Q_Old","EPS_Next_Q_New","FiveY_Median_PE","Price"]:
    if col not in base.columns:
        base[col] = np.nan

st.subheader("Step 1 — Review mapping & fill EPS/anchors")
st.caption("Check **YF_Symbol** (override if needed, e.g., 2B76.DE / RBOT.L). Then fill **EPS_Fwd_FY_Old/New** and **FiveY_Median_PE**.")
edited = st.data_editor(
    base,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Symbol": st.column_config.TextColumn(disabled=True),
        "Exchange": st.column_config.TextColumn(disabled=True),
        "Name": st.column_config.TextColumn(disabled=True),
        "Sector": st.column_config.TextColumn(disabled=True),
        "Industry": st.column_config.TextColumn(disabled=True),
        "Country": st.column_config.TextColumn(disabled=True),
        "Asset_Type": st.column_config.TextColumn(disabled=True),
    },
)

# Fetch prices only if requested & yfinance available
if fetch_prices and YF_OK:
    with st.status("Fetching prices from yfinance...", expanded=False):
        prices = []
        for yf_sym, cur_p in zip(edited["YF_Symbol"].astype(str), edited["Price"]):
            if (pd.isna(cur_p) or cur_p == "") and isinstance(yf_sym, str) and yf_sym.strip():
                prices.append(fetch_price(yf_sym.strip()))
            else:
                try:
                    prices.append(float(cur_p))
                except Exception:
                    prices.append(np.nan)
        edited["Price"] = prices

# ---------- Metrics & view ----------
signals = compute_metrics(edited)

if sort_by == "Revision_% desc":
    signals = signals.sort_values("Revision_%", ascending=False, na_position="last")
elif sort_by == "PE_Discount_% desc":
    signals = signals.sort_values("PE_Discount_%", ascending=False, na_position="last")
elif sort_by == "Forward_PE asc":
    signals = signals.sort_values("Forward_PE", ascending=True, na_position="last")
else:
    signals = signals.sort_values("Symbol", ascending=True, na_position="last")

if only_attractive:
    signals = signals[signals["Flag"] == "Attractive"]

st.subheader("Step 2 — Signals")
st.caption("**Attractive** = Revision_% > 0 & Forward_PE ≤ 95% of 5Y median.  "
           "**Caution** = Revision_% < 0 or Forward_PE > 110% of 5Y median.")
st.dataframe(signals, use_container_width=True, height=420)

st.subheader("Step 3 — Snapshot")
if not signals.empty:
    top = signals.head(10).copy()
    good = int((signals["Flag"] == "Attractive").sum())
    st.write(f"**Attractive candidates:** {good} / {len(signals)}")
    st.dataframe(top[["Symbol","Name","Revision_%","Forward_PE",
                      "FiveY_Median_PE","PE_Discount_%","Flag"]],
                 use_container_width=True)

# ---------- Export ----------
st.subheader("Step 4 — Export Excel")
ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S") if add_timestamp else ""
fname = f"Revisions_Valuation_withSignals_{ts}.xlsx" if ts else "Revisions_Valuation_withSignals.xlsx"
def to_excel_bytes(df_dict: dict) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for sheet, data in df_dict.items():
            (data if isinstance(data, pd.DataFrame) else pd.DataFrame()
             ).to_excel(writer, index=False, sheet_name=sheet[:31])
    return output.getvalue()
xls = to_excel_bytes({"Signals": signals, "Inputs": edited})
st.download_button("📥 Download workbook", data=xls, file_name=fname,
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown("---")
st.caption("Tip: keep your XLSX as the source of truth. Update EPS forward after earnings/guidance; the app flags changes.")
