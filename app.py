# Update the Streamlit app to accept the user's XLSX schema and map to analysis fields.
import os, zipfile

project_dir = "/mnt/data/revisions_app_v2"
os.makedirs(project_dir, exist_ok=True)
os.makedirs(os.path.join(project_dir, ".streamlit"), exist_ok=True)

app_py = r'''
import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime
import yfinance as yf

st.set_page_config(page_title="Revisions & Valuation — XLSX Workflow", page_icon="📈", layout="wide")

# ---------------------
# Utilities
# ---------------------
EXCHANGE_SUFFIX = {
    # Common mappings for Yahoo Finance tickers
    "NASDAQ": "", "NYQ": "", "NYSE": "", "NMS": "", "NMQ": "",
    "LSE": ".L", "XLON": ".L",
    "XETRA": ".DE", "ETR": ".DE",
    "FWB": ".F",
    "AMS": ".AS", "AEX": ".AS", "EN AMS": ".AS",
    "EPA": ".PA", "PAR": ".PA",
    "BME": ".MC",
    "SWX": ".SW",
    "VTX": ".SW",
    "Borsa Italiana": ".MI", "MIL": ".MI",
    "TSX": ".TO", "TSE": ".TO",
    "ASX": ".AX",
    "HKEX": ".HK",
    "JSE": ".JO",
    "SIX": ".SW",
    "ISE": ".IR",
    "OMX": ".ST", "STO": ".ST",
    "CPH": ".CO",
    "OSL": ".OL",
    "HEL": ".HE",
}

def infer_yf_symbol(symbol: str, exchange: str) -> str:
    if not isinstance(symbol, str) or len(symbol.strip()) == 0:
        return ""
    suff = EXCHANGE_SUFFIX.get(str(exchange).strip(), "")
    return f"{symbol.strip()}{suff}"

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price(yf_symbol: str):
    if not yf_symbol:
        return None
    try:
        t = yf.Ticker(yf_symbol)
        price = t.fast_info.get("lastPrice")
        if price is None or np.isnan(price):
            hist = t.history(period="5d", interval="1d")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
        return float(price) if price is not None else None
    except Exception:
        return None

def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["EPS_TTM","EPS_Fwd_FY_Old","EPS_Fwd_FY_New","EPS_Next_Q_Old","EPS_Next_Q_New","FiveY_Median_PE","Price"]:
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
            # Guard against empty frames
            if isinstance(data, pd.DataFrame) and not data.empty:
                data.to_excel(writer, index=False, sheet_name=sheet[:31])
            else:
                pd.DataFrame().to_excel(writer, index=False, sheet_name=sheet[:31])
    return output.getvalue()

# ---------------------
# Sidebar
# ---------------------
with st.sidebar:
    st.title("📈 Revisions & Valuation (XLSX)")
    st.caption("Upload your **master XLSX** then enrich with EPS forward & anchors.")
    st.markdown("---")
    fetch_prices = st.checkbox("Fetch live prices from Yahoo (yfinance)", value=True)
    st.markdown("---")
    only_attractive = st.checkbox("Show only Attractive", value=False)
    sort_by = st.selectbox("Sort by", ["Revision_% desc", "PE_Discount_% desc", "Forward_PE asc", "Symbol asc"], index=0)
    st.markdown("---")
    add_timestamp = st.checkbox("Timestamp on export filename", value=True)

st.title("Revisions & Valuation — Upload XLSX and Analyze")
st.write("Expected columns in your upload: **Symbol, Exchange, Name, Sector, Industry, Country, Asset_Type, Notes**. The app will derive **YF_Symbol**, let you fill EPS forward & 5Y median P/E, then compute **Revision %**, **Forward P/E**, and a **Flag**.")

up = st.file_uploader("Upload your portfolio XLSX", type=["xlsx"])

if up is None:
    st.info("Please upload an .xlsx with the columns above to continue.")
    st.stop()

# Read xlsx
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

# Derive YF_Symbol
base["YF_Symbol"] = [infer_yf_symbol(s, ex) for s, ex in zip(base["Symbol"], base["Exchange"])]

# Add analysis columns if not present
for col in ["EPS_TTM","EPS_Fwd_FY_Old","EPS_Fwd_FY_New","EPS_Next_Q_Old","EPS_Next_Q_New","FiveY_Median_PE","Price"]:
    if col not in base.columns:
        base[col] = np.nan

st.subheader("Step 1 — Review mapping & fill EPS/anchors")
st.caption("Check **YF_Symbol** mapping; override if needed (e.g., 2B76.DE, RBOT.L). Then fill **EPS_Fwd_FY_Old/New** and **FiveY_Median_PE**.")
editable_cols = ["YF_Symbol","EPS_TTM","EPS_Fwd_FY_Old","EPS_Fwd_FY_New","EPS_Next_Q_Old","EPS_Next_Q_New","FiveY_Median_PE","Price","Notes"]
edited = st.data_editor(base, num_rows="dynamic", use_container_width=True, column_config={
    "Symbol": st.column_config.TextColumn(disabled=True),
    "Exchange": st.column_config.TextColumn(disabled=True),
    "Name": st.column_config.TextColumn(disabled=True),
    "Sector": st.column_config.TextColumn(disabled=True),
    "Industry": st.column_config.TextColumn(disabled=True),
    "Country": st.column_config.TextColumn(disabled=True),
    "Asset_Type": st.column_config.TextColumn(disabled=True),
})

# Fetch prices if requested and missing
if fetch_prices:
    with st.status("Fetching prices from yfinance...", expanded=False):
        prices = []
        for yf_sym, cur_p in zip(edited["YF_Symbol"].astype(str), edited["Price"]):
            if (pd.isna(cur_p) or cur_p == "") and isinstance(yf_sym, str) and len(yf_sym.strip()) > 0:
                prices.append(fetch_price(yf_sym.strip()))
            else:
                try:
                    prices.append(float(cur_p))
                except Exception:
                    prices.append(np.nan)
        edited["Price"] = prices

# Compute metrics
signals = compute_metrics(edited)

# Sort
if sort_by == "Revision_% desc":
    signals = signals.sort_values("Revision_%", ascending=False, na_position="last")
elif sort_by == "PE_Discount_% desc":
    signals = signals.sort_values("PE_Discount_%", ascending=False, na_position="last")
elif sort_by == "Forward_PE asc":
    signals = signals.sort_values("Forward_PE", ascending=True, na_position="last")
else:
    signals = signals.sort_values("Symbol", ascending=True, na_position="last")

# Filter Attractive if requested
if only_attractive:
    signals = signals[signals["Flag"] == "Attractive"]

st.subheader("Step 2 — Signals")
st.caption("**Attractive** = Revision_% > 0 and Forward_PE ≤ 95% of 5Y median.  **Caution** = Revision_% < 0 or Forward_PE > 110% of 5Y median.")
st.dataframe(signals, use_container_width=True, height=420)

# Quick metrics
st.subheader("Step 3 — Snapshot")
if not signals.empty:
    top = signals.head(10).copy()
    good = (signals["Flag"]=="Attractive").sum()
    st.write(f"**Attractive candidates:** {good} / {len(signals)}")
    st.write("Top by Revision_%:")
    st.dataframe(top[["Symbol","Name","Revision_%","Forward_PE","FiveY_Median_PE","PE_Discount_%","Flag"]], use_container_width=True)

# Export
st.subheader("Step 4 — Export Excel")
ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S") if add_timestamp else ""
fname = f"Revisions_Valuation_withSignals_{ts}.xlsx" if ts else "Revisions_Valuation_withSignals.xlsx"
xls = to_excel_bytes({"Signals": signals, "Inputs": edited})
st.download_button("📥 Download workbook", data=xls, file_name=fname, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown("---")
st.caption("Tip: Maintain your master list with metadata (Sector/Industry/Country/Theme). Update EPS forward when earnings/guidance change; the app does the rest.")
'''

requirements_txt = """
streamlit==1.37.1
pandas==2.2.2
yfinance==0.2.52
numpy==1.26.4
xlsxwriter==3.2.0
"""

config_toml = """
[theme]
base="dark"
primaryColor="#7bdcb5"
backgroundColor="#0e1117"
secondaryBackgroundColor="#1b1f2a"
textColor="#e6e6e6"
"""

# Sample XLSX with requested columns
import pandas as pd
sample_df = pd.DataFrame([
    {"Symbol":"NVDA","Exchange":"NASDAQ","Name":"NVIDIA Corp","Sector":"Information Technology","Industry":"Semiconductors","Country":"USA","Asset_Type":"Equity","Notes":"Add EPS forward"},
    {"Symbol":"ASML","Exchange":"AMS","Name":"ASML Holding NV","Sector":"Information Technology","Industry":"Semiconductor Equipment","Country":"Netherlands","Asset_Type":"Equity","Notes":""},
    {"Symbol":"2B76","Exchange":"XETRA","Name":"iShares Automation & Robotics UCITS","Sector":"ETF","Industry":"Robotics & AI","Country":"Ireland","Asset_Type":"ETF","Notes":"RBOT UCITS"},
    {"Symbol":"GDX","Exchange":"LSE","Name":"VanEck Gold Miners UCITS","Sector":"Materials","Industry":"Gold Miners","Country":"Ireland","Asset_Type":"ETF","Notes":"UCITS version"},
])

# Write files
with open(os.path.join(project_dir, "app.py"), "w", encoding="utf-8") as f:
    f.write(app_py)

with open(os.path.join(project_dir, "requirements.txt"), "w", encoding="utf-8") as f:
    f.write(requirements_txt.strip())

with open(os.path.join(project_dir, ".streamlit/config.toml"), "w", encoding="utf-8") as f:
    f.write(config_toml.strip())

# Save sample xlsx
sample_xlsx = os.path.join(project_dir, "sample_master.xlsx")
with pd.ExcelWriter(sample_xlsx, engine="xlsxwriter") as writer:
    sample_df.to_excel(writer, index=False, sheet_name="Master")

# Zip project
zip_path = "/mnt/data/revisions_app_v2_streamlit.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(project_dir):
        for file in files:
            full = os.path.join(root, file)
            arc = os.path.relpath(full, project_dir)
            z.write(full, arcname=arc)

zip_path
