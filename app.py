# app.py — Earnings Revision Tracker (session-state fixed)
# Keeps all features; robust yfinance; EU exchange support; duplicate-column guard; filters now reactive.

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px

# ---------- Page configuration ----------
st.set_page_config(page_title="Earnings Revision Tracker", page_icon="📈", layout="wide")

# ---------- Custom CSS ----------
st.markdown(
    """
<style>
    .main-header { font-size: 2.5rem; color: #1E88E5; text-align: center; margin-bottom: 2rem; }
    .metric-card { background-color: #f8f9fa; padding: 1rem; border-radius: .5rem; border-left: 4px solid #1E88E5; margin-bottom: 1rem; }
    .positive-revision { color: #28a745; font-weight: bold; }
    .cheap-valuation { color: #28a745; font-weight: bold; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Helpers ----------
EX_SUFFIX = {
    "NYQ": "", "NYSE": "", "NYS": "", "NAS": "", "NASDAQ": "", "NMS": "", "AMEX": "", "ASE": "",
    "LSE": ".L", "LON": ".L",
    "XETRA": ".DE", "XET": ".DE", "GER": ".DE", "FRA": ".F", "FRANKFURT": ".F",
    "PAR": ".PA", "EURONEXT PARIS": ".PA",
    "AMS": ".AS", "EURONEXT AMSTERDAM": ".AS",
    "MIL": ".MI", "EURONEXT MILAN": ".MI",
    "MAD": ".MC", "BME": ".MC",
    "SIX": ".SW", "SWX": ".SW",
    "STO": ".ST", "STOCKHOLM": ".ST", "CPH": ".CO", "COPENHAGEN": ".CO",
    "HEL": ".HE", "HELSINKI": ".HE", "OSL": ".OL", "OSLO": ".OL",
    "LIS": ".LS", "EURONEXT LISBON": ".LS",
    "TSX": ".TO", "TSE": ".TO", "TSXV": ".V", "TSX-V": ".V",
    "ASX": ".AX",
    "TSEJP": ".T", "JPX": ".T", "TSE": ".T",
    "HKEX": ".HK", "KOSPI": ".KS", "KOSDAQ": ".KQ", "NSE": ".NS", "BSE": ".BO",
}

def _clean_string(x): return str(x).strip() if pd.notnull(x) else ""
def _ensure_columns(df, cols):
    for c in cols:
        if c not in df.columns: df[c] = ""
    return df
def _parse_symbol_and_exchange(symbol_value, exchange_value):
    sym = _clean_string(symbol_value); exch = _clean_string(exchange_value)
    if ":" in sym and not exch:
        ex, base = sym.split(":", 1); exch = ex.strip(); sym = base.strip()
    if "." in sym or "-" in sym: return sym
    if exch and exch.upper() in EX_SUFFIX: return f"{sym}{EX_SUFFIX[exch.upper()]}"
    return sym
def _fmt_pct(x): return f"{x:+.2%}" if pd.notnull(x) else "N/A"
def _fmt_num(x, nd=2): return f"{x:.{nd}f}" if pd.notnull(x) else "N/A"

@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_ticker_payload(full_symbol: str):
    if not full_symbol: return {}
    t = yf.Ticker(full_symbol)

    # Normalize info
    info = {}
    try:
        tmp = t.get_info()
        info = tmp if isinstance(tmp, dict) else {}
    except Exception:
        try:
            tmp = t.info
            info = tmp if isinstance(tmp, dict) else {}
        except Exception:
            info = {}

    fast = {}
    try:
        fi = t.fast_info
        fast = dict(fi.__dict__) if hasattr(fi, "__dict__") else (fi if isinstance(fi, dict) else {})
    except Exception: pass

    def _iget(d, k, alt=None): return d.get(k) if isinstance(d, dict) else alt

    trailing_eps = _iget(info, "trailingEps")
    forward_eps  = _iget(info, "forwardEps")
    trailing_pe  = _iget(info, "trailingPE", _iget(fast, "trailing_pe"))
    forward_pe   = _iget(info, "forwardPE",  _iget(fast, "forward_pe"))
    shares_out   = _iget(info, "sharesOutstanding", _iget(fast, "shares_outstanding"))

    # quarterly net income proxy
    q_earn = None
    try:
        q = t.quarterly_earnings
        if isinstance(q, pd.DataFrame) and not q.empty and "Earnings" in q.columns:
            q_earn = q["Earnings"].sort_index()
    except Exception: pass

    # 5y monthly price
    hist = None
    try:
        hist = t.history(period="5y", interval="1mo", auto_adjust=False)
        if isinstance(hist, pd.DataFrame) and hist.empty: hist = None
    except Exception: pass

    pe_5y_median = None
    if (hist is not None) and (q_earn is not None) and (shares_out or 0):
        try:
            q_earn_ttm = q_earn.rolling(4, min_periods=4).sum().dropna()
            if not q_earn_ttm.empty:
                ttm_eps = q_earn_ttm / float(shares_out)
                ttm_eps_monthly = ttm_eps.reindex(hist.index, method="ffill")
                pe_series = hist["Close"] / ttm_eps_monthly.replace(0, np.nan)
                pe_series = pe_series.replace([np.inf, -np.inf], np.nan).dropna()
                if not pe_series.empty: pe_5y_median = float(pe_series.median())
        except Exception: pass

    return {
        "current_eps": trailing_eps,
        "forward_eps": forward_eps,
        "trailing_pe": trailing_pe,
        "forward_pe": forward_pe,
        "shares_out": shares_out,
        "pe_5y_median": pe_5y_median,
        "quarters_eps_like": q_earn,
    }

def _calc_revision_pct(q_earn_series: pd.Series):
    if q_earn_series is None or not isinstance(q_earn_series, pd.Series) or q_earn_series.empty:
        return None, None, None
    q = q_earn_series.sort_index()
    last = q.iloc[-1] if len(q) >= 1 else None
    prev = q.iloc[-2] if len(q) >= 2 else None
    rev = None
    if last is not None and prev not in (None, 0):
        rev = (last - prev) / abs(prev)
    return last, prev, rev

# ---------- Session state boot ----------
for k, v in {
    "input_df": None,
    "analysis_results": None,
    "failed_symbols": [],
}.items():
    if k not in st.session_state: st.session_state[k] = v

# ---------- UI ----------
st.markdown('<h1 class="main-header">📈 Earnings Revision Tracker</h1>', unsafe_allow_html=True)

st.sidebar.header("Upload Portfolio")
uploaded = st.sidebar.file_uploader(
    "Upload your Excel file",
    type=["xlsx"],
    help="Minimal column: Symbol. Optional: Exchange, Name, Sector, Industry, Country, Asset_Type, Notes",
    key="uploader",
)
max_rows = st.sidebar.number_input("Max tickers per run (0 = no cap)",
                                   min_value=0, max_value=5000, value=0, step=50, key="maxrows")

# ---------- Load file ----------
if uploaded:
    try:
        df = pd.read_excel(uploaded)
        if "Symbol" not in df.columns:
            st.error("Missing required column: 'Symbol'")
        else:
            df = _ensure_columns(df, ["Exchange","Name","Sector","Industry","Country","Asset_Type","Notes"])
            df["YF_Symbol"] = df.apply(lambda r: _parse_symbol_and_exchange(r["Symbol"], r["Exchange"]), axis=1)
            df = df.loc[:, ~df.columns.duplicated()]
            st.session_state.input_df = df
            st.success(f"Loaded {len(df)} rows.")
    except Exception as e:
        st.error(f"Error loading file: {e}")

# ---------- Analyze button ----------
if st.session_state.input_df is not None:
    if st.sidebar.button("Analyze Portfolio", type="primary", key="analyze"):
        df = st.session_state.input_df.copy()
        if st.session_state.maxrows and len(df) > st.session_state.maxrows:
            st.info(f"Processing first {st.session_state.maxrows} rows out of {len(df)} to avoid timeouts.")
            df = df.iloc[: st.session_state.maxrows].copy()

        st.info("Fetching earnings data. This can take a bit depending on list size…")
        bar = st.progress(0.0)
        results = []
        fails = []

        for i, (_, row) in enumerate(df.iterrows(), start=1):
            sym = row.get("YF_Symbol") or row.get("Symbol")
            try:
                data = _fetch_ticker_payload(sym)
                last_q, prev_q, rev_pct = _calc_revision_pct(data.get("quarters_eps_like"))
                pe_5y = data.get("pe_5y_median") if data.get("pe_5y_median") is not None else data.get("trailing_pe")
                val_prem = ((data.get("forward_pe") - pe_5y) / pe_5y) if (data.get("forward_pe") is not None and pe_5y not in (None, 0)) else None

                results.append(pd.Series({
                    "Current_EPS_TTM": data.get("current_eps"),
                    "Forward_EPS": data.get("forward_eps"),
                    "Last_Q_NetIncome": last_q,
                    "Prev_Q_NetIncome": prev_q,
                    "Revision_Pct": rev_pct,
                    "Forward_PE": data.get("forward_pe"),
                    "PE_5Y_Median": pe_5y,
                    "Valuation_Premium_Pct": val_prem,
                    "Data_Status": "Success",
                }))
            except Exception:
                results.append(pd.Series({
                    "Current_EPS_TTM": None, "Forward_EPS": None,
                    "Last_Q_NetIncome": None, "Prev_Q_NetIncome": None,
                    "Revision_Pct": None, "Forward_PE": None,
                    "PE_5Y_Median": None, "Valuation_Premium_Pct": None,
                    "Data_Status": "Error",
                }))
                fails.append(sym)

            bar.progress(i / len(df))

        out = pd.concat([df.reset_index(drop=True), pd.DataFrame(results).reset_index(drop=True)], axis=1)
        out = out.loc[:, ~out.columns.duplicated()]
        st.session_state.analysis_results = out
        st.session_state.failed_symbols = sorted(set(fails))
        st.success("Analysis complete!")

# ---------- Results & Filters (reactive) ----------
if st.session_state.analysis_results is not None:
    res = st.session_state.analysis_results.copy()
    res = res.loc[:, ~res.columns.duplicated()]

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    success_mask = res["Data_Status"].eq("Success")
    with c1: st.metric("Total Rows", len(res))
    with c2: st.metric("Successful Data", int(success_mask.sum()))
    with c3: st.metric("Positive Revisions", int((success_mask & (res["Revision_Pct"] > 0)).sum()))
    with c4: st.metric("Cheap Valuations", int((success_mask & (res["Valuation_Premium_Pct"] < 0)).sum()))

    # Optional: show failed symbols
    if st.session_state.failed_symbols:
        with st.expander("⚠️ Failed symbols"):
            st.write(", ".join(st.session_state.failed_symbols))

    st.subheader("Filters")
    f1, f2, f3, f4 = st.columns([1,1,2,1])
    min_revision = f1.slider("Minimum Revision %", -1.0, 1.0, -1.0, 0.05, key="min_rev")
    max_pe_prem  = f2.slider("Max PE Premium %", -1.0, 2.0, 2.0, 0.05, key="max_prem")
    sectors_opts = sorted([s for s in res["Sector"].dropna().unique() if s != ""])
    sectors_sel  = f3.multiselect("Sectors", options=sectors_opts, default=[], key="sect_sel")
    only_success = f4.checkbox("Only successful rows", True, key="only_ok")

    # Apply filters
    flt = res
    if only_success: flt = flt[flt["Data_Status"] == "Success"]
    flt = flt[(flt["Revision_Pct"].fillna(-1.0) >= min_revision) & (flt["Valuation_Premium_Pct"].fillna(2.0) <= max_pe_prem)]
    if sectors_sel: flt = flt[flt["Sector"].isin(sectors_sel)]

    st.subheader("Earnings Revisions & Valuation Analysis")
    display_cols = ["Symbol","YF_Symbol","Name","Sector","Current_EPS_TTM","Forward_EPS","Revision_Pct","Forward_PE","PE_5Y_Median","Valuation_Premium_Pct"]
    for c in display_cols:
        if c not in flt.columns: flt[c] = np.nan
    flt = flt.loc[:, ~flt.columns.duplicated()]
    table_df = flt[display_cols].copy()
    table_df["Revision_Pct"] = table_df["Revision_Pct"].apply(_fmt_pct)
    table_df["Valuation_Premium_Pct"] = table_df["Valuation_Premium_Pct"].apply(_fmt_pct)
    for col in ["Current_EPS_TTM","Forward_EPS","Forward_PE","PE_5Y_Median"]:
        table_df[col] = table_df[col].apply(_fmt_num)

    try:
        st.dataframe(
            table_df, use_container_width=True,
            column_config={
                "Revision_Pct": st.column_config.TextColumn("Revision %", help="(Last quarter net income − Previous) / |Previous| (proxy)"),
                "Valuation_Premium_Pct": st.column_config.TextColumn("Valuation Premium", help="(Forward P/E − 5Y median) / 5Y median"),
            },
        )
    except Exception:
        st.dataframe(table_df, use_container_width=True)

    st.download_button("Download Results as CSV", data=flt.to_csv(index=False), file_name="earnings_revisions_analysis.csv", mime="text/csv")

    st.subheader("Visualizations")
    v1, v2 = st.columns(2)
    with v1:
        st.plotly_chart(px.histogram(flt, x="Revision_Pct", nbins=30, title="EPS Revision Proxy — Distribution",
                                     labels={"Revision_Pct": "Revision % (proxy)"}), use_container_width=True)
    with v2:
        st.plotly_chart(px.scatter(flt, x="Valuation_Premium_Pct", y="Revision_Pct", color="Sector",
                                   hover_data=["Symbol","Name","YF_Symbol"],
                                   title="Valuation vs. Earnings Revision (Proxy)",
                                   labels={"Valuation_Premium_Pct":"Valuation Premium (%)","Revision_Pct":"EPS Revision Proxy (%)"}),
                        use_container_width=True)

    st.subheader("Top Investment Ideas")
    top = flt[flt["Revision_Pct"].fillna(-1) > 0]
    top = top[top["Valuation_Premium_Pct"].fillna(1) < 0]
    if not top.empty:
        for _, s in top.iterrows():
            st.markdown(
                f"""
<div class="metric-card">
  <h3>{s.get('Name') or ''} ({s.get('Symbol') or ''})</h3>
  <p><strong>Sector:</strong> {s.get('Sector') or 'N/A'}</p>
  <p><strong>EPS Revision (proxy):</strong> <span class="positive-revision">{_fmt_pct(s['Revision_Pct'])}</span></p>
  <p><strong>Valuation:</strong> <span class="cheap-valuation">{_fmt_pct(s['Valuation_Premium_Pct'])} vs 5Y median</span></p>
  <p><strong>Forward P/E:</strong> {_fmt_num(s['Forward_PE'])} &nbsp; | &nbsp; <strong>5Y Median:</strong> {_fmt_num(s['PE_5Y_Median'])}</p>
</div>
""",
                unsafe_allow_html=True,
            )
    else:
        st.info("No stocks currently meet the criteria for positive revisions and cheap valuation.")

# Empty state
if (st.session_state.input_df is None) and (st.session_state.analysis_results is None):
    st.info("👈 Upload an Excel file to begin.")
    sample = pd.DataFrame({
        "Symbol": ["AAPL", "MSFT", "LSE:GDX"],
        "Exchange": ["NASDAQ", "NASDAQ", ""],
        "Name": ["Apple Inc", "Microsoft Corp", "VanEck Gold Miners UCITS"],
        "Sector": ["Technology", "Technology", "Materials"],
        "Industry": ["Consumer Electronics", "Software", "Gold Miners"],
        "Country": ["USA", "USA", "UK"],
        "Asset_Type": ["Stock", "Stock", "ETF"],
        "Notes": ["", "", ""],
    })
    st.dataframe(sample, use_container_width=True)
