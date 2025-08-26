# app.py — Earnings Revision Tracker (fixed & hardened)
# Preserves your UI/flows, adds robust data fetching, caching, EU exchange support,
# and fixes duplicate-column errors with pyarrow/Streamlit.

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.express as px

# ---------- Page configuration ----------
st.set_page_config(
    page_title="Earnings Revision Tracker",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Custom CSS ----------
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1E88E5;
        margin-bottom: 1rem;
    }
    .positive-revision { color: #28a745; font-weight: bold; }
    .negative-revision { color: #dc3545; font-weight: bold; }
    .cheap-valuation  { color: #28a745; font-weight: bold; }
    .expensive-valuation { color: #dc3545; font-weight: bold; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Helpers ----------

# Broad, EU-friendly exchange suffix map for Yahoo Finance
EX_SUFFIX = {
    # US
    "NYQ": "", "NYSE": "", "NYS": "", "NAS": "", "NASDAQ": "", "NMS": "", "AMEX": "", "ASE": "",
    # UK
    "LSE": ".L", "LON": ".L",
    # Germany
    "XETRA": ".DE", "XET": ".DE", "GER": ".DE", "FRA": ".F", "FRANKFURT": ".F",
    # France
    "PAR": ".PA", "EURONEXT PARIS": ".PA",
    # Netherlands
    "AMS": ".AS", "EURONEXT AMSTERDAM": ".AS",
    # Italy
    "MIL": ".MI", "EURONEXT MILAN": ".MI",
    # Spain
    "MAD": ".MC", "BME": ".MC",
    # Switzerland
    "SIX": ".SW", "SWX": ".SW",
    # Nordics
    "STO": ".ST", "STOCKHOLM": ".ST", "CPH": ".CO", "COPENHAGEN": ".CO",
    "HEL": ".HE", "HELSINKI": ".HE", "OSL": ".OL", "OSLO": ".OL",
    # Portugal
    "LIS": ".LS", "EURONEXT LISBON": ".LS",
    # Canada
    "TSX": ".TO", "TSE": ".TO", "TSXV": ".V", "TSX-V": ".V",
    # Australia
    "ASX": ".AX",
    # Japan
    "TSEJP": ".T", "JPX": ".T", "TSE": ".T",
    # Hong Kong / Korea / India
    "HKEX": ".HK", "KOSPI": ".KS", "KOSDAQ": ".KQ", "NSE": ".NS", "BSE": ".BO",
}

def _clean_string(x):
    return str(x).strip() if pd.notnull(x) else ""

def _ensure_columns(df, required):
    for c in required:
        if c not in df.columns:
            df[c] = ""
    return df

def _parse_symbol_and_exchange(symbol_value, exchange_value):
    """
    Accepts formats like 'LSE:GDX' or already-Yahoo formatted 'RMS.PA'.
    If exchange is provided separately, use it unless the symbol already has a suffix.
    """
    sym = _clean_string(symbol_value)
    exch = _clean_string(exchange_value)

    # Colon format: EXCH:SYMBOL
    if ":" in sym and not exch:
        ex, base = sym.split(":", 1)
        exch = ex.strip()
        sym = base.strip()

    # If already Yahoo-suffixed, keep as-is
    if "." in sym or "-" in sym:  # allow BRK-B etc.
        return sym

    # Attach suffix if we have a mapping for the exchange
    if exch:
        key = exch.upper()
        if key in EX_SUFFIX:
            return f"{sym}{EX_SUFFIX[key]}"

    return sym  # fallback no-suffix

def _fmt_pct(x):
    return f"{x:+.2%}" if pd.notnull(x) else "N/A"

def _fmt_num(x, nd=2):
    return f"{x:.{nd}f}" if pd.notnull(x) else "N/A"

@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_ticker_payload(full_symbol: str):
    """
    Robust fetch for EPS/PE and a best-effort 5Y PE median.
    Uses:
      - get_info()/info for EPS/PE and shares
      - quarterly_earnings for net income (Earnings)
      - monthly price for PE history construction
    Returns dict with None on missing fields.
    """
    t = yf.Ticker(full_symbol)

    # --- info (with fallbacks) ---
    info = {}
    try:
        # new yfinance
        info = t.get_info()
    except Exception:
        try:
            info = t.info
        except Exception:
            info = {}

    trailing_eps   = info.get("trailingEps")
    forward_eps    = info.get("forwardEps")
    trailing_pe    = info.get("trailingPE")
    forward_pe     = info.get("forwardPE")
    shares_out     = info.get("sharesOutstanding")

    # --- quarterly net income (Earnings, not EPS) ---
    q_earn = None
    try:
        q = t.quarterly_earnings  # columns: Revenue, Earnings
        if isinstance(q, pd.DataFrame) and not q.empty and "Earnings" in q.columns:
            q_earn = q["Earnings"].sort_index()
    except Exception:
        q_earn = None

    # --- monthly price history (reduce load vs daily) ---
    hist = None
    try:
        hist = t.history(period="5y", interval="1mo", auto_adjust=False)
        if isinstance(hist, pd.DataFrame) and hist.empty:
            hist = None
    except Exception:
        hist = None

    # --- Build a PE history to estimate 5Y median ---
    pe_5y_median = None
    if (hist is not None) and (q_earn is not None) and (shares_out is not None) and shares_out:
        try:
            # Build TTM net income per quarter (rolling sum of last 4 quarters)
            q_earn_ttm = q_earn.rolling(4, min_periods=4).sum().dropna()
            if not q_earn_ttm.empty:
                ttm_eps = q_earn_ttm / float(shares_out)

                # Align to monthly price index by forward filling last known TTM EPS
                ttm_eps_monthly = ttm_eps.reindex(hist.index, method="ffill")
                # Compute PE = Price / TTM EPS; avoid division by zero
                pe_series = hist["Close"] / ttm_eps_monthly.replace(0, np.nan)
                pe_series = pe_series.replace([np.inf, -np.inf], np.nan).dropna()

                if not pe_series.empty:
                    pe_5y_median = float(pe_series.median())
        except Exception:
            pe_5y_median = None

    # Final payload
    return {
        "current_eps": trailing_eps,
        "forward_eps": forward_eps,
        "trailing_pe": trailing_pe,
        "forward_pe": forward_pe,
        "shares_out": shares_out,
        "pe_5y_median": pe_5y_median,  # may be None; we'll fall back later
        "quarters_eps_like": q_earn,   # net income series; used for revision calc
    }

def _calc_revision_pct(q_earn_series: pd.Series):
    """
    Use last two quarters of net income (not EPS) as a proxy to measure direction of revision.
    If we have at least 2 quarters, compute (last - prev) / |prev|.
    """
    if q_earn_series is None or not isinstance(q_earn_series, pd.Series) or q_earn_series.empty:
        return None, None, None
    q = q_earn_series.sort_index()
    last = q.iloc[-1] if len(q) >= 1 else None
    prev = q.iloc[-2] if len(q) >= 2 else None
    rev = None
    if last is not None and prev not in (None, 0):
        rev = (last - prev) / abs(prev)
    return last, prev, rev

# ---------- App Core ----------

class EarningsRevisionTracker:
    def __init__(self):
        self.df = None
        self.analysis_results = None

    def load_data(self, uploaded_file):
        """Load Excel, ensure required columns exist (only Symbol truly required)."""
        try:
            df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return False

        if "Symbol" not in df.columns:
            st.error("Missing required column: 'Symbol'")
            return False

        # Ensure expected columns exist so downstream UI works unchanged.
        expected = ["Exchange", "Name", "Sector", "Industry", "Country", "Asset_Type", "Notes"]
        df = _ensure_columns(df, expected)

        # Always rebuild YF_Symbol from Symbol+Exchange to guarantee a single column
        df["YF_Symbol"] = df.apply(
            lambda r: _parse_symbol_and_exchange(r["Symbol"], r["Exchange"]), axis=1
        )

        # Ensure unique column names at the source
        df = df.loc[:, ~df.columns.duplicated()]

        self.df = df
        st.success(f"Successfully loaded {len(self.df)} rows")
        return True

    def calculate_row(self, row):
        """Fetch & compute metrics for one row."""
        full_symbol = row.get("YF_Symbol") or row.get("Symbol")

        data = _fetch_ticker_payload(full_symbol)
        if not data:
            return pd.Series({
                "Current_EPS_TTM": None,
                "Forward_EPS": None,
                "Last_Q_NetIncome": None,
                "Prev_Q_NetIncome": None,
                "Revision_Pct": None,
                "Forward_PE": None,
                "PE_5Y_Median": None,
                "Valuation_Premium_Pct": None,
                "Data_Status": "Error",
            })

        # Revision proxy from net income series
        last_q, prev_q, rev_pct = _calc_revision_pct(data.get("quarters_eps_like"))

        # 5Y PE median fallback if needed
        pe_5y = data.get("pe_5y_median")
        if pe_5y is None:
            # fall back to trailing PE as a placeholder; avoids NaN explosions in UI
            pe_5y = data.get("trailing_pe")

        # Valuation premium: (forward PE - 5Y median PE) / 5Y median PE
        val_prem = None
        if data.get("forward_pe") is not None and pe_5y not in (None, 0):
            val_prem = (data["forward_pe"] - pe_5y) / pe_5y

        # NOTE: We intentionally DO NOT return YF_Symbol here to avoid duplicates on concat
        return pd.Series({
            "Current_EPS_TTM": data.get("current_eps"),
            "Forward_EPS": data.get("forward_eps"),
            "Last_Q_NetIncome": last_q,
            "Prev_Q_NetIncome": prev_q,
            "Revision_Pct": rev_pct,
            "Forward_PE": data.get("forward_pe"),
            "PE_5Y_Median": pe_5y,
            "Valuation_Premium_Pct": val_prem,
            "Data_Status": "Success",
        })

    def analyze_portfolio(self, max_rows=None):
        """Analyze all stocks (optionally cap rows for speed on huge files)."""
        if self.df is None or self.df.empty:
            return

        df = self.df.copy()
        if max_rows and len(df) > max_rows:
            st.info(f"Processing first {max_rows} rows out of {len(df)} to avoid timeouts.")
            df = df.iloc[:max_rows].copy()

        st.info("Fetching earnings data. This can take a bit depending on list size…")

        # Progress bar
        progress_bar = st.progress(0)
        results = []

        for i, (_, row) in enumerate(df.iterrows(), start=1):
            result = self.calculate_row(row)
            results.append(result)
            progress_bar.progress(i / len(df))

        self.analysis_results = pd.concat(
            [df.reset_index(drop=True), pd.DataFrame(results).reset_index(drop=True)],
            axis=1
        )

        # Ensure unique columns after concat (fixes pyarrow duplicate-column error)
        self.analysis_results = self.analysis_results.loc[:, ~self.analysis_results.columns.duplicated()]

        st.success("Analysis complete!")

    def display_results(self):
        """Display metrics, filters, table, charts, and top picks."""
        if self.analysis_results is None or self.analysis_results.empty:
            return

        # One more safety: ensure unique columns before any rendering
        self.analysis_results = self.analysis_results.loc[:, ~self.analysis_results.columns.duplicated()]

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        success_mask = self.analysis_results["Data_Status"].eq("Success")
        positive_revisions = self.analysis_results[success_mask & (self.analysis_results["Revision_Pct"] > 0)]
        cheap_valuations = self.analysis_results[success_mask & (self.analysis_results["Valuation_Premium_Pct"] < 0)]

        with col1:
            st.metric("Total Rows", len(self.analysis_results))
        with col2:
            st.metric("Successful Data", int(success_mask.sum()))
        with col3:
            st.metric("Positive Revisions", len(positive_revisions))
        with col4:
            st.metric("Cheap Valuations", len(cheap_valuations))

        # Filters
        st.subheader("Filters")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            min_revision = st.slider("Minimum Revision %", -1.0, 1.0, -1.0, 0.05)
        with c2:
            max_pe_premium = st.slider("Max PE Premium %", -1.0, 2.0, 2.0, 0.05)
        with c3:
            sectors = st.multiselect(
                "Sectors",
                sorted([s for s in self.analysis_results["Sector"].dropna().unique() if s != ""])
            )
        with c4:
            show_only_success = st.checkbox("Only successful rows", True)

        # Apply filters
        filtered = self.analysis_results.copy()
        filtered = filtered.loc[:, ~filtered.columns.duplicated()]  # ensure unique col names

        if show_only_success:
            filtered = filtered[filtered["Data_Status"] == "Success"]

        filtered = filtered[
            (filtered["Revision_Pct"].fillna(-1.0) >= min_revision) &
            (filtered["Valuation_Premium_Pct"].fillna(2.0) <= max_pe_premium)
        ]
        if sectors:
            filtered = filtered[filtered["Sector"].isin(sectors)]

        # Display table
        st.subheader("Earnings Revisions & Valuation Analysis")

        display_cols = [
            "Symbol", "YF_Symbol", "Name", "Sector",
            "Current_EPS_TTM", "Forward_EPS",
            "Revision_Pct", "Forward_PE", "PE_5Y_Median", "Valuation_Premium_Pct"
        ]
        for c in display_cols:
            if c not in filtered.columns:
                filtered[c] = np.nan

        # One more guard before view
        filtered = filtered.loc[:, ~filtered.columns.duplicated()]

        table_df = filtered[display_cols].copy()

        # Format numeric columns safely for display
        table_df["Revision_Pct"] = table_df["Revision_Pct"].apply(_fmt_pct)
        table_df["Valuation_Premium_Pct"] = table_df["Valuation_Premium_Pct"].apply(_fmt_pct)
        for col in ["Current_EPS_TTM", "Forward_EPS", "Forward_PE", "PE_5Y_Median"]:
            table_df[col] = table_df[col].apply(_fmt_num)

        # Some Streamlit versions may not have column_config; try it, then fall back
        try:
            st.dataframe(
                table_df,
                use_container_width=True,
                column_config={
                    "Revision_Pct": st.column_config.TextColumn(
                        "Revision %",
                        help="(Last quarter net income − Previous) / |Previous| (proxy for EPS revision)"
                    ),
                    "Valuation_Premium_Pct": st.column_config.TextColumn(
                        "Valuation Premium",
                        help="(Forward P/E − 5Y median P/E) / 5Y median P/E"
                    ),
                },
            )
        except Exception:
            st.dataframe(table_df, use_container_width=True)

        # Download
        st.download_button(
            label="Download Results as CSV",
            data=filtered.to_csv(index=False),
            file_name="earnings_revisions_analysis.csv",
            mime="text/csv",
        )

        # Visualizations
        st.subheader("Visualizations")
        v1, v2 = st.columns(2)

        with v1:
            fig = px.histogram(
                filtered,
                x="Revision_Pct",
                nbins=30,
                title="EPS Revision Proxy (Net Income) — Distribution",
                labels={"Revision_Pct": "Revision % (proxy)"},
            )
            st.plotly_chart(fig, use_container_width=True)

        with v2:
            fig2 = px.scatter(
                filtered,
                x="Valuation_Premium_Pct",
                y="Revision_Pct",
                color="Sector",
                hover_data=["Symbol", "Name", "YF_Symbol"],
                title="Valuation vs. Earnings Revision (Proxy)",
                labels={
                    "Valuation_Premium_Pct": "Valuation Premium (%)",
                    "Revision_Pct": "EPS Revision Proxy (%)",
                },
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Top picks (positive revisions + cheap valuation)
        st.subheader("Top Investment Ideas")
        top = filtered[
            filtered["Revision_Pct"].fillna(-1) > 0
        ][filtered["Valuation_Premium_Pct"].fillna(1) < 0]

        if not top.empty:
            for _, stock in top.iterrows():
                rev_txt = _fmt_pct(stock["Revision_Pct"])
                prem_txt = _fmt_pct(stock["Valuation_Premium_Pct"])
                fpe_txt = _fmt_num(stock["Forward_PE"])
                pe5_txt = _fmt_num(stock["PE_5Y_Median"])
                st.markdown(
                    f"""
<div class="metric-card">
  <h3>{stock.get('Name') or ''} ({stock.get('Symbol') or ''})</h3>
  <p><strong>Sector:</strong> {stock.get('Sector') or 'N/A'}</p>
  <p><strong>EPS Revision (proxy):</strong> <span class="positive-revision">{rev_txt}</span></p>
  <p><strong>Valuation:</strong> <span class="cheap-valuation">{prem_txt} vs 5Y median</span></p>
  <p><strong>Forward P/E:</strong> {fpe_txt} &nbsp; | &nbsp; <strong>5Y Median:</strong> {pe5_txt}</p>
</div>
""",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No stocks currently meet the criteria for positive revisions and cheap valuation.")

def main():
    st.markdown('<h1 class="main-header">📈 Earnings Revision Tracker</h1>', unsafe_allow_html=True)

    tracker = EarningsRevisionTracker()

    # ---------- Sidebar: Upload & Controls ----------
    st.sidebar.header("Upload Portfolio")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your Excel file",
        type=["xlsx"],
        help="Columns: at minimum 'Symbol'. Optional: Exchange, Name, Sector, Industry, Country, Asset_Type, Notes",
    )

    max_rows = st.sidebar.number_input("Max tickers per run (0 = no cap)", min_value=0, max_value=5000, value=0, step=50)

    if uploaded_file:
        if tracker.load_data(uploaded_file):
            if st.sidebar.button("Analyze Portfolio", type="primary"):
                tracker.analyze_portfolio(max_rows if max_rows > 0 else None)

            if tracker.analysis_results is not None:
                tracker.display_results()
    else:
        st.info("👈 Please upload an Excel file with your stock portfolio to begin analysis.")
        st.subheader("Expected File Format")
        sample = pd.DataFrame(
            {
                "Symbol": ["AAPL", "MSFT", "LSE:GDX"],
                "Exchange": ["NASDAQ", "NASDAQ", ""],
                "Name": ["Apple Inc", "Microsoft Corp", "VanEck Gold Miners UCITS"],
                "Sector": ["Technology", "Technology", "Materials"],
                "Industry": ["Consumer Electronics", "Software", "Gold Miners"],
                "Country": ["USA", "USA", "UK"],
                "Asset_Type": ["Stock", "Stock", "ETF"],
                "Notes": ["", "", ""],
            }
        )
        st.dataframe(sample, use_container_width=True)

if __name__ == "__main__":
    main()
