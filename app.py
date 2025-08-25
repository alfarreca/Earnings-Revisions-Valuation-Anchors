import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Earnings Revision Tracker",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
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
    .positive-revision {
        color: #28a745;
        font-weight: bold;
    }
    .negative-revision {
        color: #dc3545;
        font-weight: bold;
    }
    .cheap-valuation {
        color: #28a745;
        font-weight: bold;
    }
    .expensive-valuation {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class EarningsRevisionTracker:
    def __init__(self):
        self.df = None
        self.analysis_results = None
    
    def load_data(self, uploaded_file):
        """Load the Excel file with stock data"""
        try:
            self.df = pd.read_excel(uploaded_file)
            required_columns = ['Symbol', 'Exchange', 'Name', 'Sector', 'Industry', 'Country', 'Asset_Type']
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {missing_columns}")
                return False
            
            st.success(f"Successfully loaded {len(self.df)} stocks")
            return True
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return False
    
    def get_full_symbol(self, symbol, exchange):
        """Convert symbol to full format for yfinance"""
        if exchange == 'NYSE':
            return f"{symbol}"
        elif exchange == 'NASDAQ':
            return f"{symbol}"
        elif exchange == 'NSE':
            return f"{symbol}.NS"
        elif exchange == 'BSE':
            return f"{symbol}.BO"
        else:
            return symbol
    
    def fetch_earnings_data(self, symbol):
        """Fetch earnings estimates and historical data"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get current and forward EPS
            current_eps = info.get('trailingEps')
            forward_eps = info.get('forwardEps')
            
            # Get historical earnings data (last 4 quarters)
            earnings = ticker.quarterly_earnings
            if earnings is not None and not earnings.empty:
                last_quarter_eps = earnings.iloc[0]['Earnings'] if len(earnings) > 0 else None
                prev_quarter_eps = earnings.iloc[1]['Earnings'] if len(earnings) > 1 else None
            else:
                last_quarter_eps = None
                prev_quarter_eps = None
            
            # Get P/E ratios
            forward_pe = info.get('forwardPE')
            trailing_pe = info.get('trailingPE')
            
            # Get 5-year average P/E (if available)
            pe_5y_avg = None
            hist = ticker.history(period="5y")
            if not hist.empty:
                # This is a simplified approach - in practice, you'd need more sophisticated calculation
                pe_5y_avg = trailing_pe * 0.8  # Placeholder
            
            return {
                'current_eps': current_eps,
                'forward_eps': forward_eps,
                'last_quarter_eps': last_quarter_eps,
                'prev_quarter_eps': prev_quarter_eps,
                'forward_pe': forward_pe,
                'trailing_pe': trailing_pe,
                'pe_5y_avg': pe_5y_avg
            }
        except Exception as e:
            st.warning(f"Could not fetch data for {symbol}: {str(e)}")
            return None
    
    def calculate_revisions(self, row):
        """Calculate EPS revisions and valuation metrics"""
        symbol = row['Symbol']
        exchange = row['Exchange']
        full_symbol = self.get_full_symbol(symbol, exchange)
        
        data = self.fetch_earnings_data(full_symbol)
        if not data:
            return pd.Series({
                'Current_EPS_TTM': None,
                'Forward_EPS': None,
                'Last_Q_EPS': None,
                'Prev_Q_EPS': None,
                'Revision_Pct': None,
                'Forward_PE': None,
                'PE_5Y_Median': None,
                'Valuation_Premium_Pct': None,
                'Data_Status': 'Error'
            })
        
        # Calculate revision percentage
        revision_pct = None
        if data['last_quarter_eps'] is not None and data['prev_quarter_eps'] is not None:
            if data['prev_quarter_eps'] != 0:
                revision_pct = (data['last_quarter_eps'] - data['prev_quarter_eps']) / abs(data['prev_quarter_eps'])
        
        # Calculate valuation premium
        valuation_premium = None
        if data['forward_pe'] is not None and data['pe_5y_avg'] is not None:
            if data['pe_5y_avg'] != 0:
                valuation_premium = (data['forward_pe'] - data['pe_5y_avg']) / data['pe_5y_avg']
        
        return pd.Series({
            'Current_EPS_TTM': data['current_eps'],
            'Forward_EPS': data['forward_eps'],
            'Last_Q_EPS': data['last_quarter_eps'],
            'Prev_Q_EPS': data['prev_quarter_eps'],
            'Revision_Pct': revision_pct,
            'Forward_PE': data['forward_pe'],
            'PE_5Y_Median': data['pe_5y_avg'],
            'Valuation_Premium_Pct': valuation_premium,
            'Data_Status': 'Success'
        })
    
    def analyze_portfolio(self):
        """Analyze all stocks in the portfolio"""
        if self.df is None:
            return
        
        st.info("Fetching earnings data... This may take a few minutes.")
        
        # Create progress bar
        progress_bar = st.progress(0)
        results = []
        
        for i, (index, row) in enumerate(self.df.iterrows()):
            progress_bar.progress((i + 1) / len(self.df))
            result = self.calculate_revisions(row)
            results.append(result)
        
        # Combine results with original dataframe
        self.analysis_results = pd.concat([self.df, pd.DataFrame(results)], axis=1)
        
        st.success("Analysis complete!")
    
    def display_results(self):
        """Display the analysis results"""
        if self.analysis_results is None:
            return
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        positive_revisions = self.analysis_results[
            (self.analysis_results['Revision_Pct'] > 0) & 
            (self.analysis_results['Data_Status'] == 'Success')
        ]
        
        cheap_valuations = self.analysis_results[
            (self.analysis_results['Valuation_Premium_Pct'] < 0) & 
            (self.analysis_results['Data_Status'] == 'Success')
        ]
        
        with col1:
            st.metric("Total Stocks", len(self.analysis_results))
        with col2:
            st.metric("Successful Data", len(self.analysis_results[self.analysis_results['Data_Status'] == 'Success']))
        with col3:
            st.metric("Positive Revisions", len(positive_revisions))
        with col4:
            st.metric("Cheap Valuations", len(cheap_valuations))
        
        # Filters
        st.subheader("Filters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_revision = st.slider("Minimum Revision %", -1.0, 1.0, -1.0, 0.1)
        with col2:
            max_pe_premium = st.slider("Max PE Premium %", -1.0, 2.0, 2.0, 0.1)
        with col3:
            sectors = st.multiselect("Sectors", self.analysis_results['Sector'].unique())
        
        # Apply filters
        filtered_df = self.analysis_results[self.analysis_results['Data_Status'] == 'Success']
        filtered_df = filtered_df[filtered_df['Revision_Pct'] >= min_revision]
        filtered_df = filtered_df[filtered_df['Valuation_Premium_Pct'] <= max_pe_premium]
        
        if sectors:
            filtered_df = filtered_df[filtered_df['Sector'].isin(sectors)]
        
        # Display results table
        st.subheader("Earnings Revisions & Valuation Analysis")
        
        # Format the display dataframe
        display_df = filtered_df[[
            'Symbol', 'Name', 'Sector', 'Current_EPS_TTM', 'Forward_EPS',
            'Revision_Pct', 'Forward_PE', 'PE_5Y_Median', 'Valuation_Premium_Pct'
        ]].copy()
        
        # Format percentages
        display_df['Revision_Pct'] = display_df['Revision_Pct'].apply(
            lambda x: f"{x:+.2%}" if pd.notnull(x) else "N/A"
        )
        display_df['Valuation_Premium_Pct'] = display_df['Valuation_Premium_Pct'].apply(
            lambda x: f"{x:+.2%}" if pd.notnull(x) else "N/A"
        )
        
        # Format other columns
        for col in ['Current_EPS_TTM', 'Forward_EPS', 'Forward_PE', 'PE_5Y_Median']:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A"
            )
        
        st.dataframe(
            display_df,
            use_container_width=True,
            column_config={
                "Revision_Pct": st.column_config.TextColumn(
                    "Revision %",
                    help="(Current EPS - Previous EPS) / Previous EPS"
                ),
                "Valuation_Premium_Pct": st.column_config.TextColumn(
                    "Valuation Premium",
                    help="(Current P/E - 5Y Median P/E) / 5Y Median P/E"
                )
            }
        )
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="earnings_revisions_analysis.csv",
            mime="text/csv"
        )
        
        # Visualizations
        st.subheader("Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Revision distribution
            fig = px.histogram(
                filtered_df, 
                x='Revision_Pct',
                title='EPS Revision Distribution',
                labels={'Revision_Pct': 'Revision Percentage'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Valuation premium vs revision scatter
            fig = px.scatter(
                filtered_df,
                x='Valuation_Premium_Pct',
                y='Revision_Pct',
                color='Sector',
                hover_data=['Symbol', 'Name'],
                title='Valuation vs Earnings Revisions',
                labels={
                    'Valuation_Premium_Pct': 'Valuation Premium (%)',
                    'Revision_Pct': 'EPS Revision (%)'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Top picks section
        st.subheader("Top Investment Ideas")
        
        # Find stocks with positive revisions and cheap valuation
        top_picks = filtered_df[
            (filtered_df['Revision_Pct'] > 0) &
            (filtered_df['Valuation_Premium_Pct'] < 0)
        ]
        
        if not top_picks.empty:
            for _, stock in top_picks.iterrows():
                with st.container():
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>{stock['Name']} ({stock['Symbol']})</h3>
                            <p><strong>Sector:</strong> {stock['Sector']}</p>
                            <p><strong>EPS Revision:</strong> <span class="positive-revision">{stock['Revision_Pct']:+.2%}</span></p>
                            <p><strong>Valuation:</strong> <span class="cheap-valuation">{stock['Valuation_Premium_Pct']:+.2%} vs历史</span></p>
                            <p><strong>Forward P/E:</strong> {stock['Forward_PE']:.2f} | <strong>5Y Median:</strong> {stock['PE_5Y_Median']:.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.info("No stocks currently meet the criteria for positive revisions and cheap valuation.")

def main():
    st.markdown('<h1 class="main-header">📈 Earnings Revision Tracker</h1>', unsafe_allow_html=True)
    
    # Initialize tracker
    tracker = EarningsRevisionTracker()
    
    # File upload
    st.sidebar.header("Upload Portfolio")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your Excel file with stock data",
        type=['xlsx'],
        help="File should contain columns: Symbol, Exchange, Name, Sector, Industry, Country, Asset_Type, Notes"
    )
    
    if uploaded_file:
        if tracker.load_data(uploaded_file):
            if st.sidebar.button("Analyze Portfolio", type="primary"):
                tracker.analyze_portfolio()
            
            if tracker.analysis_results is not None:
                tracker.display_results()
    else:
        st.info("👈 Please upload an Excel file with your stock portfolio to begin analysis.")
        
        # Show sample data format
        st.subheader("Expected File Format")
        sample_data = pd.DataFrame({
            'Symbol': ['AAPL', 'MSFT', 'GOOGL'],
            'Exchange': ['NASDAQ', 'NASDAQ', 'NASDAQ'],
            'Name': ['Apple Inc', 'Microsoft Corp', 'Alphabet Inc'],
            'Sector': ['Technology', 'Technology', 'Technology'],
            'Industry': ['Consumer Electronics', 'Software', 'Internet'],
            'Country': ['USA', 'USA', 'USA'],
            'Asset_Type': ['Stock', 'Stock', 'Stock'],
            'Notes': ['', '', '']
        })
        st.dataframe(sample_data)

if __name__ == "__main__":
    main()
