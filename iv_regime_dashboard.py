"""
Implied Volatility Regime Analysis Dashboard
=============================================
Live implied volatility regime detection using Yahoo Finance data.
No brokerage account required — works instantly for any US equity.

Methodology:
- Pulls historical price data from Yahoo Finance (free, no API key)
- Computes realised volatility as proxy for implied volatility
- Classifies regime using rolling percentile rank over 252-day window
- Validates mean reversion hypothesis via forward regression
- Splits regression by high/low vol regime to measure regime-dependent reversion speed

Run with: streamlit run iv_regime_dashboard.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="IV Regime Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .metric-box { background-color: #161b22; border-radius: 8px; padding: 12px;
                  border: 1px solid #30363d; }
    .signal-box { background-color: #161b22; border-radius: 8px; padding: 16px;
                  border: 1px solid #30363d; margin: 8px 0; }
    h1, h2, h3 { color: #58a6ff; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA FETCHING
# =============================================================================
@st.cache_data(ttl=3600)
def fetch_volatility_data(symbol: str, period: str = "2y") -> pd.DataFrame:
    """
    Fetch historical price data and compute volatility metrics.

    Uses close-to-close log returns to compute realised volatility,
    annualised by sqrt(252). Rolling windows compute short and long
    term volatility for regime analysis.

    Note: Yahoo Finance provides free historical OHLCV data.
    True implied volatility requires options market data (paid).
    Realised volatility is a close proxy for short-dated IV.

    Parameters
    ----------
    symbol : ticker symbol e.g. 'SPY', 'AAPL', 'QQQ'
    period : history length e.g. '2y', '1y', '6mo'
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)

        if df.empty:
            return pd.DataFrame()

        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)

        # Log returns
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))

        # Realised volatility — rolling standard deviation of log returns, annualised
        # 21-day window ≈ one trading month — short-term vol
        df['rv_21d'] = df['log_return'].rolling(21).std() * np.sqrt(252)

        # 63-day window ≈ one quarter — medium-term vol
        df['rv_63d'] = df['log_return'].rolling(63).std() * np.sqrt(252)

        # Parkinson volatility — uses High/Low range, more efficient estimator
        # Parkinson: sqrt(1/(4*N*ln2) * sum(ln(H/L)^2))
        df['parkinson_vol'] = np.sqrt(
            (1 / (4 * np.log(2))) *
            (np.log(df['High'] / df['Low']) ** 2).rolling(21).mean() * 252
        )

        # Primary vol measure — use Parkinson as it's more robust
        df['implied_vol'] = df['parkinson_vol']

        # Rolling percentile rank over 252-day window
        # Answers: where does today's vol sit relative to the past year?
        df['iv_percentile'] = df['implied_vol'].rolling(252).rank(pct=True)

        # VIX-style term structure: short vol minus long vol
        # Positive = short vol elevated vs long vol = potential mean reversion
        df['vol_term_structure'] = df['rv_21d'] - df['rv_63d']

        return df.dropna()

    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()


# =============================================================================
# REGIME CLASSIFICATION
# =============================================================================
def classify_regime(percentile: float) -> tuple:
    """
    Five-bucket regime classification based on percentile rank.

    > 80th  → HIGH VOL     → sell vol, expect mean reversion down
    > 60th  → ABOVE AVG   → mild downward pressure
    > 40th  → NORMAL      → no strong signal
    > 20th  → BELOW AVG   → mild upward pressure
    ≤ 20th  → LOW VOL     → buy vol, expect mean reversion up

    Returns (regime_label, signal, regime_color, signal_color)
    """
    if percentile > 0.80:
        return "HIGH VOL",     "SELL VOL — expect reversion down", "#f85149", "#7ee787"
    elif percentile > 0.60:
        return "ABOVE AVG",   "MILD DOWNWARD PRESSURE",           "#d29922", "#58a6ff"
    elif percentile > 0.40:
        return "NORMAL",      "NEUTRAL — no strong signal",        "#c9d1d9", "#c9d1d9"
    elif percentile > 0.20:
        return "BELOW AVG",   "MILD UPWARD PRESSURE",              "#58a6ff", "#d29922"
    else:
        return "LOW VOL",     "BUY VOL — expect reversion up",     "#7ee787", "#f85149"


# =============================================================================
# FORWARD REGRESSION ANALYSIS
# =============================================================================
def forward_regression_analysis(df: pd.DataFrame, forward_days: int = 21) -> dict:
    """
    Validate mean reversion via forward regression.

    Regresses 21-day forward average vol against current vol.
    Slope < 1 confirms mean reversion — high vol today predicts lower vol ahead.

    Regime split: intersection of regression line with y=x diagonal
    defines the natural boundary between buy-vol and sell-vol territory.

    Returns dict with regression results and analysis DataFrame.
    """
    analysis = df[['implied_vol', 'iv_percentile']].copy()

    # Forward average vol — 21-day rolling mean shifted back
    analysis['forward_vol'] = analysis['implied_vol'].rolling(
        forward_days, min_periods=1
    ).mean().shift(-forward_days)

    analysis['vol_diff'] = analysis['forward_vol'] - analysis['implied_vol']
    analysis = analysis.dropna()

    if len(analysis) < 30:
        return {}

    # Full sample regression
    slope, intercept, r, p, se = stats.linregress(
        analysis['implied_vol'], analysis['forward_vol']
    )

    # Regime boundary: where regression line crosses y=x
    # y = slope*x + intercept = x  →  x = intercept / (1 - slope)
    intersection = intercept / (1 - slope) if abs(1 - slope) > 1e-6 \
                   else analysis['implied_vol'].median()

    high_mask = analysis['implied_vol'] > intersection
    low_mask  = ~high_mask

    results = {
        'analysis_df':  analysis,
        'intersection': intersection,
        'high_mask':    high_mask,
        'low_mask':     low_mask,
        'full': {
            'slope': slope, 'intercept': intercept,
            'r_squared': r**2, 'p_value': p
        }
    }

    # Regime-specific regressions on vol DIFFERENCE
    for mask, key in [(high_mask, 'high'), (low_mask, 'low')]:
        if mask.sum() > 15:
            s, i, r2, p2, _ = stats.linregress(
                analysis.loc[mask, 'implied_vol'],
                analysis.loc[mask, 'vol_diff']
            )
            results[key] = {'slope': s, 'intercept': i, 'r_squared': r2**2, 'p_value': p2}

    return results


# =============================================================================
# CHARTS
# =============================================================================
def plot_vol_timeseries(df: pd.DataFrame, symbol: str) -> go.Figure:
    """
    Volatility time series with regime band shading and current reading.
    """
    fig = go.Figure()

    # Vol history
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['implied_vol'] * 100,
        mode='lines',
        line=dict(color='#58a6ff', width=1.5),
        name='Realised Vol (Parkinson)',
        hovertemplate='%{x|%Y-%m-%d}<br>Vol: %{y:.2f}%<extra></extra>'
    ))

    # Percentile bands
    p25  = df['implied_vol'].quantile(0.25) * 100
    p75  = df['implied_vol'].quantile(0.75) * 100
    p80  = df['implied_vol'].quantile(0.80) * 100
    mean = df['implied_vol'].mean() * 100

    fig.add_hline(y=p80,  line_color='#f85149', line_width=1.5,
                  line_dash='dash', annotation_text='80th pct (SELL VOL)',
                  annotation_font_color='#f85149')
    fig.add_hline(y=p75,  line_color='#d29922', line_width=1,
                  line_dash='dot', annotation_text='75th pct',
                  annotation_font_color='#d29922')
    fig.add_hline(y=mean, line_color='#8b949e', line_width=1,
                  line_dash='dash', annotation_text='Mean',
                  annotation_font_color='#8b949e')
    fig.add_hline(y=p25,  line_color='#7ee787', line_width=1,
                  line_dash='dot', annotation_text='25th pct (BUY VOL)',
                  annotation_font_color='#7ee787')

    # Current reading
    current_vol = df['implied_vol'].iloc[-1] * 100
    fig.add_scatter(
        x=[df.index[-1]], y=[current_vol],
        mode='markers',
        marker=dict(color='#ffd700', size=10, symbol='star'),
        name=f'Current ({current_vol:.2f}%)'
    )

    fig.update_layout(
        title=f'{symbol} Volatility Time Series — Regime Band Analysis',
        xaxis_title='Date',
        yaxis_title='Annualised Volatility (%)',
        paper_bgcolor='#0d1117',
        plot_bgcolor='#161b22',
        font=dict(color='#c9d1d9'),
        legend=dict(bgcolor='#161b22', bordercolor='#30363d', borderwidth=1, font=dict(color='#c9d1d9', size=13)),
        height=420,
        margin=dict(l=60, r=120, t=50, b=50)
    )
    fig.update_xaxes(gridcolor='#30363d', showgrid=True)
    fig.update_yaxes(gridcolor='#30363d', showgrid=True)
    return fig


def plot_forward_regression(results: dict, symbol: str) -> go.Figure:
    """
    Two-panel regression chart.
    Left: forward vol vs current vol with y=x diagonal
    Right: vol difference by regime with separate regression lines
    """
    if not results:
        return go.Figure()

    df          = results['analysis_df']
    intersection = results['intersection']
    high_mask   = results['high_mask']
    low_mask    = results['low_mask']
    full        = results['full']

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            'Forward Vol vs Current Vol',
            'Vol Difference by Regime'
        ]
    )

    x_range = np.linspace(df['implied_vol'].min(), df['implied_vol'].max(), 100)

    # Panel 1: forward vs current
    fig.add_trace(go.Scatter(
        x=df['implied_vol'] * 100,
        y=df['forward_vol'] * 100,
        mode='markers',
        marker=dict(color='#58a6ff', size=4, opacity=0.5),
        name='Observations',
        hovertemplate='Current: %{x:.2f}%<br>Forward: %{y:.2f}%<extra></extra>'
    ), row=1, col=1)

    y_pred = full['slope'] * x_range + full['intercept']
    fig.add_trace(go.Scatter(
        x=x_range * 100,
        y=y_pred * 100,
        mode='lines',
        line=dict(color='#f85149', width=2),
        name=f"Regression (R²={full['r_squared']:.3f})"
    ), row=1, col=1)

    diag = [df[['implied_vol','forward_vol']].min().min(),
            df[['implied_vol','forward_vol']].max().max()]
    fig.add_trace(go.Scatter(
        x=[d*100 for d in diag], y=[d*100 for d in diag],
        mode='lines',
        line=dict(color='#8b949e', width=1, dash='dash'),
        name='y=x (no change)'
    ), row=1, col=1)

    # Panel 2: vol difference by regime
    fig.add_trace(go.Scatter(
        x=df.loc[high_mask, 'implied_vol'] * 100,
        y=df.loc[high_mask, 'vol_diff'] * 100,
        mode='markers',
        marker=dict(color='#f85149', size=4, opacity=0.5),
        name='High vol regime'
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=df.loc[low_mask, 'implied_vol'] * 100,
        y=df.loc[low_mask, 'vol_diff'] * 100,
        mode='markers',
        marker=dict(color='#7ee787', size=4, opacity=0.5),
        name='Low vol regime'
    ), row=1, col=2)

    for key, color in [('high', '#f85149'), ('low', '#7ee787')]:
        if key in results:
            r = results[key]
            mask = high_mask if key == 'high' else low_mask
            x_r = np.linspace(
                df.loc[mask, 'implied_vol'].min(),
                df.loc[mask, 'implied_vol'].max(), 100
            )
            fig.add_trace(go.Scatter(
                x=x_r * 100,
                y=(r['slope'] * x_r + r['intercept']) * 100,
                mode='lines',
                line=dict(color=color, width=2),
                name=f"{key.title()} R²={r['r_squared']:.3f}"
            ), row=1, col=2)

    fig.add_hline(y=0, line_color='#8b949e', line_width=1,
                  line_dash='dash', row=1, col=2)
    fig.add_vline(x=intersection*100, line_color='#ffd700', line_width=1.5,
                  line_dash='dot', row=1, col=2)

    fig.update_layout(
        height=400,
        paper_bgcolor='#0d1117',
        plot_bgcolor='#161b22',
        font=dict(color='#c9d1d9'),
        legend=dict(bgcolor='#161b22', bordercolor='#30363d', borderwidth=1, font=dict(color='#c9d1d9', size=13)),
        margin=dict(l=60, r=40, t=60, b=50)
    )
    fig.update_xaxes(gridcolor='#30363d', showgrid=True,
                     title_text='Current Vol (%)')
    fig.update_yaxes(gridcolor='#30363d', showgrid=True)
    fig.update_yaxes(title_text='Forward Vol (%)', row=1, col=1)
    fig.update_yaxes(title_text='Vol Difference (%)', row=1, col=2)
    return fig


def plot_percentile_history(df: pd.DataFrame) -> go.Figure:
    """IV percentile over time — shows regime transitions."""
    fig = go.Figure()

    pct = df['iv_percentile'] * 100

    # Colour each point by regime
    colors = []
    for p in pct:
        if p > 80:   colors.append('#f85149')
        elif p > 60: colors.append('#d29922')
        elif p > 40: colors.append('#c9d1d9')
        elif p > 20: colors.append('#58a6ff')
        else:        colors.append('#7ee787')

    fig.add_trace(go.Scatter(
        x=df.index,
        y=pct,
        mode='lines',
        line=dict(color='#58a6ff', width=1),
        fill='tozeroy',
        fillcolor='rgba(88,166,255,0.1)',
        name='IV Percentile',
        hovertemplate='%{x|%Y-%m-%d}<br>Percentile: %{y:.1f}%<extra></extra>'
    ))

    fig.add_hline(y=80, line_color='#f85149', line_width=1.5,
                  line_dash='dash', annotation_text='SELL VOL threshold (80th)',
                  annotation_font_color='#f85149')
    fig.add_hline(y=20, line_color='#7ee787', line_width=1.5,
                  line_dash='dash', annotation_text='BUY VOL threshold (20th)',
                  annotation_font_color='#7ee787')
    fig.add_hline(y=50, line_color='#8b949e', line_width=1,
                  line_dash='dot')

    fig.update_layout(
        title='IV Percentile Rank — Regime History',
        xaxis_title='Date',
        yaxis_title='Percentile (%)',
        paper_bgcolor='#0d1117',
        plot_bgcolor='#161b22',
        font=dict(color='#c9d1d9'),
        height=300,
        margin=dict(l=60, r=120, t=50, b=50),
        yaxis=dict(range=[0, 100])
    )
    fig.update_xaxes(gridcolor='#30363d', showgrid=True)
    fig.update_yaxes(gridcolor='#30363d', showgrid=True)
    return fig


# =============================================================================
# MAIN APP
# =============================================================================
def main():
    st.markdown("""
    <h1 style='font-family: JetBrains Mono, monospace; color: #58a6ff;'>
        📈 Implied Volatility Regime Analysis
    </h1>
    <p style='color: #8b949e; font-size: 0.95em;'>
        Regime-aware volatility analysis using rolling percentile classification
        and forward regression. Identifies when volatility is elevated or depressed
        relative to its own history and validates mean reversion statistically.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        symbol = st.text_input("Ticker Symbol", value="SPY").upper()

        period = st.selectbox(
            "History",
            ["1y", "2y", "3y", "5y"],
            index=1
        )

        forward_days = st.slider(
            "Forward window (days)", 5, 63, 21
        )

        st.markdown("---")
        st.markdown("### 📖 Methodology")
        st.markdown("""
        **Volatility measure**
        Parkinson estimator using
        High/Low range — more efficient
        than close-to-close returns.

        **Regime classification**
        Rolling 252-day percentile rank.
        Normalised across time — comparable
        across instruments and periods.

        **Mean reversion validation**
        Forward regression of future vol
        against current vol. Slope < 1
        confirms mean reversion. Regime
        split tests whether high vol
        reverts faster than low vol.

        **Trading implication**
        HIGH regime → sell options
        (collect elevated premium,
        expect vol to compress).
        LOW regime → buy options
        (cheap premium, expect expansion).
        """)

        st.markdown("---")
        st.markdown("### 🔗 Data")
        st.markdown("Yahoo Finance — free, no API key")

        if st.button("🔄 Refresh"):
            st.cache_data.clear()
            st.rerun()

    # Fetch data
    with st.spinner(f"Fetching {symbol} volatility data..."):
        df = fetch_volatility_data(symbol, period)

    if df.empty:
        st.error(f"No data found for {symbol}. Try SPY, QQQ, AAPL, MSFT.")
        return

    # Current metrics
    current_vol = df['implied_vol'].iloc[-1]
    current_pct = df['iv_percentile'].iloc[-1]
    prev_vol    = df['implied_vol'].iloc[-2]
    vol_change  = current_vol - prev_vol

    regime, signal, regime_color, signal_color = classify_regime(current_pct)

    # Metric row
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Current Vol",
            f"{current_vol*100:.2f}%",
            delta=f"{vol_change*100:.2f}%"
        )
    with col2:
        st.metric("IV Percentile", f"{current_pct*100:.1f}th")
    with col3:
        st.metric("52w High Vol",  f"{df['implied_vol'].tail(252).max()*100:.2f}%")
    with col4:
        st.metric("52w Low Vol",   f"{df['implied_vol'].tail(252).min()*100:.2f}%")
    with col5:
        st.metric("52w Mean Vol",  f"{df['implied_vol'].tail(252).mean()*100:.2f}%")

    st.markdown("---")

    # Signal box
    st.markdown(f"""
    <div class='signal-box'>
        <span style='color:#8b949e; font-size:0.85em;'>CURRENT REGIME & SIGNAL</span><br>
        <span style='color:{regime_color}; font-weight:bold; font-size:1.4em;'>{regime}</span>
        &nbsp;&nbsp;
        <span style='color:{signal_color}; font-weight:bold;'>{signal}</span>
        <br>
        <span style='color:#8b949e; font-size:0.85em;'>
        Percentile: {current_pct*100:.1f}th &nbsp;|&nbsp;
        {len(df)} trading days of history &nbsp;|&nbsp;
        Last updated: {df.index[-1].strftime('%Y-%m-%d')}
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Main chart
    st.plotly_chart(plot_vol_timeseries(df, symbol), use_container_width=True)

    # Percentile history
    st.plotly_chart(plot_percentile_history(df), use_container_width=True)

    # Forward regression
    st.markdown("### Mean Reversion Analysis")

    with st.spinner("Running forward regression..."):
        results = forward_regression_analysis(df, forward_days)

    if results:
        st.plotly_chart(
            plot_forward_regression(results, symbol),
            use_container_width=True
        )

        full = results['full']
        col_l, col_r = st.columns(2)

        with col_l:
            if full['slope'] < 1:
                st.markdown(f"""
                <div class='signal-box'>
                    ✅ <strong>Mean reversion confirmed</strong><br>
                    Regression slope = {full['slope']:.3f} (below 1 = reversion)<br>
                    R² = {full['r_squared']:.3f} &nbsp;|&nbsp;
                    p = {full['p_value']:.4f}<br>
                    <span style='color:#8b949e; font-size:0.85em;'>
                    High vol today predicts lower vol in {forward_days} days.
                    {"Statistically significant." if full['p_value'] < 0.05 else "Weak significance."}
                    </span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='signal-box'>
                    ⚠️ <strong>No clear mean reversion</strong><br>
                    Slope = {full['slope']:.3f} | R² = {full['r_squared']:.3f}
                </div>
                """, unsafe_allow_html=True)

        with col_r:
            if 'high' in results and 'low' in results:
                high_slope = results['high']['slope']
                low_slope  = results['low']['slope']
                st.markdown(f"""
                <div class='signal-box'>
                    <strong>Regime-dependent reversion</strong><br>
                    High vol regime slope: {high_slope:.3f}<br>
                    Low vol regime slope:  {low_slope:.3f}<br>
                    <span style='color:#8b949e; font-size:0.85em;'>
                    {"✅ High vol reverts faster — regime-aware strategy justified."
                      if high_slope < low_slope
                      else "⚠️ Similar reversion speed across regimes."}
                    </span>
                </div>
                """, unsafe_allow_html=True)

    # Regime history table
    st.markdown("### Regime Distribution")

    regime_counts = []
    for label, low, high in [
        ("HIGH VOL (>80th)",    0.80, 1.01),
        ("ABOVE AVG (60-80th)", 0.60, 0.80),
        ("NORMAL (40-60th)",    0.40, 0.60),
        ("BELOW AVG (20-40th)", 0.20, 0.40),
        ("LOW VOL (<20th)",     0.00, 0.20),
    ]:
        mask = (df['iv_percentile'] >= low) & (df['iv_percentile'] < high)
        count = mask.sum()
        mean_vol = df.loc[mask, 'implied_vol'].mean() * 100 if count > 0 else 0
        regime_counts.append({
            'Regime':         label,
            'Days':           count,
            'Pct of History': f"{count/len(df)*100:.1f}%",
            'Mean Vol':       f"{mean_vol:.2f}%"
        })

    import pandas as pd
    st.dataframe(
        pd.DataFrame(regime_counts),
        use_container_width=True,
        hide_index=True
    )

    # Raw data
    with st.expander("📊 Raw Data"):
        display = df[['Close', 'implied_vol', 'rv_21d', 'rv_63d',
                       'iv_percentile']].tail(60).copy()
        display.columns = ['Close', 'Parkinson Vol', '21d RV', '63d RV', 'Percentile']
        for col in ['Parkinson Vol', '21d RV', '63d RV']:
            display[col] = display[col].apply(lambda x: f"{x*100:.2f}%")
        display['Percentile'] = display['Percentile'].apply(lambda x: f"{x*100:.1f}th")
        st.dataframe(display, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='color:#8b949e; font-size:0.8em;'>
    <strong>Volatility measure:</strong> Parkinson estimator using daily High/Low range.
    More efficient than close-to-close returns — uses full intraday price information.
    Annualised by √252 convention.<br><br>
    <strong>Regime classification:</strong> Rolling 252-day percentile rank.
    Normalised signal — comparable across instruments and time periods.<br><br>
    <strong>Mean reversion validation:</strong> Forward regression of {forward_days}-day average
    future vol against current vol. Regime split at intersection of regression line with y=x diagonal.
    Negative slope in high-vol regime relative to low-vol regime confirms faster reversion when vol is elevated.<br><br>
    <strong>Data:</strong> Yahoo Finance via yfinance — free, no API key required.
    </div>
    """.format(forward_days=forward_days), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
