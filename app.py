"""
app.py
------
Quant Research Lab + Trading Terminal
Streamlit UI — tabs, sidebar controls, Plotly charts.

Run with:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from backtest import build_trade_log, run_backtest
from data import align_basket, fetch_basket, fetch_data
from events import (
    PLACEHOLDER_EVENTS,
    compute_event_impact,
    event_return_distribution,
    flag_events,
)
from indicators import add_all, add_bollinger_bands, add_sma
from metrics import compute_all
from regime import classify_regime, regime_summary
from strategies import (
    mean_reversion_signal,
    momentum_signal,
    regime_aware_signal,
    rotation_signal,
)
from walkforward import walk_forward

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Quant Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS — Institutional Deep Navy theme
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0F172A;
    color: #E5E7EB;
}
.block-container {
    padding: 1.5rem 2rem 2rem 2rem;
    max-width: 1400px;
}
[data-testid="stSidebar"] {
    background-color: #0F172A;
    border-right: 1px solid #1F2937;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }

/* Metric cards */
[data-testid="metric-container"] {
    background-color: #111827;
    border: 1px solid #1F2937;
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="metric-container"] label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #9CA3AF !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 22px;
    font-weight: 700;
    color: #E5E7EB !important;
}

/* Tabs */
[data-testid="stTabs"] [role="tablist"] { border-bottom: 1px solid #1F2937; gap: 4px; }
[data-testid="stTabs"] button[role="tab"] {
    font-size: 13px; font-weight: 500; color: #9CA3AF;
    border-radius: 6px 6px 0 0; padding: 8px 16px;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: #E5E7EB;
    border-bottom: 2px solid #3B82F6;
    background-color: #111827;
}

/* Buttons */
.stButton > button {
    background-color: #3B82F6; color: white; border: none;
    border-radius: 8px; font-weight: 600; padding: 8px 20px;
    font-size: 13px; transition: background 0.15s;
}
.stButton > button:hover { background-color: #38BDF8; color: #0F172A; }

hr { border-color: #1F2937; }
[data-testid="stDataFrame"] { border: 1px solid #1F2937; border-radius: 8px; }
[data-testid="stSelectbox"] > div,
[data-testid="stTextInput"] > div > input {
    background-color: #111827 !important;
    border-color: #1F2937 !important;
    color: #E5E7EB !important;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ── Color tokens ──────────────────────────────────────────────────────────────
# Background app:      #0F172A
# Background panel:    #111827
# Border:              #1F2937
# Text primary:        #E5E7EB
# Text secondary:      #9CA3AF
# Accent primary:      #3B82F6  (strategy line, buttons)
# Accent interactive:  #38BDF8  (hover)
# Positive:            #10B981
# Negative:            #F87171
# Warning / neutral:   #F59E0B
# Benchmark line:      #9CA3AF  (quiet, dotted)
# ─────────────────────────────────────────────────────────────────────────────

_PLOT_BG = "#0F172A"
_GRID    = "#1F2937"
_TEXT    = "#9CA3AF"
_TITLE   = "#E5E7EB"

REGIME_COLORS = {
    "Trending": "rgba(34,197,94,0.08)",
    "Choppy":   "rgba(234,179,8,0.08)",
    "HighVol":  "rgba(239,68,68,0.08)",
}
REGIME_LINE = {
    "Trending": "#10B981",
    "Choppy":   "#F59E0B",
    "HighVol":  "#F87171",
}
BADGE_CSS = {
    "Trending": "background:#10B981;color:#0F172A;padding:4px 14px;border-radius:6px;font-weight:700;font-size:13px;letter-spacing:0.05em",
    "Choppy":   "background:#F59E0B;color:#0F172A;padding:4px 14px;border-radius:6px;font-weight:700;font-size:13px;letter-spacing:0.05em",
    "HighVol":  "background:#F87171;color:#0F172A;padding:4px 14px;border-radius:6px;font-weight:700;font-size:13px;letter-spacing:0.05em",
}

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — Global Controls
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.title("⚙️ Terminal Controls")

ticker = st.sidebar.text_input("Ticker", value="QQQ").upper()

mode = st.sidebar.selectbox(
    "Mode",
    ["Backtest", "Walk-Forward", "Live Signal"],
    index=0,
)

col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Start", value=pd.Timestamp("2020-01-01"))
end_date   = col2.date_input("End",   value=pd.Timestamp("2025-01-01"))

st.sidebar.markdown("---")
st.sidebar.subheader("Transaction Costs")
commission_bps = st.sidebar.slider("Commission (bps)", 0, 20, 5)
slippage_bps   = st.sidebar.slider("Slippage (bps)",   0, 20, 2)

st.sidebar.markdown("---")
st.sidebar.subheader("Strategy")
strategy_type = st.sidebar.selectbox("Strategy", ["Momentum", "Mean Reversion", "Regime-Aware"])
fast_window   = st.sidebar.slider("Fast SMA", 5, 50, 20)
slow_window   = st.sidebar.slider("Slow SMA", 21, 200, 50)
bb_k          = st.sidebar.slider("Bollinger k", 0.5, 3.0, 2.0, step=0.1)
allow_short   = st.sidebar.checkbox("Allow Short", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("Display")
show_regime_shading = st.sidebar.checkbox("Regime shading on equity chart", value=True)
event_type_sel      = st.sidebar.selectbox("Event Type", list(PLACEHOLDER_EVENTS.keys()))

# ─────────────────────────────────────────────────────────────────────────────
# Data Loading (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Fetching market data…")
def load_data(t, s, e):
    return fetch_data(t, str(s), str(e))


@st.cache_data(show_spinner="Fetching basket data…")
def load_basket_cached(tickers, s, e):
    return fetch_basket(list(tickers), str(s), str(e))


df_raw = load_data(ticker, start_date, end_date)

if df_raw.empty:
    st.error(f"No data returned for {ticker}. Check the ticker and date range.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Build indicators + signal
# ─────────────────────────────────────────────────────────────────────────────

def build_df(raw: pd.DataFrame) -> pd.DataFrame:
    """Add all indicators, regime, events, and the chosen strategy signal."""
    d = raw.copy()
    d = add_all(d)
    d = add_sma(d, fast_window)
    d = add_sma(d, slow_window)
    d = add_bollinger_bands(d, 20, bb_k)
    d = classify_regime(d)
    d = flag_events(d, event_type=event_type_sel)

    if strategy_type == "Momentum":
        d = momentum_signal(d, fast_window, slow_window)
    elif strategy_type == "Mean Reversion":
        d = mean_reversion_signal(d, 20, bb_k, allow_short=allow_short)
    else:
        d = regime_aware_signal(d, fast_window, slow_window, 20, bb_k, allow_short=allow_short)

    return d


df = build_df(df_raw)

# ─────────────────────────────────────────────────────────────────────────────
# Run backtest (or skip in Live Signal mode)
# ─────────────────────────────────────────────────────────────────────────────

if mode != "Live Signal":
    try:
        df = run_backtest(
            df,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            allow_short=allow_short,
        )
        trade_log = build_trade_log(df)
        metrics   = compute_all(df, trade_log)
    except ValueError as e:
        st.error(f"Backtest error: {e}")
        st.stop()
else:
    trade_log = pd.DataFrame()
    metrics   = {}

reg_info = regime_summary(df)

# ─────────────────────────────────────────────────────────────────────────────
# Theme helpers
# ─────────────────────────────────────────────────────────────────────────────

def apply_plot_theme(fig: go.Figure, height: int = None) -> go.Figure:
    """Apply Institutional Deep Navy theme to any Plotly figure."""
    updates = dict(
        template="plotly_dark",
        plot_bgcolor=_PLOT_BG,
        paper_bgcolor=_PLOT_BG,
        font=dict(family="Inter, sans-serif", color=_TEXT, size=12),
        title_font=dict(color=_TITLE, size=14, family="Inter, sans-serif"),
        legend=dict(
            bgcolor="rgba(17,24,39,0.85)",
            bordercolor=_GRID,
            borderwidth=1,
            font=dict(color=_TEXT, size=11),
        ),
        xaxis=dict(
            gridcolor=_GRID,
            zerolinecolor=_GRID,
            tickfont=dict(color=_TEXT),
            title_font=dict(color=_TEXT),
        ),
        yaxis=dict(
            gridcolor=_GRID,
            zerolinecolor=_GRID,
            tickfont=dict(color=_TEXT),
            title_font=dict(color=_TEXT),
        ),
        margin=dict(l=48, r=24, t=48, b=40),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#111827",
            bordercolor=_GRID,
            font=dict(color=_TITLE, size=12),
        ),
    )
    if height:
        updates["height"] = height
    fig.update_layout(**updates)
    return fig


def render_metric_cards(m: dict) -> None:
    """Render a row of themed metric cards."""
    pairs = [
        ("CAGR",        m.get("CAGR"),               "%",  True),
        ("Sharpe",       m.get("Sharpe"),              "",   True),
        ("Sortino",      m.get("Sortino"),             "",   True),
        ("Max Drawdown", m.get("Max Drawdown"),        "%",  False),
        ("Total Return", m.get("Total Return"),        "%",  True),
        ("Turnover",     m.get("Turnover (avg)"),      "%",  None),
    ]
    cols = st.columns(len(pairs))
    for col, (label, val, unit, positive_is_good) in zip(cols, pairs):
        if val is None:
            col.metric(label, "—")
            continue
        display = f"{val}{unit}"
        if positive_is_good is True:
            delta = "▲" if float(val) > 0 else ("▼" if float(val) < 0 else None)
            delta_color = "normal"
        elif positive_is_good is False:
            delta = "▼" if float(val) < 0 else "▲"
            delta_color = "inverse"
        else:
            delta, delta_color = None, "off"
        col.metric(label, display, delta=delta, delta_color=delta_color)


def _add_regime_shading(fig: go.Figure, data: pd.DataFrame) -> go.Figure:
    """Add subtle regime background rectangles."""
    if "Regime" not in data.columns:
        return fig
    regime_col = data["Regime"]
    prev_regime, seg_start = None, None
    for date, regime in regime_col.items():
        if regime != prev_regime:
            if prev_regime is not None:
                fig.add_vrect(
                    x0=seg_start, x1=date,
                    fillcolor=REGIME_COLORS.get(prev_regime, "rgba(0,0,0,0)"),
                    layer="below", line_width=0,
                )
            seg_start, prev_regime = date, regime
    if prev_regime is not None and seg_start is not None:
        fig.add_vrect(
            x0=seg_start, x1=data.index[-1],
            fillcolor=REGIME_COLORS.get(prev_regime, "rgba(0,0,0,0)"),
            layer="below", line_width=0,
        )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Chart functions
# ─────────────────────────────────────────────────────────────────────────────

def equity_chart(result_df: pd.DataFrame, raw_df: pd.DataFrame, title: str = "Equity Curve") -> go.Figure:
    fig = go.Figure()
    if show_regime_shading:
        fig = _add_regime_shading(fig, result_df)

    bh = raw_df["Close"] / raw_df["Close"].iloc[0] * 100_000
    fig.add_trace(go.Scatter(
        x=bh.index, y=bh,
        name="Buy & Hold",
        line=dict(color="#9CA3AF", dash="dot", width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=result_df.index, y=result_df["Equity"],
        name="Strategy",
        line=dict(color="#3B82F6", width=3),
    ))

    apply_plot_theme(fig, height=420)
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def drawdown_chart(result_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result_df.index,
        y=result_df["Drawdown"],
        fill="tozeroy",
        name="Drawdown",
        line=dict(color="#F87171", width=1),
        fillcolor="rgba(248,113,113,0.2)",
    ))
    apply_plot_theme(fig, height=220)
    fig.update_layout(
        title="Drawdown",
        yaxis=dict(tickformat=".1%", gridcolor=_GRID, zerolinecolor=_GRID),
    )
    return fig


def regime_bar_chart(dist: dict) -> go.Figure:
    colors = [REGIME_LINE.get(r, "#9CA3AF") for r in dist]
    fig = go.Figure(go.Bar(
        x=list(dist.keys()),
        y=list(dist.values()),
        marker_color=colors,
        text=[f"{v:.1f}%" for v in dist.values()],
        textposition="outside",
        textfont=dict(color=_TITLE),
    ))
    apply_plot_theme(fig, height=260)
    fig.update_layout(
        title="Regime Distribution (%)",
        yaxis_title="%",
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab_home, tab_lab, tab_events, tab_portfolio, tab_trades = st.tabs([
    "🏠 Home Terminal",
    "🔬 Strategy Lab",
    "📅 Event Impact",
    "🗂 Portfolio",
    "📋 Trade Log",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Home Terminal
# ═══════════════════════════════════════════════════════════════════════════════

with tab_home:
    st.title(f"📈 {ticker}  ·  {strategy_type}  ·  {mode}")

    regime_now   = reg_info["current_regime"]
    signal_now   = int(df["Signal"].iloc[-1])
    signal_label = {1: "🟢 LONG", 0: "⚪ FLAT", -1: "🔴 SHORT"}.get(signal_now, "—")

    s1, s2, s3, s4 = st.columns(4)
    s1.markdown(
        f"**Regime**<br><span style='{BADGE_CSS.get(regime_now, '')}'>{regime_now}</span>",
        unsafe_allow_html=True,
    )
    s2.metric("Signal",         signal_label)
    s3.metric("Vol Percentile", f"{reg_info['current_vol_pct']}%")
    s4.metric("SMA Slope",      f"{reg_info['current_slope']}%")

    # ── Live Signal mode ────────────────────────────────────────────────────
    if mode == "Live Signal":
        st.markdown("---")
        st.subheader("📡 Live Signal Panel")

        fast_sma_col = f"SMA_{fast_window}"
        slow_sma_col = f"SMA_{slow_window}"
        last = df.iloc[-1]

        lcol, rcol = st.columns(2)
        with lcol:
            st.markdown("**Latest Indicator Values**")
            rows = {
                "Close":              round(last["Close"], 2),
                f"SMA {fast_window}": round(last[fast_sma_col], 2) if fast_sma_col in df.columns else "—",
                f"SMA {slow_window}": round(last[slow_sma_col], 2) if slow_sma_col in df.columns else "—",
                "BB Lower":           round(last.get("BB_LOWER_20", np.nan), 2),
                "BB Mid":             round(last.get("BB_MID_20", np.nan), 2),
                "BB Upper":           round(last.get("BB_UPPER_20", np.nan), 2),
                "Vol (ann.)":         f"{round(last.get('VOL_20', np.nan)*100, 1)}%",
            }
            st.table(pd.DataFrame.from_dict(rows, orient="index", columns=["Value"]))

        with rcol:
            st.markdown("**Signal Explanation**")
            expl = []
            if strategy_type == "Momentum":
                if fast_sma_col in df.columns and slow_sma_col in df.columns:
                    diff = last[fast_sma_col] - last[slow_sma_col]
                    direction = "above" if diff > 0 else "below"
                    expl.append(f"SMA{fast_window} is **{direction}** SMA{slow_window} by {abs(diff):.2f}")
                expl.append(f"→ Signal: **{signal_label}**")
            elif strategy_type == "Mean Reversion":
                if "BB_LOWER_20" in df.columns:
                    if last["Close"] < last["BB_LOWER_20"]:
                        expl.append("Price is **below lower band** → oversold entry condition met")
                    elif last["Close"] > last["BB_UPPER_20"]:
                        expl.append("Price is **above upper band** → overbought condition met")
                    else:
                        expl.append("Price is **within bands**")
                expl.append(f"→ Signal: **{signal_label}**")
            else:
                expl.append(f"Regime is **{regime_now}**")
                expl.append(f"→ Delegating to {'Momentum' if regime_now == 'Trending' else 'Mean Reversion' if regime_now == 'Choppy' else 'Flat (HighVol)'}")
                expl.append(f"→ Signal: **{signal_label}**")
            for line in expl:
                st.markdown(f"• {line}")

        st.info("Switch to **Backtest** or **Walk-Forward** mode to run a full simulation.")
        st.stop()

    # ── Backtest / Walk-Forward ──────────────────────────────────────────────
    if mode == "Walk-Forward":
        st.info("▶ Configure and run Walk-Forward in the **Strategy Lab** tab, then return here.")

    st.markdown("---")
    render_metric_cards(metrics)

    st.plotly_chart(equity_chart(df, df_raw), use_container_width=True)
    st.plotly_chart(drawdown_chart(df), use_container_width=True)
    st.plotly_chart(regime_bar_chart(reg_info["distribution"]), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Strategy Lab
# ═══════════════════════════════════════════════════════════════════════════════

with tab_lab:
    st.header("🔬 Strategy Lab")

    # ── Walk-Forward panel ───────────────────────────────────────────────────
    if mode == "Walk-Forward":
        st.subheader("Walk-Forward Optimisation")
        wf_col1, wf_col2 = st.columns(2)
        wf_train = wf_col1.slider("Training Window (days)", 63, 504, 252, key="wf_train")
        wf_test  = wf_col2.slider("Test Window (days)",     21, 126, 63,  key="wf_test")
        flatten_boundary = st.checkbox("Flatten position at fold boundaries", value=True)

        if st.button("▶ Run Walk-Forward", type="primary"):
            with st.spinner("Running walk-forward optimisation…"):
                oos_df, fold_log = walk_forward(
                    df_raw,
                    train_days=wf_train,
                    test_days=wf_test,
                    commission_bps=commission_bps,
                    slippage_bps=slippage_bps,
                    flatten_at_boundary=flatten_boundary,
                )

            if oos_df.empty:
                st.warning("Not enough data for the chosen window sizes.")
            else:
                st.success(f"Walk-forward complete: {len(fold_log)} folds")
                oos_metrics = compute_all(oos_df)
                render_metric_cards(oos_metrics)
                st.plotly_chart(equity_chart(oos_df, df_raw, "Walk-Forward OOS Equity"),
                                use_container_width=True)
                st.plotly_chart(drawdown_chart(oos_df), use_container_width=True)

                st.subheader("Fold Log")
                fold_df = pd.DataFrame(fold_log)
                st.dataframe(
                    fold_df.style
                    .background_gradient(subset=["OOS_Sharpe"], cmap="RdYlGn")
                    .format({"OOS_Sharpe": "{:.2f}"}),
                    use_container_width=True,
                )

    # ── Backtest / heatmap ───────────────────────────────────────────────────
    else:
        st.subheader("Signal vs Price")
        fig_sig = go.Figure()
        fig_sig.add_trace(go.Scatter(
            x=df.index, y=df["Close"], name="Price",
            line=dict(color="#3B82F6", width=1.5),
        ))
        fig_sig.add_trace(go.Bar(
            x=df.index, y=df["Signal"],
            name="Signal", yaxis="y2",
            marker_color=np.where(df["Signal"] > 0, "#10B981",
                         np.where(df["Signal"] < 0, "#F87171", "#4B5563")),
            opacity=0.5,
        ))
        apply_plot_theme(fig_sig)
        fig_sig.update_layout(
            yaxis=dict(title="Price"),
            yaxis2=dict(title="Signal", overlaying="y", side="right", range=[-2, 2],
                        gridcolor=_GRID, tickfont=dict(color=_TEXT)),
            height=380, title="Price + Signal",
        )
        st.plotly_chart(fig_sig, use_container_width=True)

        st.subheader("Parameter Sensitivity Heatmap (Momentum)")
        if st.button("▶ Run Heatmap"):
            fast_range = range(5, 35, 5)
            slow_range = range(20, 120, 10)
            heat = pd.DataFrame(index=list(fast_range), columns=list(slow_range), dtype=float)

            prog  = st.progress(0)
            total = len(list(fast_range)) * len(list(slow_range))
            count = 0

            for f in fast_range:
                for s in slow_range:
                    count += 1
                    prog.progress(count / total)
                    if f >= s:
                        heat.loc[f, s] = np.nan
                        continue
                    try:
                        tmp = df_raw.copy()
                        tmp = add_sma(tmp, f)
                        tmp = add_sma(tmp, s)
                        tmp = momentum_signal(tmp, f, s)
                        tmp = run_backtest(tmp, commission_bps=commission_bps,
                                           slippage_bps=slippage_bps, validate=False)
                        from metrics import sharpe_ratio as _sr
                        heat.loc[f, s] = round(_sr(tmp["Net_Return"]), 2)
                    except Exception:
                        heat.loc[f, s] = np.nan

            fig_heat = px.imshow(
                heat.astype(float),
                labels=dict(x="Slow SMA", y="Fast SMA", color="Sharpe"),
                title="Sharpe Ratio — Fast vs Slow SMA",
                color_continuous_scale="RdYlGn",
                aspect="auto",
                text_auto=".2f",
            )
            apply_plot_theme(fig_heat)
            fig_heat.update_layout(height=420)
            st.plotly_chart(fig_heat, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Event Impact
# ═══════════════════════════════════════════════════════════════════════════════

with tab_events:
    st.header("📅 Event Impact Analysis")
    st.caption(f"Event type: **{event_type_sel}** — configured in sidebar")

    fwd_days = st.multiselect(
        "Forward Return Windows (days)", [1, 3, 5, 10], default=[1, 3, 5]
    )

    if not fwd_days:
        st.warning("Select at least one forward window.")
    else:
        impact_df = compute_event_impact(df, forward_windows=fwd_days)

        st.subheader("Mean Forward Returns: Event vs Non-Event")
        fwd_cols = [c for c in impact_df.columns if c.startswith("Fwd")]
        st.dataframe(
            impact_df.style
            .format("{:.2%}", subset=fwd_cols)
            .format("{:.4f}", subset=["Avg_DayVol"])
            .background_gradient(subset=fwd_cols, cmap="RdYlGn", axis=None),
            use_container_width=True,
        )

        fwd_sel = st.selectbox("Distribution window", fwd_days, index=0)
        ev_ret, non_ev_ret = event_return_distribution(df, window=fwd_sel)

        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=ev_ret * 100, name=f"{event_type_sel} Days",
            opacity=0.75, marker_color="#F59E0B", nbinsx=30,
        ))
        fig_dist.add_trace(go.Histogram(
            x=non_ev_ret * 100, name="Non-Event Days",
            opacity=0.75, marker_color="#3B82F6", nbinsx=30,
        ))
        fig_dist.add_vline(x=0, line_color="#E5E7EB", line_dash="dash", line_width=1)
        apply_plot_theme(fig_dist, height=380)
        fig_dist.update_layout(
            barmode="overlay",
            title=f"{fwd_sel}-Day Forward Return Distribution (%)",
            xaxis_title="Return (%)", yaxis_title="Count",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        fig_ev = go.Figure()
        fig_ev.add_trace(go.Scatter(
            x=df.index, y=df["Close"], name="Close",
            line=dict(color="#9CA3AF", width=1.2),
        ))
        event_dates = df[df["IsEvent"]].index
        fig_ev.add_trace(go.Scatter(
            x=event_dates, y=df.loc[event_dates, "Close"],
            mode="markers", name=event_type_sel,
            marker=dict(color="#F59E0B", size=9, symbol="diamond"),
        ))
        apply_plot_theme(fig_ev, height=320)
        fig_ev.update_layout(title=f"{event_type_sel} Dates on Price Chart")
        st.plotly_chart(fig_ev, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Portfolio Rotation
# ═══════════════════════════════════════════════════════════════════════════════

with tab_portfolio:
    st.header("🗂 Rotation Portfolio")

    basket_tickers = st.multiselect(
        "Universe",
        ["QQQ", "SPY", "IWM", "TLT", "GLD", "XLK", "XLE"],
        default=["QQQ", "SPY", "IWM", "TLT"],
    )
    p1, p2 = st.columns(2)
    lookback_rot = p1.slider("Momentum Lookback (days)", 5, 60, 20)
    vol_filter   = p2.checkbox("Vol Filter → fallback to TLT", value=True)

    if st.button("▶ Run Rotation Backtest", type="primary"):
        if len(basket_tickers) < 2:
            st.warning("Select at least 2 tickers.")
        else:
            with st.spinner("Fetching basket data…"):
                basket = load_basket_cached(tuple(basket_tickers), start_date, end_date)
                prices = align_basket(basket)

            safe = "TLT" if "TLT" in basket_tickers else basket_tickers[0]
            selected = rotation_signal(
                prices, lookback=lookback_rot, vol_filter=vol_filter, safe_haven=safe
            )
            selected_shifted = selected.shift(1)

            ret_matrix  = prices.pct_change()
            rot_returns = pd.Series(index=prices.index, dtype=float)
            for date in prices.index:
                pick = selected_shifted.get(date)
                if pick and pick in ret_matrix.columns:
                    rot_returns[date] = ret_matrix.loc[date, pick]

            rot_equity = (1 + rot_returns.fillna(0)).cumprod() * 100_000

            BASKET_COLORS = ["#38BDF8", "#10B981", "#F59E0B", "#F87171",
                             "#A78BFA", "#FB923C", "#34D399"]
            fig_rot = go.Figure()
            fig_rot.add_trace(go.Scatter(
                x=rot_equity.index, y=rot_equity,
                name="Rotation", line=dict(color="#3B82F6", width=3),
            ))
            for i, t in enumerate(basket_tickers):
                bh = prices[t] / prices[t].iloc[0] * 100_000
                fig_rot.add_trace(go.Scatter(
                    x=bh.index, y=bh, name=t,
                    line=dict(dash="dot", width=1.5,
                              color=BASKET_COLORS[i % len(BASKET_COLORS)]),
                ))
            apply_plot_theme(fig_rot, height=420)
            fig_rot.update_layout(
                title="Rotation vs Buy & Hold",
                yaxis_title="Portfolio Value ($)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_rot, use_container_width=True)

            a1, a2 = st.columns(2)
            last_pick = selected.dropna().iloc[-1] if not selected.dropna().empty else "N/A"
            a1.metric("Current Allocation", last_pick)

            alloc_counts = selected.dropna().value_counts()
            fig_alloc = px.pie(
                values=alloc_counts.values,
                names=alloc_counts.index,
                title="Historical Allocation",
                color_discrete_sequence=BASKET_COLORS,
            )
            apply_plot_theme(fig_alloc, height=300)
            a2.plotly_chart(fig_alloc, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Trade Log
# ═══════════════════════════════════════════════════════════════════════════════

with tab_trades:
    st.header("📋 Trade Log")

    if mode == "Live Signal":
        st.info("Trade log is only available in Backtest or Walk-Forward mode.")
    elif trade_log.empty:
        st.info("No completed trades for the selected parameters and date range.")
    else:
        fl1, fl2, fl3 = st.columns(3)
        side_filter      = fl1.multiselect("Side", ["Long", "Short"], default=["Long", "Short"])
        min_ret, max_ret = fl2.slider("Return % range", -50.0, 100.0, (-50.0, 100.0))
        min_bars         = fl3.slider("Min bars held", 0, 200, 0)

        filtered = trade_log[
            trade_log["Side"].isin(side_filter) &
            trade_log["Return_pct"].between(min_ret, max_ret) &
            (trade_log.get("Bars_Held", pd.Series([0]*len(trade_log))) >= min_bars)
        ]

        st.caption(f"Showing {len(filtered)} of {len(trade_log)} trades")

        def colour_return(val):
            if val > 0:  return "color:#10B981;font-weight:600"
            if val < 0:  return "color:#F87171;font-weight:600"
            return "color:#9CA3AF"

        st.dataframe(
            filtered.style.applymap(colour_return, subset=["Return_pct"]),
            use_container_width=True,
            height=360,
        )

        st.markdown("---")
        tc1, tc2, tc3, tc4, tc5 = st.columns(5)
        tc1.metric("Total Trades",  len(filtered))
        tc2.metric("Win Rate",      f"{metrics.get('Win Rate', '—')}%")
        tc3.metric("Avg Win",       f"{metrics.get('Avg Win',  '—')}%")
        tc4.metric("Avg Loss",      f"{metrics.get('Avg Loss', '—')}%")
        tc5.metric("Avg Bars Held",
                   round(filtered["Bars_Held"].mean()) if "Bars_Held" in filtered.columns else "—")

        fig_ret = go.Figure()
        for side, color in [("Long", "#3B82F6"), ("Short", "#F87171")]:
            subset = filtered[filtered["Side"] == side]["Return_pct"]
            if not subset.empty:
                fig_ret.add_trace(go.Histogram(
                    x=subset, name=side,
                    marker_color=color, opacity=0.75, nbinsx=25,
                ))
        fig_ret.add_vline(x=0, line_color="#E5E7EB", line_dash="dash", line_width=1)
        apply_plot_theme(fig_ret, height=300)
        fig_ret.update_layout(
            barmode="overlay",
            title="Trade Return Distribution (%)",
            xaxis_title="Return (%)", yaxis_title="Count",
        )
        st.plotly_chart(fig_ret, use_container_width=True)
