from __future__ import annotations
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from core.data import (
    generate_data, get_state, ACCOUNTS, FX_RATES, CLEARING_DAYS, CHANNEL_RELIABILITY
)
from core.ml import CashFlowForecaster
from core.engine import RiskEngine, LiquidityOptimizer, compute_whatif, SEV_COLORS

st.set_page_config(
    page_title="LiquidityAI",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

components.html("""
<script>
(function () {
    var lastX = 0, lastY = 0, lastT = 0;
    var targetBody = window.parent.document.body;

    function spawnCoins(x, y, n) {
        for (var i = 0; i < n; i++) {
            (function () {
                var coin = window.parent.document.createElement('div');
                var dur  = 500 + Math.random() * 400;
                var tx   = (Math.random() > .5 ? 1 : -1) * (20 + Math.random() * 60);
                var ty   = 50 + Math.random() * 80;
                var rot  = (Math.random() * 720 - 360);

                coin.style.cssText = [
                    'position:fixed',
                    'left:' + x + 'px',
                    'top:'  + y + 'px',
                    'width:12px', 'height:12px',
                    'border-radius:50%',
                    'background:radial-gradient(circle at 35% 35%,#ffd700,#b8860b)',
                    'border:1px solid #ffa500',
                    'box-shadow:0 0 6px rgba(255,215,0,.7)',
                    'pointer-events:none',
                    'z-index:999999',
                    'opacity:1',
                    'transform:translate(0,0) rotate(0deg) scale(1)',
                    'transition:transform ' + dur + 'ms ease-out, opacity ' + dur + 'ms ease-out'
                ].join(';');

                targetBody.appendChild(coin);

                requestAnimationFrame(function () {
                    requestAnimationFrame(function () {
                        coin.style.transform = 'translate(' + tx + 'px,' + ty + 'px) rotate(' + rot + 'deg) scale(0.3)';
                        coin.style.opacity   = '0';
                    });
                });

                setTimeout(function () {
                    if (coin.parentNode) coin.parentNode.removeChild(coin);
                }, dur + 100);
            })();
        }
    }

    function onMove(e) {
        var x = e.clientX, y = e.clientY;
        var now = Date.now();
        var d = Math.hypot(x - lastX, y - lastY);
        if (now - lastT > 40 && d > 6) {
            lastT = now; lastX = x; lastY = y;
            spawnCoins(x, y, Math.min(3, 1 + Math.floor(d / 15)));
        }
    }

    window.parent.document.addEventListener('mousemove', onMove);
})();
</script>
""", height=0)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif;
}

.stApp { background-color: #0f172a; }
section[data-testid="stSidebar"] {
    background-color: #1e293b;
    border-right: 1px solid #334155;
}

[data-testid="stMetric"] {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="stMetric"] label {
    color: #94a3b8 !important;
    font-size: 12px !important;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
    font-weight: 700;
}

.page-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.02em;
}
.page-subtitle {
    font-size: 0.85rem;
    color: #64748b;
    margin-top: 4px;
}
.sidebar-logo {
    font-size: 1.1rem;
    font-weight: 700;
    color: #3b82f6;
    letter-spacing: -0.01em;
}

.card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 16px;
    margin: 4px 0;
}
.card-ok    { background: #0f2d1f; border: 1px solid #166534; border-radius: 12px; padding: 16px; margin: 4px 0; }
.card-warn  { background: #2d1f00; border: 1px solid #854d0e; border-radius: 12px; padding: 16px; margin: 4px 0; }
.card-bad   { background: #2d0f0f; border: 1px solid #991b1b; border-radius: 12px; padding: 16px; margin: 4px 0; }

.alert-critical { background: #1a0a0a; border-left: 3px solid #ef4444; padding: 14px 16px; border-radius: 8px; margin: 8px 0; }
.alert-high     { background: #1a1000; border-left: 3px solid #f97316; padding: 14px 16px; border-radius: 8px; margin: 8px 0; }
.alert-medium   { background: #1a1500; border-left: 3px solid #eab308; padding: 14px 16px; border-radius: 8px; margin: 8px 0; }
.alert-low      { background: #0a1a10; border-left: 3px solid #22c55e; padding: 14px 16px; border-radius: 8px; margin: 8px 0; }

.rec-card    { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 18px; margin: 10px 0; }
.rec-done    { background: #0f2d1f; border: 1px solid #166534; border-radius: 12px; padding: 18px; margin: 10px 0; }

.badge-urgent { background: #ef4444; color: #fff; font-size: 11px; font-weight: 600; padding: 3px 10px; border-radius: 20px; }
.badge-normal { background: #f59e0b; color: #000; font-size: 11px; font-weight: 600; padding: 3px 10px; border-radius: 20px; }

.factor-row {
    display: flex;
    justify-content: space-between;
    padding: 6px 12px;
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 6px;
    margin: 3px 0;
    font-size: 13px;
}
.progress-track { background: #0f172a; border-radius: 4px; height: 6px; margin: 4px 0; overflow: hidden; }

[data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 13px;
    font-weight: 500;
    color: #64748b !important;
}
[data-baseweb="tab"][aria-selected="true"] {
    color: #3b82f6 !important;
    border-bottom-color: #3b82f6 !important;
}

hr { border-color: #334155 !important; }

.kpi-row { display: flex; justify-content: space-between; padding: 8px 12px; background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; margin: 4px 0; }

/* ── BUTTON ANIMATIONS ─────────────────────────────────── */
.stButton > button {
    transition: all 0.22s cubic-bezier(0.4,0,0.2,1) !important;
    position: relative !important;
    overflow: hidden !important;
}
.stButton > button:hover {
    transform: translateY(-3px) scale(1.02) !important;
    box-shadow: 0 10px 28px rgba(59,130,246,0.45) !important;
    filter: brightness(1.12) !important;
}
.stButton > button:active {
    transform: translateY(0) scale(0.97) !important;
    box-shadow: 0 2px 8px rgba(59,130,246,0.25) !important;
    transition-duration: 0.08s !important;
}
/* Ripple on click */
.stButton > button::after {
    content: '';
    position: absolute;
    top: 50%; left: 50%;
    width: 6px; height: 6px;
    background: rgba(255,255,255,0.45);
    border-radius: 50%;
    opacity: 0;
    transform: scale(1) translate(-50%,-50%);
    transform-origin: 50% 50%;
}
.stButton > button:focus::after {
    animation: btn-ripple 0.55s ease-out !important;
}
@keyframes btn-ripple {
    0%   { transform: scale(0) translate(-50%,-50%); opacity: 0.5; }
    100% { transform: scale(38) translate(-50%,-50%); opacity: 0; }
}
/* Primary button — breathing glow */
.stButton > button[kind="primary"] {
    animation: btn-pulse 2.4s ease-in-out infinite !important;
}
@keyframes btn-pulse {
    0%,100% { box-shadow: 0 0 0 0 rgba(59,130,246,0.5); }
    50%      { box-shadow: 0 0 0 7px rgba(59,130,246,0); }
}
/* Form submit button — shimmer */
[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg,#1d4ed8,#3b82f6,#1d4ed8) !important;
    background-size: 200% auto !important;
    animation: btn-shimmer 2.8s linear infinite !important;
}
@keyframes btn-shimmer {
    0%   { background-position: 0% center; }
    100% { background-position: 200% center; }
}

/* ── METRIC CARDS ──────────────────────────────────────── */
[data-testid="stMetric"] {
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    animation: fade-up 0.45s ease-out both !important;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-4px) !important;
    box-shadow: 0 14px 32px rgba(59,130,246,0.22) !important;
}
@keyframes fade-up {
    from { opacity:0; transform:translateY(16px); }
    to   { opacity:1; transform:translateY(0); }
}

/* ── TABS ──────────────────────────────────────────────── */
[data-baseweb="tab"] {
    transition: color 0.18s ease, transform 0.18s ease !important;
}
[data-baseweb="tab"]:hover {
    transform: translateY(-1px) !important;
    color: #93c5fd !important;
}

/* ── CARDS (custom HTML) ───────────────────────────────── */
.card,.card-ok,.card-warn,.card-bad,.rec-card,.rec-done {
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}
.card:hover,.card-ok:hover,.card-warn:hover,.card-bad:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 10px 28px rgba(0,0,0,0.35) !important;
}

/* ── ALERT CARDS ───────────────────────────────────────── */
.alert-critical,.alert-high,.alert-medium,.alert-low {
    animation: slide-in 0.3s ease-out both !important;
    transition: transform 0.18s ease !important;
}
.alert-critical:hover,.alert-high:hover,.alert-medium:hover,.alert-low:hover {
    transform: translateX(4px) !important;
}
@keyframes slide-in {
    from { opacity:0; transform:translateX(-18px); }
    to   { opacity:1; transform:translateX(0); }
}

/* ── EXPANDERS ─────────────────────────────────────────── */
[data-testid="stExpander"] {
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
[data-testid="stExpander"]:hover {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 1px rgba(59,130,246,0.3) !important;
}

/* ── SELECT / DROPDOWN ─────────────────────────────────── */
[data-baseweb="select"] {
    transition: box-shadow 0.2s ease !important;
}
[data-baseweb="select"]:hover {
    box-shadow: 0 0 0 2px rgba(59,130,246,0.35) !important;
}

/* ── SLIDER THUMB ──────────────────────────────────────── */
[data-testid="stSlider"] [role="slider"] {
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
[data-testid="stSlider"] [role="slider"]:hover {
    transform: scale(1.35) !important;
    box-shadow: 0 0 0 6px rgba(59,130,246,0.25) !important;
}

/* ── INPUT FOCUS ───────────────────────────────────────── */
input:focus, textarea:focus {
    box-shadow: 0 0 0 2px rgba(59,130,246,0.4) !important;
    transition: box-shadow 0.2s ease !important;
}

/* ── PAGE TITLE ENTRANCE ───────────────────────────────── */
.page-title {
    animation: title-in 0.55s cubic-bezier(0.4,0,0.2,1) both !important;
}
@keyframes title-in {
    from { opacity:0; transform:translateY(-12px); }
    to   { opacity:1; transform:translateY(0); }
}

/* ── LOGO ──────────────────────────────────────────────── */
.liquidity-logo {
    display: block;
    margin-bottom: 4px;
    filter: drop-shadow(0 0 6px rgba(56,189,248,0.35));
    transition: filter 0.3s ease;
    animation: logo-glow 3.5s ease-in-out infinite;
}
.liquidity-logo:hover { filter: drop-shadow(0 0 14px rgba(56,189,248,0.75)); }
@keyframes logo-glow {
    0%,100% { filter: drop-shadow(0 0 4px rgba(56,189,248,0.3)); }
    50%      { filter: drop-shadow(0 0 12px rgba(56,189,248,0.7)); }
}

/* ── CHECKBOX ──────────────────────────────────────────── */
[data-testid="stCheckbox"]:hover label {
    color: #60a5fa !important;
    transition: color 0.2s ease !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
for k, v in {
    "user_accounts": [],
    "use_custom_data": False,
    "custom_fx": dict(FX_RATES),
    "confirmed_transfers": {},
    "demo_mode": False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Cached loaders ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data...")
def _load_df(accounts_json: str = "") -> pd.DataFrame:
    if accounts_json:
        return generate_data(months=12, accounts=json.loads(accounts_json))
    return generate_data(months=12)

@st.cache_resource(show_spinner="Initializing models...")
def _load_model(df_hash: int, accounts_json: str = "") -> CashFlowForecaster:
    df = _load_df(accounts_json)
    fc = CashFlowForecaster()
    fc.train(df)
    return fc

def _gen_accounts():
    if st.session_state.use_custom_data and st.session_state.user_accounts:
        return [{k: v for k, v in a.items() if k != "current_balance"}
                for a in st.session_state.user_accounts]
    return None

# ── Sidebar ────────────────────────────────────────────────────────────────────
LOGO_SVG = """
<svg class="liquidity-logo" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 58" width="170" height="42">
  <defs>
    <linearGradient id="wg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#38bdf8"/>
      <stop offset="100%" stop-color="#1d4ed8"/>
    </linearGradient>
    <linearGradient id="bg" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#7dd3fc"/>
      <stop offset="100%" stop-color="#0284c7"/>
    </linearGradient>
  </defs>
  <path d="M4,50 C14,35 26,46 36,41 C44,37 50,28 58,34 L58,56 C38,60 18,58 4,55 Z" fill="url(#wg)" opacity="0.88"/>
  <rect x="28" y="34" width="6" height="17" rx="2" fill="url(#bg)"/>
  <rect x="37" y="24" width="6" height="27" rx="2" fill="url(#bg)"/>
  <rect x="46" y="15" width="6" height="36" rx="2" fill="url(#bg)"/>
  <path d="M28,31 Q37,20 46,11 Q53,6 62,15" stroke="#38bdf8" stroke-width="2.2" fill="none" stroke-linecap="round"/>
  <text x="74" y="40" font-family="Inter,sans-serif" font-size="18" font-weight="500" fill="#94a3b8">Liquidity</text>
  <text x="178" y="40" font-family="Inter,sans-serif" font-size="18" font-weight="700" fill="#38bdf8">AI</text>
</svg>"""

with st.sidebar:
    st.markdown(LOGO_SVG, unsafe_allow_html=True)
    st.caption("Treasury Management System v2.0")
    st.divider()

    horizon = st.slider("Forecast horizon (days)", 1, 7, 3)

    st.markdown("**Currency filter**")
    c1, c2, c3 = st.columns(3)
    show_usd = c1.checkbox("USD", value=True)
    show_eur = c2.checkbox("EUR", value=True)
    show_gbp = c3.checkbox("GBP", value=True)
    currencies = [c for c, v in [("USD", show_usd), ("EUR", show_eur), ("GBP", show_gbp)] if v]

    st.divider()

    demo_label = "Stop Demo" if st.session_state.demo_mode else "Demo: SWIFT Outage"
    demo_type  = "secondary" if st.session_state.demo_mode else "primary"
    if st.button(demo_label, use_container_width=True, type=demo_type):
        st.session_state.demo_mode = not st.session_state.demo_mode
        st.session_state.confirmed_transfers = {}
        st.rerun()

    if st.button("Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    st.divider()

    if st.session_state.demo_mode:
        st.error("Demo active: SWIFT disconnected")
    if st.session_state.use_custom_data and st.session_state.user_accounts:
        st.success(f"Custom data: {len(st.session_state.user_accounts)} account(s)")
    else:
        st.caption("Source: synthetic data")

    st.caption("FinTech Hackathon 2026 | LiquidityAI Team")

# ── Load & compute ─────────────────────────────────────────────────────────────
_accs = _gen_accounts()
_acc_json = json.dumps(_accs, ensure_ascii=False, sort_keys=True) if _accs else ""
_fx = st.session_state.custom_fx

df      = _load_df(_acc_json)
model   = _load_model(hash(str(df.shape) + _acc_json), _acc_json)
state   = get_state(df)

if _accs:
    for acc in st.session_state.user_accounts:
        cb = float(acc.get("current_balance", 0))
        if cb > 0:
            m = state["account_id"] == acc["id"]
            if m.any():
                pi = state.loc[m, "pending_inflow"].values[0]
                state.loc[m, ["balance", "available_balance", "usd_equivalent",
                               "excess", "deficit"]] = [
                    cb, cb + pi, cb * _fx.get(acc["currency"], 1.0),
                    max(0.0, cb - acc["target_balance"]),
                    max(0.0, acc["min_balance"] - cb),
                ]

if st.session_state.demo_mode:
    for idx, row in state.iterrows():
        if row["payment_system"] == "SWIFT":
            nb = row["balance"] * 0.05
            state.loc[idx, "balance"]        = nb
            state.loc[idx, "usd_equivalent"] = nb * _fx.get(row["currency"], 1.0)
            state.loc[idx, "deficit"]        = max(0.0, row["min_balance"] - nb)
            state.loc[idx, "excess"]         = 0.0

for tid, rec in st.session_state.confirmed_transfers.items():
    fm = state["account_id"] == rec["from_id"]
    tm = state["account_id"] == rec["to_id"]
    if fm.any():
        nb = max(0.0, state.loc[fm, "balance"].values[0] - rec["amount"])
        state.loc[fm, "balance"]        = nb
        state.loc[fm, "usd_equivalent"] = nb * _fx.get(rec["currency_from"], 1.0)
        state.loc[fm, "deficit"]        = max(0.0, state.loc[fm, "min_balance"].values[0] - nb)
    if tm.any():
        nb = state.loc[tm, "balance"].values[0] + rec["amount_dest"]
        state.loc[tm, "balance"]        = nb
        state.loc[tm, "usd_equivalent"] = nb * _fx.get(rec["currency_to"], 1.0)
        state.loc[tm, "excess"]         = max(0.0, nb - state.loc[tm, "target_balance"].values[0])

state_f = state[state["currency"].isin(currencies)].copy()
balances = dict(zip(state["account_id"], state["balance"]))
forecasts = model.forecast_all(days=horizon, current_balances=balances)
fc_f = forecasts[forecasts["currency"].isin(currencies)] if not forecasts.empty else pd.DataFrame()

risk_engine = RiskEngine()
alerts      = risk_engine.generate_alerts(state_f, fc_f)
alert_sum   = risk_engine.summary(alerts)
optimizer   = LiquidityOptimizer()
recs        = optimizer.recommend(state_f, fc_f)
idle        = optimizer.idle_report(state_f)

# ── Header ─────────────────────────────────────────────────────────────────────
LOGO_SVG_LG = """
<svg class="liquidity-logo" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 58" width="220" height="54">
  <defs>
    <linearGradient id="wg2" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#38bdf8"/>
      <stop offset="100%" stop-color="#1d4ed8"/>
    </linearGradient>
    <linearGradient id="bg2" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#7dd3fc"/>
      <stop offset="100%" stop-color="#0284c7"/>
    </linearGradient>
  </defs>
  <path d="M4,50 C14,35 26,46 36,41 C44,37 50,28 58,34 L58,56 C38,60 18,58 4,55 Z" fill="url(#wg2)" opacity="0.88"/>
  <rect x="28" y="34" width="6" height="17" rx="2" fill="url(#bg2)"/>
  <rect x="37" y="24" width="6" height="27" rx="2" fill="url(#bg2)"/>
  <rect x="46" y="15" width="6" height="36" rx="2" fill="url(#bg2)"/>
  <path d="M28,31 Q37,20 46,11 Q53,6 62,15" stroke="#38bdf8" stroke-width="2.2" fill="none" stroke-linecap="round"/>
  <text x="74" y="40" font-family="Inter,sans-serif" font-size="18" font-weight="500" fill="#94a3b8">Liquidity</text>
  <text x="178" y="40" font-family="Inter,sans-serif" font-size="18" font-weight="700" fill="#38bdf8">AI</text>
</svg>"""

if st.session_state.demo_mode:
    st.markdown(LOGO_SVG_LG + '<div class="page-title" style="color:#ef4444;margin-top:8px">Critical — SWIFT Outage Detected</div>', unsafe_allow_html=True)
    st.error("System detected: liquidity deficit risk within 48h on SWIFT accounts. Go to Rebalancing to confirm transfers.")
else:
    st.markdown(LOGO_SVG_LG + '<div class="page-title" style="margin-top:8px">LiquidityAI — Treasury Management</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-subtitle">{df["date"].max().strftime("%d %B %Y")} &nbsp;|&nbsp; {len(state_f)} accounts &nbsp;|&nbsp; Horizon: {horizon} days</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

total_liq = state_f["usd_equivalent"].sum()
frozen    = idle["total_idle_usd"]
opp_cost  = idle["annual_opp_cost_usd"]
crit_cnt  = alert_sum.get("CRITICAL", 0) + alert_sum.get("HIGH", 0)
pend_usd  = (state_f["pending_inflow"] * state_f["currency"].map(_fx)).sum()
efficiency= (total_liq - frozen) / total_liq * 100 if total_liq else 0

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total Liquidity",    f"${total_liq/1e6:.2f}M")
k2.metric("Efficiency",         f"{efficiency:.1f}%",
          delta=f"{efficiency-75:.1f}% vs norm",
          delta_color="normal" if efficiency >= 75 else "inverse")
k3.metric("Critical Alerts",    str(crit_cnt),
          delta=f"+{crit_cnt}" if crit_cnt else None,
          delta_color="inverse" if crit_cnt else "off")
k4.metric("Idle Capital",       f"${frozen/1e6:.2f}M")
k5.metric("Lost Revenue / yr",  f"${opp_cost/1e3:.0f}K")
k6.metric("Pending Clearing",   f"${pend_usd/1e6:.2f}M")

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#cbd5e1",
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
)

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Dashboard", "Forecast", "Risk Engine",
    "Rebalancing", "What-If", "Analytics", "Data", "Business Model",
])

# ── TAB 1: Dashboard ───────────────────────────────────────────────────────────
with tab1:
    st.subheader("Account Overview")

    cols3 = st.columns(3)
    for i, (_, row) in enumerate(state_f.iterrows()):
        bal, min_b, tgt_b = row["balance"], row["min_balance"], row["target_balance"]
        if bal < min_b:
            css, status, sc = "card-bad",  "Deficit", "#ef4444"
        elif bal < tgt_b * 0.8:
            css, status, sc = "card-warn", "Warning", "#f59e0b"
        else:
            css, status, sc = "card-ok",   "Healthy", "#22c55e"
        fill = min(100, bal / tgt_b * 100)
        bar_c = "#22c55e" if fill >= 80 else "#f59e0b" if fill >= 50 else "#ef4444"
        with cols3[i % 3]:
            st.markdown(f"""
<div class="{css}">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
    <span style="font-size:12px;color:#64748b;font-weight:500">{row['payment_system']} &middot; {row['currency']}</span>
    <span style="font-size:12px;font-weight:600;color:{sc}">{status}</span>
  </div>
  <div style="font-weight:600;font-size:14px;color:#f1f5f9;margin-bottom:2px">{row['account_name']}</div>
  <div style="font-size:26px;font-weight:700;color:#f1f5f9;margin:8px 0">{bal:,.0f} <span style="font-size:14px;color:#64748b;font-weight:400">{row['currency']}</span></div>
  <div class="progress-track"><div style="width:{fill:.0f}%;background:{bar_c};height:6px;border-radius:4px"></div></div>
  <div style="display:flex;justify-content:space-between;margin-top:6px;font-size:11px;color:#64748b">
    <span>Min {min_b:,.0f}</span>
    <span style="color:#94a3b8">{fill:.0f}% of target</span>
    <span>Tgt {tgt_b:,.0f}</span>
  </div>
  {'<div style="font-size:11px;color:#60a5fa;margin-top:6px">Clearing: '+f"{row['pending_inflow']:,.0f}"+'</div>' if row['pending_inflow'] > 0 else ''}
</div>""", unsafe_allow_html=True)

    st.divider()
    left, right = st.columns([3, 1])
    with left:
        st.markdown("#### Liquidity trend — 90 days")
        h90 = df[df["date"] >= df["date"].max() - pd.Timedelta(days=90)]
        h90 = h90[h90["currency"].isin(currencies)]
        daily = h90.groupby("date").apply(
            lambda g: (g["balance"] * g["currency"].map(_fx)).sum()
        ).reset_index(name="usd")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily["date"], y=daily["usd"],
            fill="tozeroy",
            line=dict(color="#3b82f6", width=2),
            fillcolor="rgba(59,130,246,0.08)",
            name="Liquidity",
        ))
        fig.update_layout(height=230, margin=dict(t=10, b=20),
                          xaxis_title="", yaxis_title="USD", **PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("#### By currency")
        ccy = state_f.groupby("currency")["usd_equivalent"].sum().reset_index()
        fig2 = px.pie(ccy, values="usd_equivalent", names="currency", hole=0.55,
                      color_discrete_map={"USD":"#3b82f6","EUR":"#22c55e","GBP":"#f59e0b"})
        fig2.update_layout(height=230, paper_bgcolor="rgba(0,0,0,0)",
                           font_color="#cbd5e1", margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("**Alert summary**")
        sev_cfg = {"CRITICAL":"#ef4444","HIGH":"#f97316","MEDIUM":"#eab308","LOW":"#22c55e"}
        for sev, color in sev_cfg.items():
            cnt = alert_sum.get(sev, 0)
            st.markdown(f'<div style="display:flex;justify-content:space-between;padding:6px 12px;background:#0f172a;border-left:3px solid {color};border-radius:6px;margin:3px 0"><span style="color:{color};font-size:12px;font-weight:500">{sev}</span><strong style="color:#f1f5f9">{cnt}</strong></div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("#### Channel Reliability")
    ch_cols = st.columns(4)
    for i, (ch, rel) in enumerate(CHANNEL_RELIABILITY.items()):
        color = "#22c55e" if rel >= 0.98 else "#f59e0b" if rel >= 0.95 else "#ef4444"
        score = int(rel * 100)
        with ch_cols[i]:
            st.markdown(f"""
<div style="background:#1e293b;border:1px solid #334155;border-radius:12px;padding:16px;text-align:center">
  <div style="font-size:12px;color:#64748b;font-weight:500;margin-bottom:4px">{ch}</div>
  <div style="font-size:28px;font-weight:700;color:{color};margin:8px 0">{score}%</div>
  <div class="progress-track"><div style="width:{score}%;background:{color};height:6px;border-radius:4px"></div></div>
  <div style="font-size:11px;color:#64748b;margin-top:6px">Reliability</div>
</div>""", unsafe_allow_html=True)


# ── TAB 2: Forecast ────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Cash Flow Forecast — quantile scenarios")

    if fc_f.empty:
        st.warning("No forecast data. Select at least one currency.")
    else:
        sel_acc = st.selectbox(
            "Account",
            options=state_f["account_id"].tolist(),
            format_func=lambda x: state_f[state_f["account_id"]==x]["account_name"].iloc[0],
        )
        acc_fc    = fc_f[fc_f["account_id"] == sel_acc]
        acc_state = state_f[state_f["account_id"] == sel_acc].iloc[0]
        hist14    = df[(df["account_id"]==sel_acc) & (df["date"] >= df["date"].max()-pd.Timedelta(days=14))]

        p_short = model.p_shortage(sel_acc, acc_state["min_balance"], fc_f)
        p_color = "#ef4444" if p_short > 0.3 else "#f59e0b" if p_short > 0.1 else "#22c55e"

        pm1, pm2, pm3, pm4 = st.columns(4)
        pm1.metric("Current balance",  f"{acc_state['balance']:,.0f} {acc_state['currency']}")
        pm2.metric("Forecast (q50)",   f"{acc_fc.iloc[-1]['q50']:,.0f}" if not acc_fc.empty else "—")
        pm3.metric("P(deficit)",       f"{p_short*100:.1f}%",
                   delta="High risk" if p_short > 0.3 else "OK",
                   delta_color="inverse" if p_short > 0.3 else "normal")
        pm4.metric("Horizon",          f"{horizon} days")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist14["date"], y=hist14["balance"],
                                  name="Actual", line=dict(color="#60a5fa", width=2)))
        if not acc_fc.empty:
            fig.add_trace(go.Scatter(x=acc_fc["date"], y=acc_fc["q90"],
                                      name="q90 Optimistic", line=dict(color="rgba(34,197,94,0)"),
                                      fillcolor="rgba(34,197,94,0.10)", fill="tonexty",
                                      showlegend=True, legendgroup="ci"))
            fig.add_trace(go.Scatter(x=acc_fc["date"], y=acc_fc["q10"],
                                      name="q10 Pessimistic", line=dict(color="rgba(239,68,68,0)"),
                                      showlegend=True, legendgroup="ci"))
            fig.add_trace(go.Scatter(x=acc_fc["date"], y=acc_fc["q50"],
                                      name="q50 Expected", mode="lines+markers",
                                      line=dict(color="#f59e0b", width=2, dash="dash")))

        fig.add_hline(y=acc_state["min_balance"],    line_dash="dash", line_color="#ef4444", annotation_text="MIN")
        fig.add_hline(y=acc_state["target_balance"], line_dash="dot",  line_color="#22c55e", annotation_text="TGT")
        fig.add_vline(x=df["date"].max().timestamp()*1000, line_color="#475569", annotation_text="TODAY")
        fig.update_layout(height=380, hovermode="x unified",
                          legend=dict(orientation="h"),
                          xaxis_title="", yaxis_title=acc_state["currency"],
                          margin=dict(t=10), **PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

        if not acc_fc.empty:
            st.markdown("#### Forecast table")
            tbl = acc_fc[["date","q10","q50","q90","predicted_inflow","predicted_outflow"]].copy()
            tbl.columns = ["Date","q10 Pessim.","q50 Expected","q90 Optimistic","Inflow","Outflow"]
            for c in tbl.columns[1:]:
                tbl[c] = tbl[c].apply(lambda x: f"{x:,.0f}")
            st.dataframe(tbl, use_container_width=True, hide_index=True)


# ── TAB 3: Risk Engine ─────────────────────────────────────────────────────────
with tab3:
    st.subheader("Risk Engine — deficit probability")

    risk_rows = []
    for _, row in state_f.iterrows():
        p = model.p_shortage(row["account_id"], row["min_balance"], fc_f)
        risk_rows.append({
            "Account": row["account_name"],
            "Currency": row["currency"],
            "System": row["payment_system"],
            "P(deficit)": p,
            "Level": "CRITICAL" if p > 0.3 else "HIGH" if p > 0.15 else "MEDIUM" if p > 0.05 else "LOW",
        })
    risk_df = pd.DataFrame(risk_rows)

    COLOR_MAP = {"CRITICAL":"#ef4444","HIGH":"#f97316","MEDIUM":"#eab308","LOW":"#22c55e"}
    fig_risk = go.Figure()
    colors_p = [COLOR_MAP["CRITICAL"] if p > 0.3 else COLOR_MAP["HIGH"] if p > 0.15
                else COLOR_MAP["MEDIUM"] if p > 0.05 else COLOR_MAP["LOW"]
                for p in risk_df["P(deficit)"]]
    fig_risk.add_trace(go.Bar(
        x=risk_df["Account"], y=risk_df["P(deficit)"] * 100,
        marker_color=colors_p,
        text=[f"{v*100:.1f}%" for v in risk_df["P(deficit)"]],
        textposition="outside",
    ))
    fig_risk.add_hline(y=30, line_dash="dash", line_color="#ef4444", annotation_text="Critical (30%)")
    fig_risk.add_hline(y=10, line_dash="dot",  line_color="#f59e0b", annotation_text="High (10%)")
    fig_risk.update_layout(height=300, yaxis_title="P(deficit) %", xaxis_title="",
                           xaxis=dict(tickangle=-20, gridcolor="rgba(255,255,255,0.05)"),
                           yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                           margin=dict(t=20), **{k: v for k, v in PLOT_LAYOUT.items() if k not in ("xaxis","yaxis")})
    st.plotly_chart(fig_risk, use_container_width=True)

    st.divider()
    st.markdown(f"#### Active Alerts ({len(alerts)})")

    if not alerts:
        st.success("No active alerts. All accounts are within normal parameters.")
    else:
        sev_cols = st.columns(4)
        for i, s in enumerate(["CRITICAL","HIGH","MEDIUM","LOW"]):
            sev_cols[i].metric(s, alert_sum.get(s, 0))
        st.divider()

        for al in alerts:
            sev = al["severity"]
            css = f"alert-{sev.lower()}"
            color = COLOR_MAP[sev]
            expl  = al.get("explanation", {})
            factors = expl.get("factors", [])
            risk_pct = expl.get("total_risk_pct", 0)
            time_str = f"{al['time_to_breach_h']}h" if al.get("time_to_breach_h") else "Now"

            st.markdown(f"""
<div class="{css}">
  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px">
    <strong style="color:{color}">{sev} &mdash; {al['type']}</strong>
    <span style="color:#64748b;font-size:12px">T-{time_str} &middot; {al['account_name']} ({al['currency']})</span>
  </div>
  <div style="margin-top:8px;color:#cbd5e1">{al['message']}</div>
  <div style="margin-top:4px;font-size:12px;color:#64748b">{al['action']}</div>
</div>""", unsafe_allow_html=True)

            if factors:
                with st.expander(f"Factor breakdown — risk +{risk_pct}%"):
                    for fname, fval in factors:
                        w = min(100, fval * 3)
                        fc_c = "#ef4444" if fval >= 20 else "#f59e0b" if fval >= 10 else "#22c55e"
                        st.markdown(f"""
<div class="factor-row">
  <span style="color:#cbd5e1">{fname}</span>
  <span style="color:{fc_c};font-weight:600">+{fval}%</span>
</div>
<div class="progress-track"><div style="width:{w}%;background:{fc_c};height:6px;border-radius:4px"></div></div>
""", unsafe_allow_html=True)
                    st.markdown(f"**Total factor contribution: +{risk_pct}% to deficit risk**")


# ── TAB 4: Rebalancing ─────────────────────────────────────────────────────────
with tab4:
    st.subheader("Rebalancing — liquidity optimization")

    ir1, ir2, ir3 = st.columns(3)
    ir1.metric("Total liquidity",   f"${idle['total_liquidity_usd']/1e6:.2f}M")
    ir2.metric("Idle capital",      f"${idle['total_idle_usd']/1e6:.2f}M",
               f"{idle['idle_pct']:.1f}% of total")
    ir3.metric("Lost revenue / yr", f"${idle['annual_opp_cost_usd']/1e3:.0f}K",
               "at 4.5% annual rate")

    st.divider()

    # Разделяем на срочные переводы и idle-оптимизацию
    urgent_recs = [r for r in recs if r.get("type") != "IDLE_OPTIMIZATION"]
    idle_recs   = [r for r in recs if r.get("type") == "IDLE_OPTIMIZATION"]

    if not urgent_recs and not idle_recs:
        st.success("All accounts balanced. No rebalancing required.")
    
    # ── Срочные переводы между счетами ──
    if urgent_recs:
        confirmed_cnt = sum(1 for r in urgent_recs if r["id"] in st.session_state.confirmed_transfers)
        st.markdown(f"### 🔴 Transfer recommendations")
        st.markdown(f"**{len(urgent_recs)} transfers** &nbsp;|&nbsp; Confirmed: {confirmed_cnt} / {len(urgent_recs)}")

        for i, rec in enumerate(urgent_recs):
            is_confirmed = rec["id"] in st.session_state.confirmed_transfers
            card_css = "rec-done" if is_confirmed else "rec-card"
            urg_css  = "badge-urgent" if rec["urgency"] == "НЕМЕДЛЕННО" else "badge-normal"
            fx_note  = f'= {rec["amount_dest"]:,.0f} {rec["currency_to"]}' if rec["currency_from"] != rec["currency_to"] else ""

            st.markdown(f"""
<div class="{card_css}">
  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">
    <strong style="font-size:14px;color:#f1f5f9">{rec['from_account']} &rarr; {rec['to_account']}</strong>
    <span class="{urg_css}">{rec['urgency']}</span>
  </div>
  <div style="font-size:22px;font-weight:700;margin:12px 0;color:#3b82f6">
    {rec['amount']:,.0f} {rec['currency_from']} <span style="font-size:14px;color:#64748b;font-weight:400">{fx_note}</span>
  </div>
  <div style="color:#64748b;font-size:12px">
    Transfer time: {rec['transfer_time_days']}d &nbsp;&middot;&nbsp; Cost: ~{rec['estimated_cost']:,.0f} {rec['currency_from']} ({rec['cost_bps']} bps)
  </div>
  <div style="margin-top:6px;font-size:13px;color:#94a3b8">{rec['reason']}</div>
</div>""", unsafe_allow_html=True)

            if not is_confirmed:
                btn_col, _ = st.columns([2, 5])
                if btn_col.button(f"Confirm Transfer #{i+1}", key=f"confirm_{i}", type="primary"):
                    st.session_state.confirmed_transfers[rec["id"]] = rec
                    st.cache_data.clear()
                    st.rerun()
            else:
                st.success("Transfer confirmed — execution in progress")

    # ── Idle capital оптимизация ──
    if idle_recs:
        if urgent_recs:
            st.divider()
        st.markdown("### 💰 Idle capital optimization")
        total_idle_annual = sum(
            r["amount"] * FX_RATES[r["currency_from"]] * 0.045
            for r in idle_recs
        )
        st.info(f"**{len(idle_recs)} accounts** have idle capital above target balance. "
                f"Total potential income: **${total_idle_annual:,.0f}/yr** at 4.5% annual rate.")

        for i, rec in enumerate(idle_recs):
            is_confirmed = rec["id"] in st.session_state.confirmed_transfers
            card_css = "rec-done" if is_confirmed else "rec-card"
            annual = rec["amount"] * FX_RATES[rec["currency_from"]] * 0.045

            st.markdown(f"""
<div class="{card_css}" style="border-left: 3px solid #22c55e;">
  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">
    <strong style="font-size:14px;color:#f1f5f9">{rec['from_account']} &rarr; {rec['to_account']}</strong>
    <span style="background:#14532d;color:#86efac;padding:3px 10px;border-radius:12px;font-size:11px;font-weight:600">ОПТИМИЗАЦИЯ</span>
  </div>
  <div style="font-size:22px;font-weight:700;margin:12px 0;color:#22c55e">
    {rec['amount']:,.0f} {rec['currency_from']}
    <span style="font-size:13px;color:#64748b;font-weight:400;margin-left:12px">+${annual:,.0f}/yr income</span>
  </div>
  <div style="color:#64748b;font-size:12px">
    Transfer time: {rec['transfer_time_days']}d &nbsp;&middot;&nbsp; Cost: ~{rec['cost_bps']} bps
  </div>
  <div style="margin-top:6px;font-size:13px;color:#94a3b8">{rec['reason']}</div>
</div>""", unsafe_allow_html=True)

            if not is_confirmed:
                btn_col, _ = st.columns([2, 5])
                if btn_col.button(f"Place in Money Market #{i+1}", key=f"idle_{i}", type="primary"):
                    st.session_state.confirmed_transfers[rec["id"]] = rec
                    st.cache_data.clear()
                    st.rerun()
            else:
                st.success("Placement confirmed — funds allocated to money market")


# ── TAB 5: What-If ─────────────────────────────────────────────────────────────
with tab5:
    st.subheader("What-If Simulator")

    col_ctrl, col_res = st.columns([1, 2], gap="large")

    with col_ctrl:
        st.markdown("#### Scenario parameters")
        swift_off    = st.toggle("Disable SWIFT", value=False)
        sepa_delay   = st.slider("SEPA extra delay (days)", 0, 5, 0)
        card_delay   = st.slider("Card extra delay (days)", 0, 5, 0)
        vol_spike    = st.slider("Outflow volume spike (%)", 0, 200, 0)
        eur_shock    = st.slider("EUR/USD shock (%)", -20, 20, 0)
        gbp_shock    = st.slider("GBP/USD shock (%)", -20, 20, 0)
        outage_names = st.multiselect("Banking outage (accounts)", state_f["account_name"].tolist())
        outage_ids   = state_f[state_f["account_name"].isin(outage_names)]["account_id"].tolist()

        scenario_active = any([swift_off, sepa_delay, card_delay, vol_spike,
                                eur_shock != 0, gbp_shock != 0, outage_ids])
        if scenario_active:
            st.info("Results update in real time")
        else:
            st.caption("Adjust parameters to run a simulation")

    wi_df = compute_whatif(state_f, fc_f, {
        "swift_disabled":   swift_off,
        "sepa_extra_delay": sepa_delay,
        "card_extra_delay": card_delay,
        "volume_spike_pct": vol_spike,
        "eur_shock_pct":    eur_shock,
        "gbp_shock_pct":    gbp_shock,
        "outage_accounts":  outage_ids,
    })

    with col_res:
        if wi_df.empty:
            st.info("No simulation data available.")
        else:
            total_delta = wi_df["delta_usd"].sum()
            at_risk_cnt = int(wi_df["at_risk"].sum())
            shortfall   = wi_df["shortfall_usd"].sum()
            sev_label   = ("Critical" if at_risk_cnt >= 3 or shortfall > 2e6
                           else "High"   if at_risk_cnt >= 2 or shortfall > 5e5
                           else "Medium" if at_risk_cnt >= 1
                           else "Low")
            sev_color   = ("#ef4444" if sev_label == "Critical" else
                           "#f59e0b" if sev_label in ("High","Medium") else "#22c55e")

            w1, w2, w3, w4 = st.columns(4)
            w1.metric("Impact (USD)",       f"${total_delta/1e6:.2f}M", delta_color="inverse" if total_delta < 0 else "normal")
            w2.metric("Accounts at risk",   str(at_risk_cnt))
            w3.metric("Shortfall (USD)",    f"${shortfall/1e3:.0f}K")
            w4.metric("Risk level",         sev_label)

            fig_wi = go.Figure()
            fig_wi.add_trace(go.Bar(name="Base forecast",   x=wi_df["account_name"],
                                    y=wi_df["base_end"], marker_color="#3b82f6"))
            fig_wi.add_trace(go.Bar(name="Stress scenario", x=wi_df["account_name"],
                                    y=wi_df["stressed_end"],
                                    marker_color=["#ef4444" if r else "#f59e0b" for r in wi_df["at_risk"]]))
            for _, r in wi_df.iterrows():
                fig_wi.add_shape(type="line",
                                 x0=r["account_name"], x1=r["account_name"],
                                 y0=0, y1=r["min_balance"],
                                 line=dict(color="#ef4444", width=1, dash="dot"))
            fig_wi.update_layout(barmode="group", height=350, hovermode="x unified",
                                 xaxis=dict(tickangle=-20, gridcolor="rgba(255,255,255,0.05)"),
                                 yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                                 **{k: v for k, v in PLOT_LAYOUT.items() if k not in ("xaxis","yaxis")})
            st.plotly_chart(fig_wi, use_container_width=True)

            tbl = wi_df[["account_name","currency","base_end","stressed_end","delta","shortfall","at_risk"]].copy()
            tbl.columns = ["Account","Currency","Base","Stress","Change","Shortfall","At risk"]
            for c in ["Base","Stress","Change","Shortfall"]:
                tbl[c] = tbl[c].apply(lambda x: f"{x:,.0f}")
            tbl["At risk"] = tbl["At risk"].apply(lambda x: "Yes" if x else "No")
            st.dataframe(tbl, use_container_width=True, hide_index=True)


# ── TAB 6: Analytics ───────────────────────────────────────────────────────────
with tab6:
    st.subheader("KPI & Analytics")

    kl, kr = st.columns(2)
    with kl:
        st.markdown("#### Liquidity efficiency — 90 days")
        h90 = df[df["date"] >= df["date"].max() - pd.Timedelta(days=90)]
        h90 = h90[h90["currency"].isin(currencies)]
        eff_series = h90.groupby("date").apply(lambda g: (
            (g["balance"] * g["currency"].map(_fx)).sum() /
            max((g["target_balance"] * g["currency"].map(_fx)).sum(), 1) * 100
        )).reset_index(name="eff")
        fig_eff = go.Figure()
        fig_eff.add_trace(go.Scatter(x=eff_series["date"], y=eff_series["eff"],
                                      fill="tozeroy", line=dict(color="#3b82f6", width=2),
                                      fillcolor="rgba(59,130,246,0.07)"))
        fig_eff.add_hline(y=100, line_dash="dot", line_color="#475569", annotation_text="100%")
        fig_eff.update_layout(height=220, yaxis_title="% of target", margin=dict(t=10), **PLOT_LAYOUT)
        st.plotly_chart(fig_eff, use_container_width=True)

        st.markdown("#### Idle capital by account")
        idle_df = pd.DataFrame(idle["details"])
        if not idle_df.empty:
            fig_idle = px.bar(idle_df.sort_values("idle_usd", ascending=False),
                              x="account", y=["balance_usd","idle_usd"],
                              barmode="overlay",
                              color_discrete_map={"balance_usd":"#3b82f6","idle_usd":"#ef4444"})
            fig_idle.update_layout(height=220, yaxis_title="USD", margin=dict(t=10),
                                   xaxis=dict(tickangle=-20, gridcolor="rgba(255,255,255,0.05)"),
                                   yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                                   **{k: v for k, v in PLOT_LAYOUT.items() if k not in ("xaxis","yaxis")})
            st.plotly_chart(fig_idle, use_container_width=True)

    with kr:
        st.markdown("#### Inflow / Outflow heatmap — 14 days")
        h14 = df[(df["date"] >= df["date"].max() - pd.Timedelta(days=14)) &
                  (df["currency"].isin(currencies))]
        pivot = h14.pivot_table(index="account_name", columns="date",
                                 values="net_flow", aggfunc="sum").fillna(0)
        if not pivot.empty:
            fig_hm = go.Figure(go.Heatmap(
                z=pivot.values,
                x=[str(c.date()) for c in pivot.columns],
                y=pivot.index.tolist(),
                colorscale=[[0,"#ef4444"],[0.5,"#1e293b"],[1,"#22c55e"]],
                zmid=0,
            ))
            fig_hm.update_layout(height=240, paper_bgcolor="rgba(0,0,0,0)",
                                  font_color="#cbd5e1", margin=dict(t=10))
            st.plotly_chart(fig_hm, use_container_width=True)

        st.markdown("#### System KPIs")
        avg_pending = state_f["pending_inflow"].mean()
        avg_balance = state_f["balance"].mean()
        settlement_eff = max(0, 1 - avg_pending / max(avg_balance, 1)) * 100

        kpi_data = {
            "Liquidity efficiency":    f"{efficiency:.1f}%",
            "Settlement efficiency":   f"{settlement_eff:.1f}%",
            "Lost revenue / yr":       f"${idle['annual_opp_cost_usd']/1e3:.0f}K",
            "Accounts in deficit":     str(int((state_f["deficit"] > 0).sum())),
            "Avg. clearing delay":     f"{np.mean([CLEARING_DAYS[ps] for ps in state_f['payment_system']]):.1f} days",
            "Alerts generated":        str(len(alerts)),
        }
        for label, val in kpi_data.items():
            st.markdown(f'<div class="kpi-row"><span style="color:#64748b;font-size:12px;font-weight:500">{label}</span><strong style="color:#f1f5f9">{val}</strong></div>', unsafe_allow_html=True)


# ── TAB 7: Data ────────────────────────────────────────────────────────────────
with tab7:
    st.subheader("Data Management")

    mode_idx = 1 if st.session_state.use_custom_data else 0
    data_mode = st.radio("Data source", ["Demo Data", "Custom Data"],
                          horizontal=True, index=mode_idx)
    new_mode = data_mode.startswith("Custom")
    if new_mode != st.session_state.use_custom_data:
        st.session_state.use_custom_data = new_mode
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    st.divider()

    if not st.session_state.use_custom_data:
        st.markdown("#### Demo accounts")
        demo_tbl = pd.DataFrame([{
            "Account": a["name"], "Currency": a["currency"], "System": a["payment_system"],
            "Min balance": f"{a['min_balance']:,.0f}", "Target balance": f"{a['target_balance']:,.0f}",
            "Daily volume": f"{a['daily_volume']:,.0f}",
        } for a in ACCOUNTS])
        st.dataframe(demo_tbl, use_container_width=True, hide_index=True)
        st.info("Switch to **Custom Data** to add your own accounts.")
    else:
        l_col, r_col = st.columns([1, 1], gap="large")
        with l_col:
            st.markdown("#### Add account")
            with st.form("add_acc_form", clear_on_submit=True):
                name = st.text_input("Account name *", placeholder="My Bank USD")
                a1, a2 = st.columns(2)
                ccy = a1.selectbox("Currency", ["USD","EUR","GBP"])
                ps  = a2.selectbox("Payment system", ["SWIFT","SEPA","CARD","LOCAL"])
                b1, b2 = st.columns(2)
                min_b = b1.number_input("Min balance",    0, value=100_000, step=10_000)
                tgt_b = b2.number_input("Target balance", 0, value=500_000, step=10_000)
                c1_, c2_ = st.columns(2)
                cur_b = c1_.number_input("Current balance", 0, value=350_000, step=10_000)
                dv    = c2_.number_input("Daily volume",    1, value=200_000, step=10_000)
                if st.form_submit_button("Add Account", type="primary", use_container_width=True):
                    if not name.strip():
                        st.error("Enter an account name")
                    elif min_b >= tgt_b:
                        st.error("Target balance must be greater than minimum")
                    else:
                        acc_id = f"user_{len(st.session_state.user_accounts)+1}_{name.strip()[:15].replace(' ','_').upper()}"
                        st.session_state.user_accounts.append({
                            "id": acc_id, "name": name.strip(), "currency": ccy,
                            "payment_system": ps, "min_balance": float(min_b),
                            "target_balance": float(tgt_b), "current_balance": float(cur_b),
                            "daily_volume": float(dv),
                        })
                        st.cache_data.clear(); st.cache_resource.clear()
                        st.success(f"Account **{name.strip()}** added.")
                        st.rerun()

        with r_col:
            st.markdown("#### Exchange rates (to USD)")
            with st.form("fx_form"):
                eur_r = st.number_input("EUR/USD", 0.5, 2.0,
                                        value=st.session_state.custom_fx.get("EUR", 1.08),
                                        step=0.01, format="%.3f")
                gbp_r = st.number_input("GBP/USD", 0.5, 2.0,
                                        value=st.session_state.custom_fx.get("GBP", 1.27),
                                        step=0.01, format="%.3f")
                if st.form_submit_button("Apply", use_container_width=True):
                    st.session_state.custom_fx = {"USD":1.0,"EUR":eur_r,"GBP":gbp_r}
                    st.rerun()

        st.divider()
        if not st.session_state.user_accounts:
            st.info("Add an account on the left.")
        else:
            st.markdown(f"#### My accounts ({len(st.session_state.user_accounts)})")
            for i, acc in enumerate(st.session_state.user_accounts):
                bal = acc["current_balance"]
                ok = "OK" if bal >= acc["target_balance"]*0.8 else "WARN" if bal >= acc["min_balance"] else "ERR"
                with st.expander(f"{ok}  {acc['name']} — {bal:,.0f} {acc['currency']}"):
                    ea, eb = st.columns([2, 2])
                    new_b = ea.number_input("Current balance", 0.0, value=float(bal),
                                            step=float(max(1000, acc["daily_volume"]*0.01)),
                                            key=f"nb_{i}", format="%.0f")
                    eb.metric("Min", f"{acc['min_balance']:,.0f}")
                    st.caption(f"Target: {acc['target_balance']:,.0f} | System: {acc['payment_system']} | Daily: {acc['daily_volume']:,.0f}")
                    uc, dc = st.columns([3, 1])
                    if uc.button("Save", key=f"save_{i}", use_container_width=True):
                        st.session_state.user_accounts[i]["current_balance"] = new_b
                        st.rerun()
                    if dc.button("Remove", key=f"del_{i}", use_container_width=True, type="secondary"):
                        st.session_state.user_accounts.pop(i)
                        st.cache_data.clear(); st.cache_resource.clear()
                        st.rerun()
            if st.button("Clear all accounts", type="secondary"):
                st.session_state.user_accounts = []
                st.cache_data.clear(); st.cache_resource.clear()
                st.rerun()

        st.divider()
        st.markdown("#### Upload CSV")
        st.caption("Format: `date, account_name, currency, payment_system, inflow, outflow, balance`")
        uploaded = st.file_uploader("CSV file", type=["csv"])
        if uploaded:
            try:
                csv_df = pd.read_csv(uploaded)
                missing = {"date","account_name","currency","payment_system","inflow","outflow","balance"} - set(csv_df.columns)
                if missing:
                    st.error(f"Missing columns: {', '.join(missing)}")
                else:
                    st.success(f"Loaded {len(csv_df)} rows, {csv_df['account_name'].nunique()} accounts")
                    st.dataframe(csv_df.head(5), use_container_width=True, hide_index=True)
                    if st.button("Import from CSV", type="primary"):
                        new_accs = []
                        for aname, grp in csv_df.groupby("account_name"):
                            grp = grp.sort_values("date")
                            new_accs.append({
                                "id": f"csv_{str(aname)[:15].replace(' ','_').upper()}",
                                "name": str(aname),
                                "currency": grp["currency"].iloc[0] if grp["currency"].iloc[0] in ("USD","EUR","GBP") else "USD",
                                "payment_system": grp["payment_system"].iloc[0] if grp["payment_system"].iloc[0] in ("SWIFT","SEPA","CARD","LOCAL") else "SWIFT",
                                "min_balance":    float(grp["balance"].quantile(0.05)),
                                "target_balance": float(grp["balance"].median()),
                                "current_balance":float(grp["balance"].iloc[-1]),
                                "daily_volume":   float((grp["inflow"]+grp["outflow"]).mean()),
                            })
                        st.session_state.user_accounts = new_accs
                        st.session_state.use_custom_data = True
                        st.cache_data.clear(); st.cache_resource.clear()
                        st.success(f"Created {len(new_accs)} accounts")
                        st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

# ── TAB 8: Business Model ──────────────────────────────────────────────────────
with tab8:
    st.subheader("Business Model — LiquidityAI")

    # ── Monetization ────────────────────────────────────────────────────────────
    st.markdown("#### Монетизация")
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.markdown("""
<div class="card">
  <div style="font-size:13px;color:#94a3b8;font-weight:500;margin-bottom:8px">SaaS Подписка</div>
  <div style="font-size:22px;font-weight:700;color:#3b82f6;margin-bottom:8px">$2K–$15K / мес</div>
  <div style="font-size:13px;color:#cbd5e1;line-height:1.6">
    <b>Starter</b> — до 10 счетов, базовый прогноз: $2K<br>
    <b>Pro</b> — до 50 счетов, API, алерты: $7K<br>
    <b>Enterprise</b> — безлимит, SLA, on-premise: $15K+
  </div>
</div>""", unsafe_allow_html=True)
    with mc2:
        st.markdown("""
<div class="card">
  <div style="font-size:13px;color:#94a3b8;font-weight:500;margin-bottom:8px">Revenue Share</div>
  <div style="font-size:22px;font-weight:700;color:#22c55e;margin-bottom:8px">5–15 bps</div>
  <div style="font-size:13px;color:#cbd5e1;line-height:1.6">
    С каждого подтверждённого перевода по рекомендации системы.<br>
    При $50M/мес оборота → <b>$25K–75K</b> дополнительно.<br>
    Клиент платит только за результат.
  </div>
</div>""", unsafe_allow_html=True)
    with mc3:
        st.markdown("""
<div class="card">
  <div style="font-size:13px;color:#94a3b8;font-weight:500;margin-bottom:8px">Professional Services</div>
  <div style="font-size:22px;font-weight:700;color:#f59e0b;margin-bottom:8px">$500–$1500 / час</div>
  <div style="font-size:13px;color:#cbd5e1;line-height:1.6">
    Интеграция с ERP/CBS (SAP, Oracle).<br>
    Кастомные ML-модели под клиента.<br>
    Обучение treasury-команды.
  </div>
</div>""", unsafe_allow_html=True)

    st.divider()

    # ── TAM / SAM / SOM ─────────────────────────────────────────────────────────
    st.markdown("#### Объём рынка (TAM / SAM / SOM)")

    tm1, tm2, tm3 = st.columns(3)
    with tm1:
        st.markdown("""
<div style="background:#1e293b;border:1px solid #334155;border-top:3px solid #3b82f6;border-radius:12px;padding:20px;text-align:center">
  <div style="font-size:12px;color:#94a3b8;font-weight:600;letter-spacing:0.08em;margin-bottom:8px">TAM — Total Addressable Market</div>
  <div style="font-size:36px;font-weight:800;color:#3b82f6;margin:8px 0">$42B</div>
  <div style="font-size:12px;color:#64748b">Глобальный рынок Treasury Management Software.<br>CAGR 12.4% до 2030 (Grand View Research 2024)</div>
</div>""", unsafe_allow_html=True)
    with tm2:
        st.markdown("""
<div style="background:#1e293b;border:1px solid #334155;border-top:3px solid #22c55e;border-radius:12px;padding:20px;text-align:center">
  <div style="font-size:12px;color:#94a3b8;font-weight:600;letter-spacing:0.08em;margin-bottom:8px">SAM — Serviceable Addressable Market</div>
  <div style="font-size:36px;font-weight:800;color:#22c55e;margin:8px 0">$8.4B</div>
  <div style="font-size:12px;color:#64748b">Mid-size fintech и корпоративные казначейства (50–5000 сотрудников) в Европе, СНГ, MENA.<br>~20% от TAM</div>
</div>""", unsafe_allow_html=True)
    with tm3:
        st.markdown("""
<div style="background:#1e293b;border:1px solid #334155;border-top:3px solid #f59e0b;border-radius:12px;padding:20px;text-align:center">
  <div style="font-size:12px;color:#94a3b8;font-weight:600;letter-spacing:0.08em;margin-bottom:8px">SOM — Serviceable Obtainable Market</div>
  <div style="font-size:36px;font-weight:800;color:#f59e0b;margin:8px 0">$84M</div>
  <div style="font-size:12px;color:#64748b">Целевой захват 1% SAM за 3 года.<br>~350 клиентов × $20K ARR avg.<br>Реалистичный план выхода</div>
</div>""", unsafe_allow_html=True)

    # Воронка
    st.markdown("<br>", unsafe_allow_html=True)
    fig_tam = go.Figure(go.Funnel(
        y=["TAM — $42B", "SAM — $8.4B", "SOM — $84M"],
        x=[42000, 8400, 84],
        textinfo="label+percent initial",
        marker=dict(color=["#3b82f6", "#22c55e", "#f59e0b"]),
        connector=dict(line=dict(color="#334155", width=1)),
    ))
    fig_tam.update_layout(
        height=280, margin=dict(t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1",
    )
    st.plotly_chart(fig_tam, use_container_width=True)

    st.divider()

    # ── Competitor analysis ──────────────────────────────────────────────────────
    st.markdown("#### Анализ конкурентов")

    competitors = [
        {"name": "Kyriba",        "price": "$$$$$", "ai": "Базовый",  "deploy": "Cloud",      "api": "Да",  "realtime": "Нет", "score": 65},
        {"name": "SAP TRM",       "price": "$$$$$", "ai": "Нет",      "deploy": "On-premise", "api": "Да",  "realtime": "Нет", "score": 55},
        {"name": "FIS Integrity", "price": "$$$$",  "ai": "Нет",      "deploy": "On-premise", "api": "Огр","realtime": "Нет", "score": 50},
        {"name": "Coupa Treasury","price": "$$$$",  "ai": "Базовый",  "deploy": "Cloud",      "api": "Да",  "realtime": "Нет", "score": 60},
        {"name": "LiquidityAI",   "price": "$$",    "ai": "ML-native","deploy": "Cloud/API",  "api": "Да",  "realtime": "Да",  "score": 92},
    ]
    comp_df = pd.DataFrame(competitors)

    def style_row(row):
        if row["name"] == "LiquidityAI":
            return ["background-color:#0f2d1f;color:#22c55e;font-weight:700"] * len(row)
        return [""] * len(row)

    st.dataframe(
        comp_df.rename(columns={
            "name": "Продукт", "price": "Цена", "ai": "AI/ML",
            "deploy": "Деплой", "api": "REST API", "realtime": "Real-time", "score": "Оценка /100"
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    fig_comp = go.Figure()
    colors = ["#64748b","#64748b","#64748b","#64748b","#22c55e"]
    fig_comp.add_trace(go.Bar(
        x=[c["name"] for c in competitors],
        y=[c["score"] for c in competitors],
        marker_color=colors,
        text=[f"{c['score']}/100" for c in competitors],
        textposition="outside",
    ))
    fig_comp.add_hline(y=80, line_dash="dot", line_color="#3b82f6", annotation_text="Целевой порог")
    fig_comp.update_layout(
        height=280, yaxis_title="Интегральная оценка", margin=dict(t=20, b=10),
        yaxis=dict(range=[0,105], gridcolor="rgba(255,255,255,0.05)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        **{k: v for k, v in PLOT_LAYOUT.items() if k not in ("xaxis","yaxis")},
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    st.divider()

    # ── Unit economics ───────────────────────────────────────────────────────────
    st.markdown("#### Unit Economics")
    ue1, ue2, ue3, ue4 = st.columns(4)
    ue1.metric("CAC (привлечение клиента)",  "$8,500")
    ue2.metric("LTV (3 года)",               "$72,000",  delta="LTV/CAC = 8.5x")
    ue3.metric("Payback period",             "6 месяцев")
    ue4.metric("Gross Margin",               "78%",       delta="SaaS benchmark 70–80%")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown('<div style="text-align:center;color:#334155;font-size:12px">LiquidityAI v2.0 &nbsp;&middot;&nbsp; FinTech Hackathon 2026 &nbsp;&middot;&nbsp; Treasury Management System</div>', unsafe_allow_html=True)
