import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
import base64
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FixedLocator

plt.style.use("dark_background")

# -----------------------------
# Assets (put these in the same folder as the .py in GitHub)
# -----------------------------
LOGO_PATH = "MIA logo 5.jpg"
LOBSTER_TTF = "Lobster-Regular.ttf"   # <-- add this file to your repo

# -----------------------------
# Config
# -----------------------------
DEFAULT_DISCOUNT_RATE = 0.11
DEFAULT_MARGIN_OF_SAFETY = 0.25
HEADERS = {"User-Agent": "Mozilla/5.0"}
DIVS_PH_BASE = "https://dividends.ph/company/"

SEED_CMPY_ID_MAP = {
    "AREIT": 679,
    "ICT": 83,
    "JGS": 210,
    "URC": 124,
    "MREIT": 685,
    "BPI": 234,
    "FPH": 197,
    "PSE": 478,
    "SMPH": 112,
    "SMC": 154,
    "TEL": 6,
    "SCC": 157,
    "ACEN": 233,
    "MEG": 127,
    "AP": 609,
}
DEFAULT_TICKERS = sorted(SEED_CMPY_ID_MAP.keys())

# -----------------------------
# Load local font (no internet)
# -----------------------------
def load_local_font_base64(ttf_path: str) -> str | None:
    try:
        with open(ttf_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None

lobster_b64 = load_local_font_base64(LOBSTER_TTF)

# -----------------------------
# Helpers: sync watchlist with cmpy_id_map
# -----------------------------
def normalize_ticker(x: str) -> str:
    return str(x or "").upper().strip()

def sync_watchlist_with_mapping(watchlist: pd.DataFrame, id_map: dict) -> pd.DataFrame:
    wl = watchlist.copy()
    if wl.empty:
        wl = pd.DataFrame(columns=["Ticker", "Dividends (annual totals, oldest→newest)", "Notes"])

    wl["Ticker"] = wl["Ticker"].astype(str).map(normalize_ticker)
    existing = set(wl["Ticker"].tolist())
    to_add = sorted(set(map(normalize_ticker, id_map.keys())) - existing)

    if to_add:
        add_df = pd.DataFrame([{
            "Ticker": t,
            "Dividends (annual totals, oldest→newest)": "",
            "Notes": ""
        } for t in to_add])
        wl = pd.concat([wl, add_df], ignore_index=True)

    wl = wl[wl["Ticker"].astype(str).str.strip() != ""]
    wl = wl.drop_duplicates(subset=["Ticker"], keep="last")
    wl = wl.sort_values("Ticker").reset_index(drop=True)
    return wl

# -----------------------------
# PSE Edge fetch + parsing
# -----------------------------
@st.cache_data(ttl=60 * 30)
def fetch_pse_edge_stock_text(cmpy_id: int) -> str:
    url = f"https://edge.pse.com.ph/companyPage/stockData.do?cmpy_id={cmpy_id}"
    html = requests.get(url, headers=HEADERS, timeout=30).text
    soup = BeautifulSoup(html, "lxml")
    return soup.get_text(" ", strip=True)

def _pick_number(text: str, label: str) -> float | None:
    m = re.search(re.escape(label) + r"\s*([0-9,]+(?:\.[0-9]+)?)", text)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except Exception:
        return None

@st.cache_data(ttl=60 * 30)
def fetch_stock_snapshot(cmpy_id: int) -> dict:
    try:
        text = fetch_pse_edge_stock_text(cmpy_id)
        snap = {
            "Last Traded Price": _pick_number(text, "Last Traded Price"),
            "Previous Close": _pick_number(text, "Previous Close"),
            "Open": _pick_number(text, "Open"),
            "Day High": _pick_number(text, "Day High"),
            "Day Low": _pick_number(text, "Day Low"),
            "52-Week High": _pick_number(text, "52-Week High"),
            "52-Week Low": _pick_number(text, "52-Week Low"),
            "Volume": _pick_number(text, "Volume"),
            "Value": _pick_number(text, "Value"),
        }
        if snap["Last Traded Price"] is None:
            snap["Last Traded Price"] = snap["Previous Close"]
        return snap
    except Exception:
        return {}

# -----------------------------
# dividends.ph raw dividends (always fetch)
# -----------------------------
def _to_float_rate(x: str) -> float | None:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", s.replace(",", ""))
    return float(m.group(1)) if m else None

@st.cache_data(ttl=60 * 60 * 24)
def fetch_dividendsph_raw(ticker: str) -> pd.DataFrame:
    ticker = normalize_ticker(ticker)
    if not ticker:
        return pd.DataFrame()

    url = f"{DIVS_PH_BASE}{ticker}"
    try:
        html = requests.get(url, headers=HEADERS, timeout=30).text
        tables = pd.read_html(html)
    except Exception:
        return pd.DataFrame()

    if not tables:
        return pd.DataFrame()

    target = None
    for t in tables:
        cols = " ".join([str(c).lower() for c in t.columns])
        if "payment" in cols and "dividend" in cols and "rate" in cols:
            target = t.copy()
            break
    if target is None:
        return pd.DataFrame()

    target.columns = [str(c).strip() for c in target.columns]
    pay_col = next((c for c in target.columns if "Payment" in c), None)
    ex_col = next((c for c in target.columns if "Ex" in c), None)
    rate_col = next((c for c in target.columns if "Dividend Rate" in c), None)
    if not pay_col or not rate_col:
        return pd.DataFrame()

    df = pd.DataFrame({
        "payment_date": target[pay_col],
        "ex_date": target[ex_col] if ex_col else None,
        "dividend_rate": target[rate_col],
    })
    df["payment_date"] = pd.to_datetime(df["payment_date"], errors="coerce")
    df["ex_date"] = pd.to_datetime(df["ex_date"], errors="coerce") if ex_col else pd.NaT
    df["rate"] = df["dividend_rate"].apply(_to_float_rate)
    df = df.dropna(subset=["payment_date", "rate"]).copy()
    df["year"] = df["payment_date"].dt.year
    return df[["payment_date", "ex_date", "rate", "year"]].sort_values("payment_date", ascending=False).reset_index(drop=True)

def annual_totals_from_raw(raw_df: pd.DataFrame, years: int = 4) -> list[float]:
    if raw_df is None or raw_df.empty:
        return []
    g = raw_df.groupby("year")["rate"].sum().sort_index()
    g = g.tail(years)
    return g.tolist()

# -----------------------------
# DDM valuation helpers
# -----------------------------
def ddm(dividend_next_year: float, discount_rate: float, growth_rate: float) -> float | None:
    if discount_rate <= growth_rate:
        return None
    return dividend_next_year / (discount_rate - growth_rate)

def parse_dividends(div_str: str) -> list[float]:
    if not div_str or not str(div_str).strip():
        return []
    parts = [p.strip() for p in str(div_str).split(",")]
    out = []
    for p in parts:
        if not p:
            continue
        try:
            out.append(float(p))
        except Exception:
            pass
    return out

def estimate_growth_from_divs(divs: list[float]) -> float:
    if len(divs) < 2:
        return 0.0
    arr = np.array(divs, dtype=float)
    yoy = np.diff(arr) / arr[:-1]
    g = float(np.mean(yoy)) if len(yoy) else 0.0
    return float(min(max(g, -0.05), 0.06))

def dividend_trend(divs: list[float]) -> str:
    if len(divs) < 2:
        return "No/Insufficient data"
    first, last = divs[0], divs[-1]
    n = len(divs) - 1
    if first <= 0 or last <= 0:
        return "No/Insufficient data"
    cagr = (last / first) ** (1 / n) - 1
    if cagr > 0.03:
        return f"↗ Increasing (~{cagr*100:.1f}%/yr)"
    if cagr < -0.03:
        return f"↘ Declining (~{cagr*100:.1f}%/yr)"
    return f"→ Flat (~{cagr*100:.1f}%/yr)"

# -----------------------------
# UI styling
# -----------------------------
st.set_page_config(page_title="Merlyn's Stock Valuation Dashboard", page_icon="📈", layout="wide")

font_face = ""
if lobster_b64:
    font_face = f"""
    @font-face {{
      font-family: 'LobsterLocal';
      src: url(data:font/ttf;base64,{lobster_b64}) format('truetype');
      font-weight: normal;
      font-style: normal;
    }}
    """

st.markdown(
    f"""
<style>
{font_face}

.block-container {{ padding-top: 1.0rem; padding-bottom: 2.2rem; max-width: 1200px; }}
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
.muted {{ opacity: 0.76; font-size: 0.93rem; line-height: 1.55; }}

.stButton>button, .stDownloadButton>button, .stLinkButton>button {{
  border-radius: 14px !important;
  padding: 10px 14px !important;
  border: 1px solid rgba(120,120,120,0.25) !important;
}}
.stButton>button:hover, .stDownloadButton>button:hover, .stLinkButton>button:hover {{
  transform: translateY(-1px);
  transition: 120ms ease;
}}
button[kind="primary"] {{
  background: #FFD54A !important;
  color: #1b1b1b !important;
  border: 0 !important;
  box-shadow: 0 10px 24px rgba(255, 213, 74, 0.25) !important;
}}

[data-testid="stDataFrame"] {{
  border-radius: 16px;
  overflow: hidden;
  border: 1px solid rgba(120,120,120,0.18);
}}

.title-wrap {{ padding-top: 15px; }}
.title-green {{
  color: #00C853;
  font-family: {'LobsterLocal' if lobster_b64 else 'system-ui, -apple-system, Segoe UI, Roboto, Arial'};
  margin: 0;
  padding: 0;
  line-height: 1.05;
}}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Session init
# -----------------------------
if "cmpy_id_map" not in st.session_state:
    st.session_state.cmpy_id_map = dict(SEED_CMPY_ID_MAP)

if "watchlist" not in st.session_state:
    st.session_state.watchlist = pd.DataFrame(
        [{"Ticker": t, "Dividends (annual totals, oldest→newest)": "", "Notes": ""} for t in DEFAULT_TICKERS]
    )

st.session_state.watchlist = sync_watchlist_with_mapping(st.session_state.watchlist, st.session_state.cmpy_id_map)

# -----------------------------
# Main Title row: logo + Lobster green title + top padding
# -----------------------------
st.markdown('<div class="title-wrap">', unsafe_allow_html=True)
tcol1, tcol2 = st.columns([0.16, 0.84], vertical_alignment="center")
with tcol1:
    try:
        st.image(LOGO_PATH, width=95)
    except Exception:
        st.write("")
with tcol2:
    st.markdown('<h2 class="title-green">Merlyn’s Stock Valuation Dashboard</h2>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Quick check if Lobster loaded
# -----------------------------
if not lobster_b64:
    st.warning("Lobster font file not found. Upload Lobster-Regular.ttf to your GitHub repo (same folder as the app).")

# (rest of your app remains unchanged…)
st.info("✅ Font fix applied. Now upload Lobster-Regular.ttf to GitHub and redeploy.")
