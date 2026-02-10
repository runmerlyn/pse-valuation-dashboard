import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# -----------------------------
# Chart style: darker, colorful lines
# -----------------------------
plt.style.use("dark_background")

# -----------------------------
# Configuration
# -----------------------------
DEFAULT_DISCOUNT_RATE = 0.11
DEFAULT_MARGIN_OF_SAFETY = 0.25
HEADERS = {"User-Agent": "Mozilla/5.0"}

# Stable cmpy_id mapping for your 15 tickers (used for PSE Edge price)
CMPY_ID_MAP = {
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
    "TEL": 6,     # cmpy_id (not price)
    "SCC": 157,
    "ACEN": 233,
    "MEG": 127,
    "AP": 609,
}
DEFAULT_TICKERS = sorted(CMPY_ID_MAP.keys())

DIVS_PH_BASE = "https://dividends.ph/company/"

# -----------------------------
# PSE Edge price fetching
# -----------------------------
@st.cache_data(ttl=60 * 30)  # cache prices for 30 minutes
def fetch_price_by_cmpy_id(cmpy_id: int) -> float | None:
    try:
        url = f"https://edge.pse.com.ph/companyPage/stockData.do?cmpy_id={cmpy_id}"
        html = requests.get(url, headers=HEADERS, timeout=30).text
        soup = BeautifulSoup(html, "lxml")
        text = soup.get_text(" ", strip=True)

        def pick(label):
            m = re.search(label + r"\s*([0-9,]+(?:\.[0-9]+)?)", text)
            return float(m.group(1).replace(",", "")) if m else None

        return pick("Last Traded Price") or pick("Previous Close")
    except Exception:
        return None

# -----------------------------
# dividends.ph fetching (raw payments + annual totals)
# -----------------------------
def _to_float_rate(x: str) -> float | None:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", s.replace(",", ""))
    return float(m.group(1)) if m else None

@st.cache_data(ttl=60 * 60 * 24)  # cache dividends.ph results for 24 hours
def fetch_dividendsph_raw(ticker: str) -> pd.DataFrame:
    """
    Returns raw dividend payments from dividends.ph for a ticker.
    Best-effort: returns empty dataframe if missing/unparseable.
    """
    ticker = ticker.upper().strip()
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
    ex_col = next((c for c in target.columns if "Ex" in c), None)  # optional
    rate_col = next((c for c in target.columns if "Dividend Rate" in c), None)

    if not pay_col or not rate_col:
        return pd.DataFrame()

    df = pd.DataFrame({
        "payment_date": target[pay_col],
        "ex_date": target[ex_col] if ex_col else None,
        "dividend_rate": target[rate_col],
    })

    df["payment_date"] = pd.to_datetime(df["payment_date"], errors="coerce")
    if ex_col:
        df["ex_date"] = pd.to_datetime(df["ex_date"], errors="coerce")
    else:
        df["ex_date"] = pd.NaT

    df["rate"] = df["dividend_rate"].apply(_to_float_rate)
    df = df.dropna(subset=["payment_date", "rate"]).copy()
    df["year"] = df["payment_date"].dt.year

    # newest first for table
    df = df.sort_values("payment_date", ascending=False).reset_index(drop=True)
    return df[["payment_date", "ex_date", "rate", "year"]]

def annual_totals_from_raw(raw_df: pd.DataFrame, years: int = 4) -> list[float]:
    """
    Returns annual totals oldest->newest for last N years, from raw payment rows.
    """
    if raw_df is None or raw_df.empty:
        return []
    g = raw_df.groupby("year")["rate"].sum().sort_index()
    g = g.tail(years)
    return g.tolist()

# -----------------------------
# Valuation helpers (DDM)
# -----------------------------
def ddm(dividend_next_year: float, discount_rate: float, growth_rate: float) -> float | None:
    """
    Gordon Growth Dividend Discount Model:
      Intrinsic = D1 / (r - g)
    """
    if discount_rate <= growth_rate:
        return None
    return dividend_next_year / (discount_rate - growth_rate)

def parse_dividends(div_str: str) -> list[float]:
    """
    Manual input: annual totals (oldest -> newest) like "20, 22, 24, 25"
    """
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
    """
    Growth estimate from average year-over-year change of annual dividend totals.
    Capped to keep results stable.
    """
    if len(divs) < 2:
        return 0.0
    arr = np.array(divs, dtype=float)
    yoy = np.diff(arr) / arr[:-1]
    g = float(np.mean(yoy)) if len(yoy) else 0.0
    return float(min(max(g, -0.05), 0.06))

def dividend_trend(divs: list[float]) -> str:
    """
    Simple trend based on CAGR across annual dividend totals.
    """
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
st.set_page_config(page_title="PSE Valuation Dashboard", page_icon="💸", layout="wide")

st.markdown(
    """
<style>
.block-container { padding-top: 1.1rem; padding-bottom: 2.2rem; max-width: 1200px; }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.muted { opacity: 0.72; font-size: 0.92rem; line-height: 1.45; }

/* Rounded buttons */
.stButton>button, .stDownloadButton>button, .stLinkButton>button {
  border-radius: 14px !important;
  padding: 10px 14px !important;
  border: 1px solid rgba(120,120,120,0.25) !important;
}
.stButton>button:hover, .stDownloadButton>button:hover, .stLinkButton>button:hover {
  transform: translateY(-1px);
  transition: 120ms ease;
}

/* Primary button = yellow */
button[kind="primary"] {
  background: #FFD54A !important;
  color: #1b1b1b !important;
  border: 0 !important;
  box-shadow: 0 10px 24px rgba(255, 213, 74, 0.25) !important;
}

/* Secondary buttons a bit brighter */
.stButton>button[kind="secondary"] {
  background: rgba(255,255,255,0.06) !important;
}

/* Dataframe styling */
[data-testid="stDataFrame"] {
  border-radius: 16px;
  overflow: hidden;
  border: 1px solid rgba(120,120,120,0.18);
}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Header + links
# -----------------------------
st.markdown("## 💸 Valuation Dashboard")
st.markdown(
    '<div class="muted">'
    '<b>Idea:</b> Compare the market price vs an intrinsic value estimate based on dividends. '
    'Prices come from <b>PSE Edge</b> using <code>cmpy_id</code>. '
    'Dividends are pulled from <b>dividends.ph</b> as raw payments and converted to annual totals (last 4 years) for DDM.'
    '</div>',
    unsafe_allow_html=True,
)

link_c1, link_c2, link_c3 = st.columns([1.2, 1.2, 1.2])
with link_c1:
    st.link_button(
        "📅 PSE Edge Dividend List",
        "https://edge.pse.com.ph/disclosureData/dividends_and_rights_info_list.ax",
        use_container_width=True,
    )
with link_c2:
    st.link_button(
        "🌐 dividends.ph (lookup)",
        "https://dividends.ph/",
        use_container_width=True,
    )
with link_c3:
    st.link_button(
        "🏦 PSE ETF Page",
        "https://www.pse.com.ph/exchange-traded-fund/",
        use_container_width=True,
    )

with st.expander("📘 What this app is doing (short explanation)"):
    st.markdown(
        """
**What the app is for**  
This app helps you quickly compare **market price** vs an **intrinsic value** estimate for dividend-paying PH stocks/REITs using the **Dividend Discount Model (DDM)**. It also shows dividend trend and raw dividend payment history.

**Core model (Gordon Growth DDM)**  
- **Intrinsic Value** = **D1 / (r − g)**  
  - **D1** = next year's dividend per share (estimated)  
  - **r** = discount rate (your required return)  
  - **g** = dividend growth rate (estimated from the annual dividend totals)

**How D1 and g are estimated**  
- We build **annual dividend totals** from raw dividend payments and take the last 4 years (if available).  
- **g** is estimated from the average year-over-year change in annual totals and capped between **-5% and +6%** to keep results stable.  
- **D1 = last_annual_dividend × (1 + g)**

**Buy Below (Margin of Safety)**  
- **Buy Below** = Intrinsic Value × (1 − margin_of_safety)

**Valuation labels**  
- **Undervalued**: Price < Buy Below  
- **Fairly Valued**: Buy Below ≤ Price ≤ Intrinsic Value  
- **Overvalued**: Price > Intrinsic Value  

**What each result column means**  
- **Price**: fetched from PSE Edge (Last Traded Price, fallback to Previous Close)  
- **Dividend points (annual)**: number of annual totals used (manual or auto)  
- **Dividend Trend (annual)**: trend based on dividend CAGR  
- **Intrinsic Value**: DDM intrinsic value (if enough dividend points exist)  
- **Buy Below**: intrinsic value with margin of safety applied  
- **Dividend source**: `manual` if you typed annual totals, otherwise `dividends.ph` if auto-fetched  
- **Notes**: your own notes per ticker
        """.strip()
    )

with st.expander("✨ How to add a new ticker (share-friendly)"):
    st.write(
        "**Step 1 — Find `cmpy_id`:** open PSE Edge stock data in a browser.\n\n"
        "Example:\n"
        "`https://edge.pse.com.ph/companyPage/stockData.do?cmpy_id=118`\n\n"
        "**Step 2 — Add a row** in the watchlist and fill:\n"
        "- Ticker\n- cmpy_id\n- (Optional) Annual dividends like `20, 22, 24, 25`\n- Notes\n\n"
        "If dividends are blank, the app tries dividends.ph automatically."
    )

st.markdown("---")

# -----------------------------
# Controls
# -----------------------------
c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    discount_rate = st.slider("Discount rate (r)", 0.08, 0.15, DEFAULT_DISCOUNT_RATE, key="r")
with c2:
    margin_safety = st.slider("Margin of safety", 0.10, 0.40, DEFAULT_MARGIN_OF_SAFETY, key="mos")
with c3:
    min_div_years = st.selectbox("Min dividend points", [2, 3, 4], index=0, key="min_divs")

left, mid, right = st.columns([1, 1.2, 1])
with mid:
    run_clicked = st.button("Run valuation", type="primary", use_container_width=True)

b1, b2, b3 = st.columns([1, 1, 1])
with b1:
    sort_clicked = st.button("Sort A→Z", type="secondary", use_container_width=True)
with b2:
    clear_divs = st.button("Clear dividends", type="secondary", use_container_width=True)
with b3:
    clear_notes = st.button("Clear notes", type="secondary", use_container_width=True)

st.markdown("### 📌 Watchlist")

# Initialize watchlist (dynamic rows)
if "watchlist" not in st.session_state:
    st.session_state.watchlist = pd.DataFrame(
        [
            {
                "Ticker": t,
                "cmpy_id": CMPY_ID_MAP[t],
                "Dividends (annual totals, oldest→newest)": "",  # leave blank to auto-fetch
                "Notes": "",
            }
            for t in DEFAULT_TICKERS
        ]
    )

edited = st.data_editor(
    st.session_state.watchlist,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
)

# Apply utility actions
if sort_clicked:
    edited = edited.copy()
    edited["Ticker"] = edited["Ticker"].astype(str).str.upper().str.strip()
    edited = edited.sort_values("Ticker").reset_index(drop=True)

if clear_divs:
    edited = edited.copy()
    edited["Dividends (annual totals, oldest→newest)"] = ""

if clear_notes:
    edited = edited.copy()
    edited["Notes"] = ""

# Persist edits immediately
st.session_state.watchlist = edited

st.markdown("---")
st.markdown("### 📈 Results")

# -----------------------------
# Compute + store results (so charts don't reset)
# -----------------------------
def compute_results(rows: list[dict]) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    Returns:
      - results dataframe (one row per ticker)
      - row_map {ticker: {divs_annual, raw_payments_df, cmpy_id}}
      - updated_watchlist dataframe (with auto-filled annual totals where applicable)
    """
    results = []
    row_map = {}

    r = float(st.session_state.get("r", DEFAULT_DISCOUNT_RATE))
    mos = float(st.session_state.get("mos", DEFAULT_MARGIN_OF_SAFETY))
    min_pts = int(st.session_state.get("min_divs", 2))

    updated_rows = []

    for row in rows:
        row = dict(row)
        ticker = str(row.get("Ticker", "")).upper().strip()
        cmpy_id_raw = row.get("cmpy_id", "")
        notes = str(row.get("Notes", "")).strip()

        # Parse cmpy_id
        try:
            cmpy_id = int(cmpy_id_raw) if str(cmpy_id_raw).strip() else None
        except Exception:
            cmpy_id = None

        price = fetch_price_by_cmpy_id(cmpy_id) if cmpy_id else None

        # Manual annual totals
        manual_annual = parse_dividends(row.get("Dividends (annual totals, oldest→newest)", ""))
        annual_used = manual_annual
        raw_df = pd.DataFrame()
        source = ""

        if len(annual_used) >= 2:
            source = "manual"
        else:
            # Auto from dividends.ph raw payments -> annual totals
            if ticker:
                raw_df = fetch_dividendsph_raw(ticker)
                annual_auto = annual_totals_from_raw(raw_df, years=4)
                if len(annual_auto) >= 2:
                    annual_used = annual_auto
                    source = "dividends.ph"
                    # Auto-fill the watchlist cell with what we used
                    row["Dividends (annual totals, oldest→newest)"] = ", ".join([str(x) for x in annual_auto])

        trend = dividend_trend(annual_used)

        intrinsic = None
        buy_below = None
        status = "Insufficient data"

        if price is not None and len(annual_used) >= min_pts:
            g = estimate_growth_from_divs(annual_used)
            d1 = annual_used[-1] * (1 + g)
            intrinsic = ddm(d1, r, g)

            if intrinsic is not None:
                buy_below = intrinsic * (1 - mos)
                status = (
                    "Undervalued ✅"
                    if price < buy_below
                    else "Fairly Valued ✳️"
                    if price <= intrinsic
                    else "Overvalued ⚠️"
                )
            else:
                status = "Invalid (r ≤ g)"

        results.append(
            {
                "Ticker": ticker,
                "Price": price,
                "Dividend points (annual)": len(annual_used),
                "Dividend Trend (annual)": trend,
                "Intrinsic Value": intrinsic,
                "Buy Below": buy_below,
                "Valuation": status,
                "Dividend source": source,
                "Notes": notes,
            }
        )

        row_map[ticker] = {
            "divs_annual": annual_used,
            "raw_payments": raw_df,
            "cmpy_id": cmpy_id,
        }

        updated_rows.append(row)

    df = pd.DataFrame(results)
    for col in ["Price", "Intrinsic Value", "Buy Below"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(2)

    updated_watchlist = pd.DataFrame(updated_rows)
    return df, row_map, updated_watchlist

if run_clicked:
    rows = edited.fillna("").to_dict(orient="records")
    with st.spinner("Fetching prices + dividends + computing intrinsic values…"):
        df, row_map, updated_watchlist = compute_results(rows)

    st.session_state["results_df"] = df
    st.session_state["row_map"] = row_map
    st.session_state.watchlist = updated_watchlist

# Always show last computed results
df = st.session_state.get("results_df")
row_map = st.session_state.get("row_map", {})

if isinstance(df, pd.DataFrame) and not df.empty:
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download results (CSV)",
        data=csv,
        file_name="pse_valuation_results.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.markdown("### 📊 Charts + Raw Dividend Payments")

    tickers_available = df["Ticker"].dropna().unique().tolist()
    if "selected_ticker" not in st.session_state:
        st.session_state["selected_ticker"] = tickers_available[0] if tickers_available else ""

    selected = st.selectbox("Pick a ticker", tickers_available, key="selected_ticker")

    annual = row_map.get(selected, {}).get("divs_annual", [])
    raw_df = row_map.get(selected, {}).get("raw_payments", pd.DataFrame())

    sel_row = df[df["Ticker"] == selected].iloc[0] if not df[df["Ticker"] == selected].empty else None
    sel_price = float(sel_row["Price"]) if sel_row is not None and pd.notna(sel_row["Price"]) else None
    sel_intr = float(sel_row["Intrinsic Value"]) if sel_row is not None and pd.notna(sel_row["Intrinsic Value"]) else None
    sel_buy = float(sel_row["Buy Below"]) if sel_row is not None and pd.notna(sel_row["Buy Below"]) else None

    cA, cB = st.columns([1.2, 1.0])

    with cA:
        st.markdown("**Annual dividend totals used for DDM**")
        if len(annual) >= 2:
            x = list(range(1, len(annual) + 1))
            fig = plt.figure(facecolor="#111318")
            ax = plt.gca()
            ax.set_facecolor("#111318")
            ax.plot(x, annual, marker="o")
            ax.set_xlabel("Year index (oldest → newest)")
            ax.set_ylabel("Dividend / share (annual total)")
            ax.set_title(f"{selected} annual dividends (DDM input)")
            st.pyplot(fig)
        else:
            st.info("No annual dividend totals available. Enter manual annual dividends, or check dividends.ph.")

        st.markdown("**Raw dividend payments (from dividends.ph)**")
        if isinstance(raw_df, pd.DataFrame) and not raw_df.empty:
            show_n = st.slider("Show last N payments", 5, 50, 20, key=f"raw_n_{selected}")
            st.dataframe(raw_df.head(show_n), use_container_width=True)

            # Raw payments chart (rate over time) with MM-YY x-axis formatting
            raw_plot = raw_df.copy().sort_values("payment_date")

            fig_raw = plt.figure(facecolor="#111318")
            axr = plt.gca()
            axr.set_facecolor("#111318")

            axr.plot(raw_plot["payment_date"], raw_plot["rate"], marker="o")
            axr.set_xlabel("Payment date (MM-YY)")
            axr.set_ylabel("Dividend rate (per payment)")
            axr.set_title(f"{selected} raw dividend payments")

            # ✅ Format ticks to mm-yy and reduce clutter
            axr.xaxis.set_major_formatter(mdates.DateFormatter("%m-%y"))
            axr.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # show every 3 months
            plt.xticks(rotation=0)

            fig_raw.tight_layout()
            st.pyplot(fig_raw)
        else:
            st.info("Raw payments not found from dividends.ph for this ticker.")

    with cB:
        st.markdown("**Price vs intrinsic**")
        if sel_price is not None and sel_intr is not None and sel_buy is not None:
            labels = ["Price", "Intrinsic", "Buy Below"]
            vals = [sel_price, sel_intr, sel_buy]
            fig2 = plt.figure(facecolor="#111318")
            ax2 = plt.gca()
            ax2.set_facecolor("#111318")
            ax2.bar(labels, vals)
            ax2.set_ylabel("PHP")
            ax2.set_title(f"{selected} valuation")
            st.pyplot(fig2)
        else:
            st.info("Needs dividends + Run valuation to show intrinsic/buy-below chart.")

else:
    st.info("Click **Run valuation** to generate results. Charts will appear after that.")

st.markdown('<div class="muted">⚠️ Personal-use educational tool — not financial advice.</div>', unsafe_allow_html=True)

