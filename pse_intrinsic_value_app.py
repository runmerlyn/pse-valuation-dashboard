import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FixedLocator

plt.style.use("dark_background")

# -----------------------------
# Logo file (put this in same folder as the .py in GitHub)
# -----------------------------
LOGO_PATH = "MIA logo 5.png"

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
st.set_page_config(page_title="MIA Stock & Dividend Dashboard", page_icon="📈", layout="wide")

st.markdown(
    """
<style>
.block-container { padding-top: 1.0rem; padding-bottom: 2.2rem; max-width: 1200px; }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.muted { opacity: 0.76; font-size: 0.93rem; line-height: 1.55; }

.stButton>button, .stDownloadButton>button, .stLinkButton>button {
  border-radius: 14px !important;
  padding: 10px 14px !important;
  border: 1px solid rgba(120,120,120,0.25) !important;
}
.stButton>button:hover, .stDownloadButton>button:hover, .stLinkButton>button:hover {
  transform: translateY(-1px);
  transition: 120ms ease;
}
button[kind="primary"] {
  background: #FFD54A !important;
  color: #1b1b1b !important;
  border: 0 !important;
  box-shadow: 0 10px 24px rgba(255, 213, 74, 0.25) !important;
}
.stButton>button[kind="secondary"] { background: rgba(255,255,255,0.06) !important; }

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
# Title row with logo
# -----------------------------
tcol1, tcol2 = st.columns([0.18, 0.82], vertical_alignment="center")
with tcol1:
    try:
        st.image(LOGO_PATH, width=120)
    except Exception:
        st.write("")
with tcol2:
    st.markdown("## MIA Stock & Dividend Dashboard")

# -----------------------------
# Top explanation
# -----------------------------
st.markdown(
    """
<div class="muted">
<b>Why valuation is used:</b> Market price can be above or below what a stock is worth based on its ability to return cash to investors.
Valuation compares <b>price</b> vs an estimated <b>intrinsic value</b> to help you avoid overpaying and identify possible bargains.

<br><br>
<b>Method used here (Dividend Discount Model — Gordon Growth):</b><br>
<b>Intrinsic Value = D1 / (r − g)</b>

<br><br>
<b>Variables:</b>
<ul>
<li><b>D1</b>: next year’s dividend per share (estimated as last annual dividend × (1 + g))</li>
<li><b>r</b>: discount rate / required return (slider)</li>
<li><b>g</b>: dividend growth rate (estimated from annual dividend totals, capped for stability)</li>
</ul>

<b>Why helpful:</b> For dividend payers (REITs, mature companies), dividends are a real cash return. DDM gives a quick “is this price reasonable?” check versus dividend strength and growth assumptions.

<br><br>
<b>Data sources:</b> Stock snapshot & price from <b>PSE Edge</b> (hidden company id). Raw dividend payments from <b>dividends.ph</b> (last 10 max).
</div>
""",
    unsafe_allow_html=True,
)

link_c1, link_c2, _ = st.columns([1.2, 1.2, 2])
with link_c1:
    st.link_button("PSE Edge Dividend List", "https://edge.pse.com.ph/disclosureData/dividends_and_rights_info_list.ax", use_container_width=True)
with link_c2:
    st.link_button("dividends.ph", "https://dividends.ph/", use_container_width=True)

# -----------------------------
# Add ticker + Advanced mapping
# -----------------------------
with st.expander("Add Ticker Wizard"):
    colA, colB, colC = st.columns([1, 1, 1])
    with colA:
        new_ticker = st.text_input("Ticker (e.g. MER)", value="", key="wiz_ticker").upper().strip()
    with colB:
        new_cmpy = st.text_input("Company ID (cmpy_id) number", value="", key="wiz_cmpy")
    with colC:
        add_clicked = st.button("Add", type="secondary", use_container_width=True)

    st.caption("Find company id from URL: https://edge.pse.com.ph/companyPage/stockData.do?cmpy_id=118")

    if add_clicked:
        if not new_ticker or not new_cmpy.strip().isdigit():
            st.error("Please enter a ticker and a numeric company id.")
        else:
            st.session_state.cmpy_id_map[new_ticker] = int(new_cmpy.strip())
            st.session_state.watchlist = sync_watchlist_with_mapping(st.session_state.watchlist, st.session_state.cmpy_id_map)
            st.success(f"Added/updated {new_ticker}.")

with st.expander("Advanced: view/edit company id mapping"):
    map_df = pd.DataFrame([{"Ticker": t, "Company ID": cid} for t, cid in sorted(st.session_state.cmpy_id_map.items())])
    edited_map = st.data_editor(map_df, num_rows="dynamic", use_container_width=True, hide_index=True)
    if st.button("Save company id changes", type="secondary"):
        new_map = {}
        for r in edited_map.fillna("").to_dict(orient="records"):
            t = normalize_ticker(r.get("Ticker", ""))
            c = str(r.get("Company ID", "")).strip()
            if t and c.isdigit():
                new_map[t] = int(c)
        if new_map:
            st.session_state.cmpy_id_map.update(new_map)
            st.session_state.watchlist = sync_watchlist_with_mapping(st.session_state.watchlist, st.session_state.cmpy_id_map)
            st.success("Saved mapping + synced watchlist.")

st.markdown("---")

# -----------------------------
# Controls
# -----------------------------
c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    st.slider("Discount rate (r)", 0.08, 0.15, DEFAULT_DISCOUNT_RATE, key="r")
with c2:
    st.slider("Margin of safety", 0.10, 0.40, DEFAULT_MARGIN_OF_SAFETY, key="mos")
with c3:
    st.selectbox("Min dividend points", [2, 3, 4], index=0, key="min_divs")

left, mid, right = st.columns([1, 1.2, 1])
with mid:
    run_clicked = st.button("Run valuation", type="primary", use_container_width=True)

st.markdown("---")

# -----------------------------
# Updated Watchlist
# -----------------------------
st.markdown("### Updated Watchlist")
st.markdown(
    '<div class="muted">'
    '<b>Manual dividends:</b> If dividends.ph has no data, type annual totals (oldest → newest) like <code>2.5, 3.0, 3.2, 3.4</code>. '
    'If left blank, the app tries to derive annual totals from dividends.ph raw payments.'
    '</div>',
    unsafe_allow_html=True
)

wl = st.session_state.watchlist.copy()
wl["Ticker"] = wl["Ticker"].astype(str).map(normalize_ticker)
st.session_state.watchlist = st.data_editor(wl, num_rows="dynamic", use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("### Results")

# -----------------------------
# Compute
# -----------------------------
def compute_results(rows: list[dict]) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    results = []
    updated_rows = []
    per_ticker = {}

    r = float(st.session_state.get("r", DEFAULT_DISCOUNT_RATE))
    mos = float(st.session_state.get("mos", DEFAULT_MARGIN_OF_SAFETY))
    min_pts = int(st.session_state.get("min_divs", 2))
    id_map = st.session_state.cmpy_id_map

    for row in rows:
        row = dict(row)
        ticker = normalize_ticker(row.get("Ticker", ""))
        notes = str(row.get("Notes", "")).strip()

        cmpy_id = id_map.get(ticker)
        snapshot = fetch_stock_snapshot(cmpy_id) if isinstance(cmpy_id, int) else {}
        price = snapshot.get("Last Traded Price")

        raw_df = fetch_dividendsph_raw(ticker) if ticker else pd.DataFrame()

        manual_annual = parse_dividends(row.get("Dividends (annual totals, oldest→newest)", ""))
        annual_used = manual_annual
        source = "manual" if len(manual_annual) >= 2 else ""

        if len(annual_used) < 2:
            annual_auto = annual_totals_from_raw(raw_df, years=4)
            if len(annual_auto) >= 2:
                annual_used = annual_auto
                source = "dividends.ph"
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

        per_ticker[ticker] = {
            "snapshot": snapshot,
            "divs_annual": annual_used,
            "raw_payments": raw_df,
        }
        updated_rows.append(row)

    df = pd.DataFrame(results)
    for col in ["Price", "Intrinsic Value", "Buy Below"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(2)

    return df, per_ticker, pd.DataFrame(updated_rows)

if run_clicked:
    st.session_state.watchlist = sync_watchlist_with_mapping(st.session_state.watchlist, st.session_state.cmpy_id_map)
    rows = st.session_state.watchlist.fillna("").to_dict(orient="records")
    with st.spinner("Fetching data and computing valuation..."):
        df, per_ticker, updated_watchlist = compute_results(rows)
    st.session_state["results_df"] = df
    st.session_state["per_ticker"] = per_ticker
    st.session_state.watchlist = updated_watchlist

df = st.session_state.get("results_df")
per_ticker = st.session_state.get("per_ticker", {})

if isinstance(df, pd.DataFrame) and not df.empty:
    st.dataframe(df, use_container_width=True)
    st.download_button(
        "Download results (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="pse_valuation_results.csv",
        mime="text/csv"
    )

    st.markdown("---")
    st.markdown("## Stock & Dividend Data")

    tickers_available = df["Ticker"].dropna().unique().tolist()
    if "selected_ticker" not in st.session_state:
        st.session_state["selected_ticker"] = tickers_available[0] if tickers_available else ""
    selected = st.selectbox("Choose ticker", tickers_available, key="selected_ticker")

    info = per_ticker.get(selected, {})
    snapshot = info.get("snapshot", {}) or {}
    annual = info.get("divs_annual", []) or []
    raw_df = info.get("raw_payments", pd.DataFrame())

    sel_row = df[df["Ticker"] == selected].iloc[0] if not df[df["Ticker"] == selected].empty else None
    sel_price = float(sel_row["Price"]) if sel_row is not None and pd.notna(sel_row["Price"]) else None
    sel_intr = float(sel_row["Intrinsic Value"]) if sel_row is not None and pd.notna(sel_row["Intrinsic Value"]) else None
    sel_buy = float(sel_row["Buy Below"]) if sel_row is not None and pd.notna(sel_row["Buy Below"]) else None

    cA, cB = st.columns([1.7, 1.0])

    with cA:
        st.markdown("### Stock Snapshot (PSE Edge)")
        snap_df = pd.DataFrame([{"Metric": k, "Value": v} for k, v in snapshot.items() if v is not None])
        if snap_df.empty:
            st.info("No snapshot fields parsed for this ticker.")
        else:
            snap_df["Value"] = snap_df["Value"].apply(lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x)
            st.dataframe(snap_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### Dividends")

        st.markdown("**Annual dividend totals used for DDM**")
        if len(annual) >= 2:
            x = list(range(1, len(annual) + 1))
            fig = plt.figure(facecolor="#111318")
            ax = plt.gca()
            ax.set_facecolor("#111318")
            ax.plot(x, annual, marker="o")
            ax.set_xlabel("Year index")
            ax.set_ylabel("Dividend / share (annual total)")
            ax.set_title(f"{selected} annual dividends (DDM input)")
            fig.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No annual totals available (enter manual annual dividends).")

        st.markdown("**Raw dividend payments from dividends.ph (last 10 max)**")
        if isinstance(raw_df, pd.DataFrame) and not raw_df.empty:
            raw10 = raw_df.copy().sort_values("payment_date", ascending=False).head(10).reset_index(drop=True)
            raw10_for_plot = raw10.sort_values("payment_date").reset_index(drop=True)

            raw10_for_plot["Dividend #"] = range(1, len(raw10_for_plot) + 1)

            left_raw, right_raw = st.columns([1.2, 1.0])

            with left_raw:
                fig_raw = plt.figure(facecolor="#111318")
                axr = plt.gca()
                axr.set_facecolor("#111318")

                axr.plot(raw10_for_plot["payment_date"], raw10_for_plot["rate"], marker="o", linewidth=1)

                axr.set_title(f"{selected} raw dividends (last {len(raw10_for_plot)})", fontsize=10)
                axr.set_xlabel("Payment date (MM-YY)", fontsize=7)
                axr.set_ylabel("Dividend rate", fontsize=7)

                x_dates = mdates.date2num(raw10_for_plot["payment_date"].dt.to_pydatetime())
                axr.xaxis.set_major_locator(FixedLocator(x_dates))
                axr.xaxis.set_major_formatter(mdates.DateFormatter("%m-%y"))

                axr.tick_params(axis="x", labelsize=6)
                axr.tick_params(axis="y", labelsize=7)

                for _, rrow in raw10_for_plot.iterrows():
                    axr.annotate(
                        str(int(rrow["Dividend #"])),
                        (rrow["payment_date"], rrow["rate"]),
                        textcoords="offset points",
                        xytext=(0, 6),
                        ha="center",
                        fontsize=7,
                    )

                fig_raw.autofmt_xdate(rotation=0)
                fig_raw.tight_layout()
                st.pyplot(fig_raw)

            with right_raw:
                raw10_tbl = raw10_for_plot.copy()
                raw10_tbl["Payment (MM-YY)"] = raw10_tbl["payment_date"].dt.strftime("%m-%y")
                raw10_tbl["Ex (MM-YY)"] = raw10_tbl["ex_date"].dt.strftime("%m-%y")
                raw10_tbl = raw10_tbl[["Dividend #", "Payment (MM-YY)", "Ex (MM-YY)", "rate", "year"]]
                raw10_tbl = raw10_tbl.rename(columns={"rate": "Dividend rate", "year": "Year"})
                st.dataframe(raw10_tbl, use_container_width=True, hide_index=True)
        else:
            st.warning("No raw dividend records found on dividends.ph for this ticker.")

    with cB:
        st.markdown("### Valuation")
        if sel_price is not None and sel_intr is not None and sel_buy is not None:
            labels = ["Price", "Intrinsic", "Buy Below"]
            vals = [sel_price, sel_intr, sel_buy]
            fig2 = plt.figure(facecolor="#111318")
            ax2 = plt.gca()
            ax2.set_facecolor("#111318")
            ax2.bar(labels, vals)
            ax2.set_ylabel("PHP")
            ax2.set_title(f"{selected} valuation")
            fig2.tight_layout()
            st.pyplot(fig2)
        else:
            st.info("Run valuation + ensure enough dividend history to show intrinsic/buy-below.")
else:
    st.info("Click Run valuation to generate results.")

st.markdown('<div class="muted">⚠️ Personal-use educational tool — not financial advice.</div>', unsafe_allow_html=True)
