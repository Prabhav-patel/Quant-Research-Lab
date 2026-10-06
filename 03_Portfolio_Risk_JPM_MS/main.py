"""
Portfolio Risk & Return Analysis  -  JPMorgan (JPM) & Morgan Stanley (MS)
=========================================================================
Resume project. Pulls REAL daily prices from Yahoo Finance, builds a two-stock
portfolio and computes the full risk/return toolkit: simple & log returns,
Sharpe, Sortino, maximum drawdown, Calmar (drawdown) ratio, year-on-year and
cumulative returns. Writes an attractive self-contained HTML report.

Run:  py main.py
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from report import Report, setup_plot_style

HERE = os.path.dirname(os.path.abspath(__file__))
yf.set_tz_cache_location(os.path.join(HERE, ".yfinance-cache"))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "outputs")
TICKERS = ["JPM", "MS"]
YEARS = "6y"
TRADING_DAYS = 252
setup_plot_style()

# ----------------------------------------------------------------------
# 1. DATA  -  fetch real adjusted close prices, cache to CSV
# ----------------------------------------------------------------------
def close_frame(raw, tickers):
    """Return a clean Close-price DataFrame regardless of yfinance layout."""
    if isinstance(raw.columns, pd.MultiIndex):
        px = raw["Close"].copy()
    else:
        px = raw[["Close"]].copy()
        px.columns = tickers
    return px[tickers].dropna()

def fetch():
    raw = yf.download(TICKERS, period=YEARS, auto_adjust=True, progress=False, threads=False)
    px = close_frame(raw, TICKERS)
    px.to_csv(os.path.join(DATA, "prices.csv"))
    # risk-free proxy: 13-week US T-bill yield (^IRX), quoted in percent
    try:
        irx = yf.download("^IRX", period=YEARS, auto_adjust=True, progress=False)
        irx_close = irx["Close"]
        rf = float(np.asarray(irx_close).ravel()[~np.isnan(np.asarray(irx_close).ravel())][-250:].mean()) / 100.0
    except Exception:
        rf = 0.037
    return px, rf

px, RF = fetch()
print(f"Loaded {px.shape[0]} trading days  {px.index[0].date()} -> {px.index[-1].date()}")
print(f"Risk-free (mean 13w T-bill): {RF:.4%}")

# ----------------------------------------------------------------------
# 2. RETURNS
# ----------------------------------------------------------------------
simple = px.pct_change(fill_method=None).dropna()          # arithmetic returns
logret = np.log(px / px.shift(1)).dropna()                 # log returns (time-additive)

WEIGHTS = np.array([0.5, 0.5])                             # equal-weight portfolio
port_simple = simple @ WEIGHTS                             # portfolio return = weighted avg of simple returns
port_log = np.log1p(port_simple)

# ----------------------------------------------------------------------
# 3. RISK / RETURN METRICS
# ----------------------------------------------------------------------
def ann_return(sr):                       # geometric (CAGR)
    growth = (1 + sr).prod()
    yrs = len(sr) / TRADING_DAYS
    return growth ** (1 / yrs) - 1

def ann_vol(sr):
    return sr.std(ddof=1) * np.sqrt(TRADING_DAYS)

def sharpe(sr, rf=RF):
    ex = sr - rf / TRADING_DAYS
    return (ex.mean() / sr.std(ddof=1)) * np.sqrt(TRADING_DAYS)

def sortino(sr, rf=RF):
    ex = sr - rf / TRADING_DAYS
    downside = np.minimum(ex, 0.0)
    dd = np.sqrt((downside ** 2).mean()) * np.sqrt(TRADING_DAYS)
    return (ex.mean() * TRADING_DAYS) / dd

def max_drawdown(sr):
    wealth = (1 + sr).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    return dd.min(), dd                    # worst dd, and the dd series

metrics = {}
series = {"JPM": simple["JPM"], "MS": simple["MS"], "Portfolio (50/50)": port_simple}
for name, sr in series.items():
    mdd, _ = max_drawdown(sr)
    metrics[name] = {
        "CAGR": ann_return(sr), "Vol": ann_vol(sr), "Sharpe": sharpe(sr),
        "Sortino": sortino(sr), "MaxDD": mdd, "Calmar": ann_return(sr) / abs(mdd),
        "CumReturn": (1 + sr).prod() - 1,
    }

corr = simple["JPM"].corr(simple["MS"])
# diversification check: does the portfolio vol beat the weighted average of the two?
wavg_vol = WEIGHTS[0] * ann_vol(simple["JPM"]) + WEIGHTS[1] * ann_vol(simple["MS"])
div_benefit = wavg_vol - ann_vol(port_simple)

print("\n--- Annualised metrics ---")
for n, m in metrics.items():
    print(f"{n:20s} CAGR {m['CAGR']:6.2%}  Vol {m['Vol']:6.2%}  Sharpe {m['Sharpe']:.2f}  "
          f"Sortino {m['Sortino']:.2f}  MaxDD {m['MaxDD']:6.2%}")
print(f"JPM-MS correlation: {corr:.2f}   diversification vol saving: {div_benefit:.2%}")

# year-on-year returns (per calendar year)
yoy = (1 + simple).groupby(simple.index.year).apply(lambda x: (1 + x).prod() - 1)
yoy["Portfolio"] = (1 + port_simple).groupby(port_simple.index.year).apply(lambda x: (1 + x).prod() - 1)

# ----------------------------------------------------------------------
# 4. PLOTS
# ----------------------------------------------------------------------
# (a) cumulative growth of $1
fig, ax = plt.subplots()
for name, sr in series.items():
    ((1 + sr).cumprod()).plot(ax=ax, lw=1.8 if "Port" in name else 1.3, label=name)
ax.axhline(1, color="#999", lw=.8, ls="--")
ax.set_title("Growth of $1 invested"); ax.set_ylabel("Portfolio value ($)"); ax.legend(frameon=False)
p_cum = os.path.join(OUT, "cumulative.png"); fig.savefig(p_cum); plt.close(fig)

# (b) portfolio drawdown
_, dd_series = max_drawdown(port_simple)
fig, ax = plt.subplots()
ax.fill_between(dd_series.index, dd_series.values * 100, 0, color="#C0392B", alpha=.35)
dd_series.mul(100).plot(ax=ax, color="#C0392B", lw=1)
ax.set_title("Portfolio drawdown (peak-to-trough)"); ax.set_ylabel("Drawdown (%)")
p_dd = os.path.join(OUT, "drawdown.png"); fig.savefig(p_dd); plt.close(fig)

# (c) yearly returns bar chart
fig, ax = plt.subplots()
yoy.mul(100).plot(kind="bar", ax=ax, width=.8)
ax.axhline(0, color="#333", lw=.8); ax.set_title("Year-on-year returns")
ax.set_ylabel("Return (%)"); ax.set_xlabel(""); ax.legend(frameon=False, ncol=3)
p_yoy = os.path.join(OUT, "yoy.png"); fig.savefig(p_yoy); plt.close(fig)

# (d) risk-return scatter (diversification visual)
fig, ax = plt.subplots(figsize=(7, 5))
for name, m in metrics.items():
    ax.scatter(m["Vol"] * 100, m["CAGR"] * 100, s=140)
    ax.annotate(name, (m["Vol"] * 100, m["CAGR"] * 100),
                textcoords="offset points", xytext=(8, 6), fontsize=10)
ax.set_title("Risk vs return  (up-left is better)")
ax.set_xlabel("Annualised volatility (%)"); ax.set_ylabel("Annualised return (%)")
p_rr = os.path.join(OUT, "risk_return.png"); fig.savefig(p_rr); plt.close(fig)

# ----------------------------------------------------------------------
# 5. HTML REPORT
# ----------------------------------------------------------------------
r = Report(
    "Portfolio Risk & Return Analysis: JPMorgan & Morgan Stanley",
    "A structured risk-assessment framework for a two-stock financial-sector portfolio, "
    "built entirely on real Yahoo Finance data.",
    tags=["JPM + MS", "Real market data", "Sharpe / Sortino", "Drawdown", "Python + pandas"],
    date=f"Data through {px.index[-1].date()}",
)
r.purpose(
    "<p>Banks JPMorgan and Morgan Stanley are highly correlated, so this project asks a "
    "practical portfolio question: <em>how much risk-adjusted performance do you actually get, "
    "and how much does combining two similar names really diversify?</em> The deliverable is a "
    "reusable framework that turns raw prices into the metrics a portfolio manager uses to make "
    "data-driven allocation decisions.</p>")

r.section(
    "Data & fetching procedure",
    "<p>Daily <strong>adjusted</strong> closing prices (adjusted for splits and dividends, so "
    "returns are total returns) are pulled straight from Yahoo Finance and cached locally.</p>"
    + r.formula(
        f"source     : Yahoo Finance via  yfinance.download([\"JPM\",\"MS\"], period=\"{YEARS}\")\n"
        f"window     : {px.index[0].date()}  ->  {px.index[-1].date()}   ({px.shape[0]} trading days)\n"
        f"risk-free  : 13-week US T-bill yield (^IRX), mean over window = {RF:.2%}\n"
        f"cached to  : data/prices.csv")
    + "<p>Using the <em>adjusted</em> close matters: it folds dividends back in, so a bank paying "
      "a fat dividend isn't unfairly penalised versus a growth name.</p>",
    num="01")

r.section(
    "Theory & derivations",
    "<h3>Returns — two flavours</h3>"
    + r.formula(
        "simple return   r_t = (P_t - P_{t-1}) / P_{t-1}\n"
        "log return      R_t = ln(P_t / P_{t-1})\n\n"
        "log returns are TIME-additive:   R(0->T) = sum of daily R_t\n"
        "simple returns are ASSET-additive: r_portfolio = w1*r1 + w2*r2")
    + "<p>I use simple returns to aggregate across the two stocks (a portfolio's return is the "
      "weighted average of its holdings' simple returns) and log returns where I need clean "
      "time-aggregation and better statistical behaviour.</p>"
    + "<h3>Risk-adjusted ratios</h3>"
    + r.formula(
        "Sharpe   = (R_p - R_f) / sigma_p                 annualised x sqrt(252)\n"
        "Sortino  = (R_p - R_f) / sigma_downside          penalises only downside moves\n"
        "MaxDD    = min over t of  (W_t / running_peak_t - 1)\n"
        "Calmar   = annual return / |MaxDD|               return per unit of worst loss\n\n"
        "downside deviation:  sigma_d = sqrt( mean( min(excess_ret, 0)^2 ) ) * sqrt(252)")
    + "<p>Sharpe punishes <em>all</em> volatility; Sortino only the downside — useful because "
      "investors don't mind upside 'risk'. Annualisation uses "
      "<code>&times;&radic;252</code> because variance grows linearly with time while volatility "
      "grows with its square root.</p>"
    + "<h3>Diversification</h3>"
    + r.formula(
        "sigma_portfolio = sqrt( w1^2 s1^2 + w2^2 s2^2 + 2 w1 w2 rho s1 s2 )\n"
        "benefit only exists when rho < 1  ->  the lower rho, the bigger the risk reduction"),
    num="02")

# KPI grid — headline portfolio numbers
pm = metrics["Portfolio (50/50)"]
r.raw(r.kpi_grid([
    ("Portfolio CAGR", f"{pm['CAGR']:.1%}", "annualised", "acc"),
    ("Volatility", f"{pm['Vol']:.1%}", "annualised", ""),
    ("Sharpe ratio", f"{pm['Sharpe']:.2f}", f"Rf = {RF:.1%}", "pos" if pm['Sharpe'] > 1 else ""),
    ("Sortino ratio", f"{pm['Sortino']:.2f}", "downside-only", "pos" if pm['Sortino'] > 1 else ""),
    ("Max drawdown", f"{pm['MaxDD']:.1%}", "worst peak-to-trough", "neg"),
    ("JPM-MS corr", f"{corr:.2f}", "high = limited diversification", ""),
]))

# full metrics table
rows = []
for n, m in metrics.items():
    rows.append([n, f"{m['CAGR']:.2%}", f"{m['Vol']:.2%}", f"{m['Sharpe']:.2f}",
                 f"{m['Sortino']:.2f}", f"{m['MaxDD']:.2%}", f"{m['Calmar']:.2f}",
                 f"{m['CumReturn']:.1%}"])
r.section(
    "Results on real data",
    r.table(["", "CAGR", "Ann. Vol", "Sharpe", "Sortino", "Max DD", "Calmar", "Total return"], rows)
    + f"<p>The 50/50 portfolio's volatility is <strong>{ann_vol(port_simple):.1%}</strong> versus a "
      f"weighted-average of <strong>{wavg_vol:.1%}</strong> for the two stocks held separately — a "
      f"diversification saving of only <strong>{div_benefit:.2%}</strong>. That small number is the "
      f"whole story: with a JPM-MS correlation of {corr:.2f}, mixing two big banks barely reduces risk.</p>"
    + r.figure(p_cum, "Growth of $1 — the portfolio tracks between its two constituents.")
    + r.figure(p_dd, "Portfolio drawdown — depth and duration of every decline from a prior peak.")
    + r.figure(p_yoy, "Year-on-year returns for each name and the blended portfolio.")
    + r.figure(p_rr, "Risk-return map — the portfolio sits between the two assets, only slightly left of the line joining them (weak diversification)."),
    num="03")

r.learned([
    "<strong>Correlation is the lever of diversification.</strong> Two high-correlation bank stocks "
    f"(&rho; = {corr:.2f}) give almost no volatility reduction — the portfolio vol saving was just {div_benefit:.2%}.",
    "<strong>Sharpe vs Sortino tell different stories.</strong> A name with sharp drawdowns can flatter "
    "its Sharpe; Sortino exposes the downside the Sharpe hides.",
    "<strong>Adjusted prices matter.</strong> Using unadjusted closes would understate total return for "
    "dividend-paying banks and distort every ratio.",
    "The <strong>&radic;252 annualisation</strong> and the time-additivity of log returns are the two "
    "details that make cross-horizon comparisons correct.",
])
r.inference(
    "<p>Over the sample window the 50/50 JPM-MS book delivered a Sharpe of "
    f"<strong>{pm['Sharpe']:.2f}</strong> and a Sortino of <strong>{pm['Sortino']:.2f}</strong> at "
    f"a {pm['Vol']:.1%} volatility, with a worst drawdown of {pm['MaxDD']:.1%}. But because the two "
    "names move together, the portfolio is really a <em>concentrated sector bet</em>, not a diversified "
    "one — a genuinely diversified book would pair these with an uncorrelated or negatively-correlated "
    "asset. The framework itself (returns &rarr; ratios &rarr; drawdown &rarr; YoY) is reusable for any "
    "basket of tickers.</p>")

r.save(os.path.join(HERE, "report.html"))
print("Done.")
