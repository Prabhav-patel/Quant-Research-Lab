"""
Value at Risk, Expected Shortfall & Model Backtesting
=====================================================
Resume project. Builds an equal-weight portfolio from REAL Yahoo Finance prices
and estimates 1-day VaR three ways (Historical, Variance-Covariance, Monte Carlo),
adds Expected Shortfall (CVaR), then evaluates the model out-of-sample with a
rolling backtest scored by the Kupiec POF test and the Basel Traffic-Light test.

Run:  py main.py
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm, chi2
from report import Report, setup_plot_style

HERE = os.path.dirname(os.path.abspath(__file__))
yf.set_tz_cache_location(os.path.join(HERE, ".yfinance-cache"))
DATA, OUT = os.path.join(HERE, "data"), os.path.join(HERE, "outputs")
TICKERS = ["JPM", "MS", "AAPL"]
PV = 1_000_000                 # portfolio value ($)
setup_plot_style()
rng = np.random.default_rng(3)

# ----------------------------------------------------------------------
# 1. DATA -> equal-weight portfolio returns
# ----------------------------------------------------------------------
def close_frame(raw, tickers):
    px = raw["Close"].copy() if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if not isinstance(raw.columns, pd.MultiIndex):
        px.columns = tickers
    return px[tickers].dropna()

raw = yf.download(TICKERS, period="6y", auto_adjust=True, progress=False, threads=False)
px = close_frame(raw, TICKERS)
if px.empty:
    raise RuntimeError("No aligned price history was returned; check the data source before reporting results.")
px.to_csv(os.path.join(DATA, "prices.csv"))
rets = px.pct_change(fill_method=None).dropna()
w = np.repeat(1 / len(TICKERS), len(TICKERS))
port = rets @ w                                   # portfolio simple returns
port.to_csv(os.path.join(DATA, "portfolio_returns.csv"))
print(f"Portfolio: {TICKERS} equal weight, {len(port)} days, {port.index[0].date()}->{port.index[-1].date()}")

# ----------------------------------------------------------------------
# 2. VaR & ES  -  three methods
# ----------------------------------------------------------------------
def var_es_historical(r, c):
    q = np.percentile(r, (1 - c) * 100)           # left-tail quantile (negative)
    var = -q
    es = -r[r <= q].mean()
    return var, es

def var_es_parametric(r, c):
    mu, sig = r.mean(), r.std(ddof=1)
    z = norm.ppf(c)
    var = -(mu - z * sig)
    es = -(mu - sig * norm.pdf(z) / (1 - c))       # normal-ES closed form
    return var, es

def var_es_montecarlo(rets_df, weights, c, n=200_000):
    mu = rets_df.mean().values
    cov = rets_df.cov().values
    sims = rng.multivariate_normal(mu, cov, size=n)   # correlated asset returns
    psim = sims @ weights
    q = np.percentile(psim, (1 - c) * 100)
    return -q, -psim[psim <= q].mean()

table = {}
for c in (0.95, 0.99):
    hv, he = var_es_historical(port.values, c)
    pv_, pe = var_es_parametric(port.values, c)
    mv, me = var_es_montecarlo(rets, w, c)
    table[c] = dict(hist=(hv, he), param=(pv_, pe), mc=(mv, me))
    print(f"{c:.0%}  Hist VaR {hv:.3%} ES {he:.3%} | Param VaR {pv_:.3%} ES {pe:.3%} | MC VaR {mv:.3%} ES {me:.3%}")

# ----------------------------------------------------------------------
# 3. BACKTEST  -  rolling 252d historical VaR at 99%, out-of-sample
# ----------------------------------------------------------------------
C_BT = 0.99
WIN = 252
r = port.values
dates = port.index
var_series, actual = [], []
for t in range(WIN, len(r)):
    window = r[t - WIN:t]
    v = -np.percentile(window, (1 - C_BT) * 100)   # VaR estimated from past only
    var_series.append(v); actual.append(r[t])
var_series = np.array(var_series); actual = np.array(actual)
bt_dates = dates[WIN:]
exceptions = actual < -var_series                  # loss exceeded VaR
N, x = len(actual), int(exceptions.sum())
p = 1 - C_BT
expected = p * N

# Kupiec proportion-of-failures (LR) test
phat = x / N
log_null = (N - x) * np.log1p(-p) + x * np.log(p)
log_mle = ((N - x) * np.log1p(-phat) if x < N else 0.0) + (x * np.log(phat) if x > 0 else 0.0)
LR = -2 * (log_null - log_mle)
kupiec_pval = 1 - chi2.cdf(LR, 1)

# Basel traffic light is descriptive here: 99% VaR over the LAST 250 observations.
# It does not overturn a rejection by the full-sample Kupiec test.
last250 = exceptions[-250:]
ex250 = int(last250.sum())
zone = "GREEN" if ex250 <= 4 else ("YELLOW" if ex250 <= 9 else "RED")
print(f"Backtest: {N} days, {x} exceptions (expected {expected:.1f}), Kupiec LR={LR:.2f} p={kupiec_pval:.3f}")
print(f"Last 250d: {ex250} exceptions -> {zone} zone")

# ----------------------------------------------------------------------
# 4. PLOTS
# ----------------------------------------------------------------------
# (a) return distribution with VaR/ES marked (99% historical)
hv99, he99 = table[0.99]["hist"]
fig, ax = plt.subplots()
ax.hist(port.values * 100, bins=90, color="#0E7C7B", alpha=.55, density=True)
ax.axvline(-hv99 * 100, color="#C0392B", lw=1.6, label=f"99% VaR = {hv99:.2%}")
ax.axvline(-he99 * 100, color="#B7791F", lw=1.6, ls="--", label=f"99% ES = {he99:.2%}")
ax.set_title("Daily return distribution with 99% VaR and Expected Shortfall")
ax.set_xlabel("daily return (%)"); ax.set_ylabel("density"); ax.legend(frameon=False)
p_dist = os.path.join(OUT, "distribution.png"); fig.savefig(p_dist); plt.close(fig)

# (b) method comparison bar
fig, ax = plt.subplots()
methods = ["Historical", "Variance-\nCovariance", "Monte Carlo"]
v95 = [table[0.95][k][0] * 100 for k in ("hist", "param", "mc")]
v99 = [table[0.99][k][0] * 100 for k in ("hist", "param", "mc")]
xp = np.arange(3)
ax.bar(xp - .2, v95, .4, label="95% VaR", color="#0E7C7B")
ax.bar(xp + .2, v99, .4, label="99% VaR", color="#C0392B")
ax.set_xticks(xp); ax.set_xticklabels(methods); ax.set_ylabel("1-day VaR (% of portfolio)")
ax.set_title("VaR by method and confidence level"); ax.legend(frameon=False)
p_meth = os.path.join(OUT, "methods.png"); fig.savefig(p_meth); plt.close(fig)

# (c) backtest: losses vs VaR line with exceptions
fig, ax = plt.subplots(figsize=(9.5, 4.6))
ax.plot(bt_dates, -actual * 100, color="#9aa4b1", lw=.6, label="daily loss")
ax.plot(bt_dates, var_series * 100, color="#0E7C7B", lw=1.2, label="99% VaR (rolling)")
ax.scatter(bt_dates[exceptions], -actual[exceptions] * 100, color="#C0392B", s=16, zorder=5,
           label=f"exceptions ({x})")
ax.set_title("VaR backtest — losses breaching the VaR line are exceptions")
ax.set_ylabel("loss (%)"); ax.legend(frameon=False, ncol=3)
p_bt = os.path.join(OUT, "backtest.png"); fig.savefig(p_bt); plt.close(fig)

# ----------------------------------------------------------------------
# 5. REPORT
# ----------------------------------------------------------------------
rp = Report(
    "Value at Risk, Expected Shortfall & Model Backtesting",
    "Estimating downside risk for a real multi-asset portfolio three different ways, then proving "
    "whether the model actually holds up out-of-sample.",
    tags=[f"{'+'.join(TICKERS)}", f"${PV:,.0f} book", "3 VaR methods", "Kupiec + Traffic Light", "SciPy"],
    date=f"Data through {px.index[-1].date()}",
)
rp.purpose(
    "<p>A risk number nobody has tested is worthless. This project computes 1-day Value at Risk for a "
    "real portfolio three standard ways, upgrades it to Expected Shortfall to capture tail severity, "
    "and then does the part most people skip — <em>backtesting</em>: letting the model predict day by "
    "day on unseen data and formally scoring how often reality broke through it.</p>")

rp.section(
    "Data & fetching procedure",
    "<p>An equal-weight book of three liquid US names, priced daily from Yahoo Finance:</p>"
    + rp.formula(
        f"holdings   : {', '.join(TICKERS)}  (equal weight, ${PV:,.0f} total)\n"
        f"window     : {port.index[0].date()} -> {port.index[-1].date()}  ({len(port)} trading days)\n"
        f"returns    : daily simple returns, portfolio = weighted sum\n"
        f"cached to  : data/prices.csv, data/portfolio_returns.csv"),
    num="01")

rp.section(
    "Theory & derivations",
    "<h3>Value at Risk — three estimators</h3>"
    + rp.formula(
        "VaR_c = the loss you exceed only (1-c) of the time over the horizon\n\n"
        "Historical      : take the (1-c) empirical quantile of past returns   (no distribution assumed)\n"
        "Variance-Covar. : VaR = -(mu - z_c sigma),  z_99 = 2.326, z_95 = 1.645  (assumes normality)\n"
        "Monte Carlo     : simulate returns ~ N(mu, Sigma), revalue, take the quantile")
    + "<h3>Expected Shortfall (a.k.a. CVaR) — the upgrade</h3>"
    + rp.formula(
        "ES_c = E[ Loss | Loss > VaR_c ]   = the AVERAGE loss in the worst (1-c) tail\n\n"
        "VaR is NOT sub-additive (can penalise diversification); ES IS coherent.\n"
        "Basel moved market-risk capital from 99% VaR to 97.5% ES for exactly this reason.")
    + "<h3>Backtesting</h3>"
    + rp.formula(
        "Exception = a day whose loss exceeded that day's predicted VaR.\n"
        "At 99% over N days you EXPECT (1-c)*N exceptions.\n\n"
        "Kupiec POF (LR) test  ~ chi-square(1);  reject model if too many/few exceptions\n"
        "Basel Traffic Light (99%, 250d):  GREEN 0-4   YELLOW 5-9   RED 10+"),
    num="02")

hv, he = table[0.99]["hist"]
rp.raw(rp.kpi_grid([
    ("99% VaR (1-day)", f"{hv:.2%}", f"${hv*PV:,.0f}", "neg"),
    ("99% Expected Shortfall", f"{he:.2%}", f"${he*PV:,.0f}", "neg"),
    ("Exceptions", f"{x} / {N}", f"expected {expected:.0f}", ""),
    ("Kupiec p-value", f"{kupiec_pval:.3f}", "model OK" if kupiec_pval > 0.05 else "model rejected",
     "pos" if kupiec_pval > 0.05 else "neg"),
    ("Traffic light", zone, f"{ex250} in last 250d",
     "pos" if zone == "GREEN" else ("" if zone == "YELLOW" else "neg")),
    ("95% VaR", f"{table[0.95]['hist'][0]:.2%}", f"${table[0.95]['hist'][0]*PV:,.0f}", ""),
]))

rp.section(
    "Results on real data",
    rp.table(["Confidence", "Method", "VaR (%)", "VaR ($)", "ES (%)", "ES ($)"],
             [[f"{int(c*100)}%", name, f"{table[c][k][0]:.2%}", f"${table[c][k][0]*PV:,.0f}",
               f"{table[c][k][1]:.2%}", f"${table[c][k][1]*PV:,.0f}"]
              for c in (0.95, 0.99)
              for name, k in [("Historical", "hist"), ("Variance-Covariance", "param"), ("Monte Carlo", "mc")]])
    + "<p>The three methods broadly agree, but note the pattern: the parametric (normal) method tends to "
      "sit <em>below</em> the historical method in the deep tail, because real returns are fat-tailed and "
      "a normal distribution understates extreme losses. Expected Shortfall is always larger than VaR at "
      "the same confidence — it is the average of everything beyond the threshold.</p>"
    + f"<p><strong>Backtest verdict:</strong> over {N} out-of-sample days the 99% model was breached "
      f"<strong>{x}</strong> times versus <strong>{expected:.0f}</strong> expected. The Kupiec test "
      f"gives a p-value of <strong>{kupiec_pval:.3f}</strong> "
      f"({'fail to reject — the model is statistically acceptable' if kupiec_pval>0.05 else 'reject — the model miscounts risk'}), "
      f"and the last 250 days land <strong>{ex250}</strong> exceptions in the "
      f"<strong>{zone}</strong> Basel zone.</p>"
    + rp.figure(p_dist, "Where 99% VaR and ES sit in the real return distribution — ES lives deeper in the tail.")
    + rp.figure(p_meth, "VaR rises with confidence; methods diverge most in the 99% tail.")
    + rp.figure(p_bt, "Out-of-sample backtest: red points are days the loss broke through the rolling VaR."),
    num="03")

rp.learned([
    "<strong>VaR answers 'how bad on a normal-ish bad day'; ES answers 'how bad when it's genuinely "
    "bad'.</strong> ES is coherent (sub-additive) where VaR is not — the reason regulators prefer it.",
    "<strong>Normality understates tails.</strong> The variance-covariance method is the simplest but "
    "the least conservative in the deep tail, because markets have fatter tails than a Gaussian.",
    "<strong>A VaR model must be evaluated out of sample.</strong> Exception counts, Kupiec and the "
    "250-day traffic light can disagree because they use different windows; report both verdicts.",
    "<strong>Clustering matters.</strong> Exceptions bunch together in stress periods — a hint that "
    "constant-VaR assumptions miss volatility clustering (motivating GARCH-based VaR).",
])
rp.inference(
    f"<p>The portfolio's 1-day 99% VaR is about <strong>{hv:.1%}</strong> "
    f"(~${hv*PV:,.0f} on a ${PV:,.0f} book), with Expected Shortfall of <strong>{he:.1%}</strong> "
    f"deeper in the tail. The rolling historical VaR produced {x} exceptions versus {expected:.0f} "
    f"expected over {N} observations. Kupiec p = {kupiec_pval:.3f} "
    f"({'fails' if kupiec_pval < 0.05 else 'does not reject'} the 5% calibration test); "
    f"the separate last-250-day traffic-light result is {zone} ({ex250} exceptions). "
    "These are different windows, not interchangeable approvals. Exception clustering in stress "
    "periods motivates testing volatility-aware alternatives; it does not establish their superiority.</p>")

rp.save(os.path.join(HERE, "report.html"))
print("Done.")
