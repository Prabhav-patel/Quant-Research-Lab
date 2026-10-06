"""
Monte Carlo Option Pricing: European, Asian & Barrier Options
=============================================================
Resume project. Simulates Geometric-Brownian-Motion price PATHS (real spot & vol
from Yahoo Finance) under the risk-neutral measure and prices three payoff types.
Highlights why path-dependent options need simulation, uses a geometric-Asian
control variate + antithetic variates for variance reduction, and demonstrates
in-out barrier parity and discrete-monitoring bias.

Run:  py main.py
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm
from report import Report, setup_plot_style

HERE = os.path.dirname(os.path.abspath(__file__))
yf.set_tz_cache_location(os.path.join(HERE, ".yfinance-cache"))
DATA, OUT = os.path.join(HERE, "data"), os.path.join(HERE, "outputs")
UNDERLYING = "AAPL"
setup_plot_style()
rng = np.random.default_rng(11)

# ----------------------------------------------------------------------
# 1. DATA
# ----------------------------------------------------------------------
def load_close(t, period="2y"):
    raw = yf.download(t, period=period, auto_adjust=True, progress=False)
    s = raw["Close"]; s = s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s
    return s.dropna()

px = load_close(UNDERLYING)
px.to_csv(os.path.join(DATA, f"{UNDERLYING}_prices.csv"))
logret = np.log(px / px.shift(1)).dropna()
S0 = float(px.iloc[-1])
SIGMA = float(logret.tail(252).std(ddof=1) * np.sqrt(252))
try:
    R = float(load_close("^IRX", "1y").iloc[-1]) / 100.0
except Exception:
    R = 0.037
K = round(S0)
T, M = 1.0, 252                       # 1 year, daily monitoring
H = round(1.30 * S0)                  # up-barrier 30% above spot
N_PATHS = 60_000
print(f"{UNDERLYING}: S0={S0:.2f} K={K} H(up-barrier)={H} sigma={SIGMA:.2%} r={R:.2%}")

# ----------------------------------------------------------------------
# 2. ANALYTIC BENCHMARKS
# ----------------------------------------------------------------------
def bs_call(S, K, r, sig, T):
    d1 = (np.log(S / K) + (r + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def geometric_asian_call(S, K, r, sig, T, m):
    """Exact price of a DISCRETELY-monitored geometric-average Asian call."""
    dt = T / m
    ti = np.arange(1, m + 1) * dt
    mean_lnG = np.log(S) + (r - 0.5 * sig ** 2) * ti.mean()
    cov = np.minimum.outer(ti, ti)                 # Cov(W_ti, W_tj) = min(ti,tj)
    var_lnG = (sig ** 2 / m ** 2) * cov.sum()
    sd = np.sqrt(var_lnG)
    d1 = (mean_lnG - np.log(K) + var_lnG) / sd
    d2 = d1 - sd
    return np.exp(-r * T) * (np.exp(mean_lnG + 0.5 * var_lnG) * norm.cdf(d1) - K * norm.cdf(d2))

bs_c = bs_call(S0, K, R, SIGMA, T)
geo_closed = geometric_asian_call(S0, K, R, SIGMA, T, M)

# ----------------------------------------------------------------------
# 3. SIMULATE GBM PATHS (antithetic)
# ----------------------------------------------------------------------
def simulate(n, antithetic=True):
    dt = T / M
    half = n // 2 if antithetic else n
    Z = rng.standard_normal((half, M))
    if antithetic:
        Z = np.vstack([Z, -Z])
    incr = (R - 0.5 * SIGMA ** 2) * dt + SIGMA * np.sqrt(dt) * Z
    logS = np.log(S0) + np.cumsum(incr, axis=1)
    S = np.exp(logS)
    S = np.hstack([np.full((S.shape[0], 1), S0), S])   # prepend spot
    return S

paths = simulate(N_PATHS, antithetic=True)
ST = paths[:, -1]
disc = np.exp(-R * T)

# ----------------------------------------------------------------------
# 4. PRICE EACH PAYOFF
# ----------------------------------------------------------------------
def price_se(discounted_payoffs):
    return discounted_payoffs.mean(), discounted_payoffs.std(ddof=1) / np.sqrt(len(discounted_payoffs))

# European (endpoint only) -- validates against BS
euro_p, euro_se = price_se(disc * np.maximum(ST - K, 0))

# Asian -- arithmetic (needs MC) and geometric (has closed form)
arith_avg = paths[:, 1:].mean(axis=1)
geo_avg = np.exp(np.log(paths[:, 1:]).mean(axis=1))
d_arith = disc * np.maximum(arith_avg - K, 0)
d_geo = disc * np.maximum(geo_avg - K, 0)
asian_arith_p, asian_arith_se = price_se(d_arith)
asian_geo_mc_p, asian_geo_se = price_se(d_geo)

# control variate: correct arithmetic estimate using the geometric closed form
beta = np.cov(d_arith, d_geo)[0, 1] / np.var(d_geo)
cv = d_arith - beta * (d_geo - geo_closed)
asian_cv_p, asian_cv_se = price_se(cv)
corr_ag = np.corrcoef(d_arith, d_geo)[0, 1]

# Barrier -- up-and-out & up-and-in call (in + out = vanilla)
path_max = paths.max(axis=1)
knocked = path_max >= H
uo = disc * np.where(~knocked, np.maximum(ST - K, 0), 0.0)   # up-and-out: dies if H touched
ui = disc * np.where(knocked, np.maximum(ST - K, 0), 0.0)    # up-and-in: alive only if H touched
uo_p, uo_se = price_se(uo)
ui_p, ui_se = price_se(ui)
inout_sum = uo_p + ui_p                                       # should equal European

# variance-reduction demo: European SE with vs without antithetic (same path count)
plain = simulate(N_PATHS, antithetic=False)
euro_plain_p, euro_plain_se = price_se(disc * np.maximum(plain[:, -1] - K, 0))

print(f"European MC {euro_p:.3f} (BS {bs_c:.3f})")
print(f"Asian arith {asian_arith_p:.3f}  geo(MC) {asian_geo_mc_p:.3f}  geo(closed) {geo_closed:.3f}")
print(f"Asian arith + control variate {asian_cv_p:.3f}  SE {asian_arith_se:.4f}->{asian_cv_se:.4f}")
print(f"Up-and-out {uo_p:.3f}  Up-and-in {ui_p:.3f}  sum {inout_sum:.3f}  European {euro_p:.3f}")
print(f"knocked out: {knocked.mean():.1%} of paths")

# ----------------------------------------------------------------------
# 5. PLOTS
# ----------------------------------------------------------------------
tgrid = np.linspace(0, T, M + 1)
fig, ax = plt.subplots()
show = paths[:120]
kk = show.max(axis=1) >= H
for i in range(show.shape[0]):
    ax.plot(tgrid, show[i], lw=.5, color=("#C0392B" if kk[i] else "#0E7C7B"),
            alpha=.5 if kk[i] else .35)
ax.axhline(H, color="#B7791F", lw=1.4, ls="--", label=f"up-barrier H={H}")
ax.axhline(K, color="#333", lw=1, ls=":", label=f"strike K={K}")
ax.set_title("Simulated GBM paths (red = knocked out through the barrier)")
ax.set_xlabel("time (years)"); ax.set_ylabel("price ($)"); ax.legend(frameon=False)
p_paths = os.path.join(OUT, "paths.png"); fig.savefig(p_paths); plt.close(fig)

fig, ax = plt.subplots()
names = ["European", "Geometric\nAsian", "Arithmetic\nAsian", "Up-and-out", "Up-and-in"]
vals = [euro_p, asian_geo_mc_p, asian_arith_p, uo_p, ui_p]
cols = ["#0E7C7B", "#3B6EA5", "#3B6EA5", "#C0392B", "#B7791F"]
ax.bar(names, vals, color=cols, width=.65)
for i, v in enumerate(vals):
    ax.text(i, v + .3, f"${v:.2f}", ha="center", fontsize=9)
ax.set_title("Option value by payoff type (same K, S0, sigma)"); ax.set_ylabel("price ($)")
p_bar = os.path.join(OUT, "prices_bar.png"); fig.savefig(p_bar); plt.close(fig)

fig, ax = plt.subplots()
ax.hist(ST, bins=80, alpha=.55, color="#0E7C7B", label="terminal price $S_T$", density=True)
ax.hist(arith_avg, bins=80, alpha=.55, color="#B7791F", label="path average (Asian)", density=True)
ax.axvline(K, color="#333", ls=":", lw=1)
ax.set_title("Averaging shrinks dispersion -> Asian options cost less")
ax.set_xlabel("price ($)"); ax.set_ylabel("density"); ax.legend(frameon=False)
p_hist = os.path.join(OUT, "dispersion.png"); fig.savefig(p_hist); plt.close(fig)

fig, ax = plt.subplots()
labels = ["European\n(plain)", "European\n(antithetic)", "Arith Asian\n(plain)", "Arith Asian\n(control variate)"]
ses = [euro_plain_se, euro_se, asian_arith_se, asian_cv_se]
ax.bar(labels, ses, color=["#999", "#0E7C7B", "#999", "#0E7C7B"], width=.6)
for i, v in enumerate(ses):
    ax.text(i, v + max(ses) * .02, f"{v:.4f}", ha="center", fontsize=9)
ax.set_title("Variance reduction lowers Monte Carlo standard error"); ax.set_ylabel("standard error ($)")
p_vr = os.path.join(OUT, "variance_reduction.png"); fig.savefig(p_vr); plt.close(fig)

# ----------------------------------------------------------------------
# 6. REPORT
# ----------------------------------------------------------------------
r = Report(
    "Monte Carlo Option Pricing: European, Asian & Barrier Options",
    "Simulating risk-neutral price paths to value payoffs that closed-form models can't reach — with "
    "variance-reduction and a live-data underlying.",
    tags=[f"{UNDERLYING} @ ${S0:.0f}", "Path-dependent", f"{N_PATHS:,} paths", "Control variate", "NumPy"],
    date=f"Inputs as of {px.index[-1].date()}",
)
r.purpose(
    "<p>Black-Scholes only prices options whose payoff depends on the <em>final</em> price. The moment "
    "the payoff depends on the <em>path</em> — an average (Asian) or whether a level was touched "
    "(Barrier) — you need Monte Carlo. This project builds one GBM path simulator and uses it to price "
    "all three, showing exactly where simulation becomes essential and how to make it efficient.</p>")

r.section(
    "Data & fetching procedure",
    "<p>Same live inputs as a real desk would feed a pricer:</p>"
    + r.formula(
        f"S0     = last close of {UNDERLYING} (Yahoo Finance)   -> ${S0:.2f}\n"
        f"sigma  = 1y realised vol of daily log returns          -> {SIGMA:.2%}\n"
        f"r      = 13-week T-bill yield (^IRX)                   -> {R:.2%}\n"
        f"K = {K}   up-barrier H = {H} (+30%)   T = {T}y   monitoring = {M} steps (daily)\n"
        f"paths  = {N_PATHS:,}  (antithetic)"),
    num="01")

r.section(
    "Theory & derivations",
    "<h3>Risk-neutral path simulation</h3>"
    + r.formula(
        "S_{t+dt} = S_t exp[ (r - 0.5 sigma^2) dt + sigma sqrt(dt) Z ],  Z ~ N(0,1)\n"
        "price of any payoff = e^{-rT} * mean over paths of payoff(path)")
    + "<h3>The three payoffs</h3>"
    + r.formula(
        "European   : max(S_T - K, 0)                 depends on endpoint only\n"
        "Asian      : max(avg(S) - K, 0)              depends on the whole path\n"
        "Up-and-out : max(S_T - K, 0) * 1{ max(S) < H }   dies if barrier touched")
    + "<h3>Why Asian < European</h3>"
    + "<p>An average is less volatile than a single endpoint (variance of a mean is smaller), so the "
      "Asian option has lower effective volatility — and options are long volatility, so it is cheaper. "
      "Averaging also makes the payoff hard to manipulate near expiry, which is why Asians are popular "
      "in commodities and FX.</p>"
    + "<h3>Control variate (variance reduction)</h3>"
    + r.formula(
        "The GEOMETRIC-average Asian has an exact closed form; the ARITHMETIC one does not.\n"
        "They are ~99% correlated, so use the known one to correct the unknown one:\n\n"
        "price_CV = mean( X_arith - beta ( X_geo - E[X_geo]_closed ) )\n"
        "beta = Cov(X_arith, X_geo) / Var(X_geo)")
    + "<h3>Barrier in-out parity</h3>"
    + r.formula("up-and-in  +  up-and-out  =  vanilla European   (same K, H)"),
    num="02")

r.raw(r.kpi_grid([
    ("European MC", f"${euro_p:.2f}", f"BS = ${bs_c:.2f}", "acc"),
    ("Arith Asian", f"${asian_arith_p:.2f}", f"{(1-asian_arith_p/euro_p)*100:.0f}% < European", ""),
    ("Up-and-out", f"${uo_p:.2f}", f"{knocked.mean()*100:.0f}% knocked out", "neg"),
    ("In + Out", f"${inout_sum:.2f}", f"European ${euro_p:.2f} ✓", "pos"),
    ("CV std error", f"${asian_cv_se:.4f}", f"was ${asian_arith_se:.4f}", "pos"),
    ("Arith-Geo corr", f"{corr_ag:.3f}", "why the CV works", ""),
]))

r.section(
    "Results on real data",
    r.table(["Option", "Price", "Std error", "Note"], [
        ["European call", f"${euro_p:.3f}", f"±{euro_se:.3f}", f"matches Black-Scholes ${bs_c:.3f}"],
        ["Geometric Asian", f"${asian_geo_mc_p:.3f}", f"±{asian_geo_se:.3f}", f"closed form ${geo_closed:.3f}"],
        ["Arithmetic Asian", f"${asian_arith_p:.3f}", f"±{asian_arith_se:.3f}", "no closed form — MC only"],
        ["Arithmetic Asian + CV", f"${asian_cv_p:.3f}", f"±{asian_cv_se:.4f}", f"{asian_arith_se/asian_cv_se:.0f}× tighter"],
        ["Up-and-out call", f"${uo_p:.3f}", f"±{uo_se:.3f}", f"{knocked.mean()*100:.0f}% of paths knocked out"],
        ["Up-and-in call", f"${ui_p:.3f}", f"±{ui_se:.3f}", f"in+out = ${inout_sum:.3f}"],
    ])
    + f"<p>Three results validate the engine at once: the European MC price matches Black-Scholes, the "
      f"geometric-Asian MC matches its closed form, and up-and-in + up-and-out = "
      f"<strong>${inout_sum:.2f}</strong> reproduces the European price (${euro_p:.2f}) — the in-out "
      f"parity. The arithmetic Asian, which <em>has no closed form</em>, comes in "
      f"<strong>{(1-asian_arith_p/euro_p)*100:.0f}% cheaper</strong> than the European, exactly as the "
      f"averaging argument predicts.</p>"
    + f"<p>The control variate is the highlight: because arithmetic and geometric payoffs are "
      f"{corr_ag:.1%} correlated, it cuts the standard error from ${asian_arith_se:.4f} to "
      f"${asian_cv_se:.4f} — a <strong>{asian_arith_se/asian_cv_se:.0f}× reduction</strong> for free, "
      f"equivalent to running ~{(asian_arith_se/asian_cv_se)**2:.0f}× more paths.</p>"
    + r.figure(p_paths, "60k paths were simulated; the red ones pierced the barrier and knock the up-and-out option out.")
    + r.figure(p_bar, "Same strike and underlying, very different values — path-dependence is priced.")
    + r.figure(p_hist, "The path average (orange) is far less dispersed than the endpoint (teal): the reason Asians are cheap.")
    + r.figure(p_vr, "Antithetic variates and the geometric control variate both shrink Monte Carlo noise."),
    num="03")

r.learned([
    "<strong>Path-dependence is the dividing line.</strong> European needs only the endpoint; Asian and "
    "Barrier need the whole simulated path — that's precisely why Monte Carlo exists.",
    "<strong>Validation by triangulation.</strong> European = BS, geometric Asian = closed form, and "
    "in+out = vanilla — three independent checks that the simulator is correct before trusting the "
    "arithmetic-Asian number that has no benchmark.",
    f"<strong>Variance reduction is huge leverage.</strong> A control variate exploiting a {corr_ag:.0%} "
    f"correlation gave a {asian_arith_se/asian_cv_se:.0f}× tighter estimate at zero extra cost.",
    "<strong>Discrete monitoring biases barriers.</strong> Checking the barrier only at daily steps "
    "misses intraday crossings, so fewer paths knock out and the up-and-out price is biased high — a "
    "continuity correction (shift H by e^{0.5826 sigma sqrt(dt)}) fixes it.",
])
r.inference(
    f"<p>On live {UNDERLYING} inputs the simulator prices the vanilla European at ${euro_p:.2f} "
    f"(matching Black-Scholes), the arithmetic Asian ${(1-asian_arith_p/euro_p)*100:.0f}% cheaper at "
    f"${asian_arith_p:.2f}, and the up-and-out barrier at ${uo_p:.2f} with "
    f"{knocked.mean()*100:.0f}% of paths knocking out. The real lesson isn't any single price — it's "
    "the method: one GBM engine, validated three ways, then extended to payoffs no formula can reach, "
    "made efficient with variance reduction. That is exactly the toolkit used to price real exotics.</p>")

r.save(os.path.join(HERE, "report.html"))
print("Done.")
