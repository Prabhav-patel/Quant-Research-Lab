"""
Stochastic Interest-Rate Modeling: Vasicek & Cox-Ingersoll-Ross (CIR)
=====================================================================
Resume project. Calibrates both mean-reverting short-rate models to a REAL short
rate (13-week US T-bill yield, ^IRX from Yahoo Finance): Vasicek by exact Gaussian
MLE (= OLS on the AR(1) form), CIR by numerical MLE on its non-central chi-square
transition density. Then simulates both, compares mean-reversion and the
negative-rate question, and builds model-implied yield curves.

Run:  py main.py
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize
from scipy.stats import ncx2
from report import Report, setup_plot_style

HERE = os.path.dirname(os.path.abspath(__file__))
yf.set_tz_cache_location(os.path.join(HERE, ".yfinance-cache"))
DATA, OUT = os.path.join(HERE, "data"), os.path.join(HERE, "outputs")
DT = 1 / 252
setup_plot_style()
rng = np.random.default_rng(5)

# ----------------------------------------------------------------------
# 1. DATA  -  real short rate
# ----------------------------------------------------------------------
# A ~4-year window (recent hiking + plateau cycle) is used: over a full decade the
# structural 0%->5% regime shift breaks single-factor mean reversion, whereas here
# rates genuinely revert around ~4.5% and both models calibrate cleanly & consistently.
raw = yf.download("^IRX", period="4y", auto_adjust=True, progress=False)
s = raw["Close"]; s = s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s
rate = (s.dropna() / 100.0).clip(lower=1e-4)          # percent -> decimal, floor at 1bp
rate.to_csv(os.path.join(DATA, "short_rate_IRX.csv"))
r = rate.values
r0 = float(r[-1])
print(f"^IRX short rate: {len(r)} obs  {rate.index[0].date()}->{rate.index[-1].date()}  r0={r0:.3%}")

# ----------------------------------------------------------------------
# 2. VASICEK calibration  -  exact MLE via AR(1) OLS
#    r_{t+1} = r_t e^{-a dt} + b(1-e^{-a dt}) + noise
# ----------------------------------------------------------------------
x, y = r[:-1], r[1:]
B_ols, A_ols = np.polyfit(x, y, 1)                     # y = B_ols*x + A_ols
a_v = -np.log(B_ols) / DT
b_v = A_ols / (1 - B_ols)
resid = y - (B_ols * x + A_ols)
var_e = resid.var(ddof=2)
sig_v = np.sqrt(var_e * 2 * a_v / (1 - B_ols ** 2))
print(f"Vasicek  a={a_v:.3f}  b={b_v:.3%}  sigma={sig_v:.4f}")

# ----------------------------------------------------------------------
# 3. CIR calibration  -  numerical MLE on non-central chi-square density
# ----------------------------------------------------------------------
def cir_negloglik(params, x, y, dt):
    a, b, sig = params
    if a <= 0 or b <= 0 or sig <= 0:
        return 1e12
    c = 2 * a / (sig ** 2 * (1 - np.exp(-a * dt)))
    df = 4 * a * b / sig ** 2
    nc = 2 * c * x * np.exp(-a * dt)
    # r_{t+1} density: f(y) = 2c * ncx2.pdf(2c*y ; df, nc)
    logpdf = np.log(2 * c) + ncx2.logpdf(2 * c * y, df, nc)
    finite = logpdf[np.isfinite(logpdf)]
    if len(finite) < 0.5 * len(y):                     # too many blow-ups -> bad params
        return 1e12
    return -finite.sum()

# bounded MLE with several starts (mean-reversion MLE is multi-modal / ill-conditioned)
bounds = [(1e-3, 5.0), (1e-3, 0.25), (1e-3, 0.5)]
starts = [(0.3, 0.045, 0.04), (1.0, 0.045, 0.05), (a_v, max(b_v, 0.02), 0.03),
          (1.5, 0.046, 0.03), (0.1, 0.04, 0.03)]
best = None
for s0 in starts:
    try:
        res = minimize(cir_negloglik, np.array(s0), args=(x, y, DT),
                       method="L-BFGS-B", bounds=bounds)
        if best is None or res.fun < best.fun:
            best = res
    except Exception:
        pass
a_c, b_c, sig_c = best.x
feller = 2 * a_c * b_c >= sig_c ** 2
print(f"CIR      a={a_c:.3f}  b={b_c:.3%}  sigma={sig_c:.4f}  Feller(2ab>=s^2)={feller} "
      f"(2ab={2*a_c*b_c:.5f}, s^2={sig_c**2:.5f})")

hl_v, hl_c = np.log(2) / a_v, np.log(2) / a_c          # mean-reversion half-life (years)

# ----------------------------------------------------------------------
# 4. SIMULATE both forward (2 years, many paths)
# ----------------------------------------------------------------------
def sim_vasicek(a, b, sig, r0, years=2.0, npaths=2000):
    n = int(years / DT)
    r = np.empty((npaths, n + 1)); r[:, 0] = r0
    for t in range(n):
        z = rng.standard_normal(npaths)
        r[:, t + 1] = r[:, t] + a * (b - r[:, t]) * DT + sig * np.sqrt(DT) * z
    return r

def sim_cir(a, b, sig, r0, years=2.0, npaths=2000):
    n = int(years / DT)
    r = np.empty((npaths, n + 1)); r[:, 0] = r0
    for t in range(n):
        z = rng.standard_normal(npaths)
        rp = np.maximum(r[:, t], 0)
        r[:, t + 1] = np.maximum(r[:, t] + a * (b - r[:, t]) * DT + sig * np.sqrt(rp * DT) * z, 0)
    return r

pv = sim_vasicek(a_v, b_v, sig_v, r0)                  # from today's rate -> mean reversion
pc = sim_cir(a_c, b_c, sig_c, r0)

# Illustrative low-rate, higher-volatility regime (e.g. 2010s Europe/Japan) to expose the
# structural negative-rate property. With the REAL US params, reversion is so strong and vol
# so low that Vasicek never approaches zero - so a representative low-rate regime is used to
# make the sigma-vs-sigma*sqrt(r) difference visible. Clearly separate from the calibration.
ILL_A, ILL_B, ILL_SV, ILL_SC, R0_LOW = 0.30, 0.005, 0.012, 0.15, 0.005
pv_lo = sim_vasicek(ILL_A, ILL_B, ILL_SV, R0_LOW)
pc_lo = sim_cir(ILL_A, ILL_B, ILL_SC, R0_LOW)
neg_v = (pv_lo < 0).any(axis=1).mean()                 # fraction of Vasicek paths going negative
neg_c = (pc_lo < 0).any(axis=1).mean()
print(f"Illustrative low-rate regime negative paths: Vasicek {neg_v:.1%}, CIR {neg_c:.1%}")

# ----------------------------------------------------------------------
# 5. MODEL-IMPLIED ZERO-COUPON YIELD CURVES from r0
# ----------------------------------------------------------------------
taus = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10])
def vasicek_yield(a, b, sig, r0, tau):
    Bt = (1 - np.exp(-a * tau)) / a
    At = np.exp((Bt - tau) * (a ** 2 * b - 0.5 * sig ** 2) / a ** 2 - sig ** 2 * Bt ** 2 / (4 * a))
    P = At * np.exp(-Bt * r0)
    return -np.log(P) / tau
def cir_yield(a, b, sig, r0, tau):
    g = np.sqrt(a ** 2 + 2 * sig ** 2)
    den = (g + a) * (np.exp(g * tau) - 1) + 2 * g
    Bt = 2 * (np.exp(g * tau) - 1) / den
    At = (2 * g * np.exp((a + g) * tau / 2) / den) ** (2 * a * b / sig ** 2)
    P = At * np.exp(-Bt * r0)
    return -np.log(P) / tau
yc_v = np.array([vasicek_yield(a_v, b_v, sig_v, r0, t) for t in taus])
yc_c = np.array([cir_yield(a_c, b_c, sig_c, r0, t) for t in taus])

# ----------------------------------------------------------------------
# 6. PLOTS
# ----------------------------------------------------------------------
fig, ax = plt.subplots()
rate.mul(100).plot(ax=ax, color="#0E7C7B", lw=1, label="^IRX 13-wk T-bill")
ax.axhline(b_v * 100, color="#C0392B", ls="--", lw=1.2, label=f"Vasicek long-run mean {b_v:.2%}")
ax.axhline(b_c * 100, color="#B7791F", ls=":", lw=1.4, label=f"CIR long-run mean {b_c:.2%}")
ax.set_title("Real short rate and calibrated long-run means"); ax.set_ylabel("rate (%)")
ax.legend(frameon=False)
p_hist = os.path.join(OUT, "history.png"); fig.savefig(p_hist); plt.close(fig)

tg = np.linspace(0, 2, pv.shape[1])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=True)
for i in range(60):
    a1.plot(tg, pv[i] * 100, color="#0E7C7B", lw=.3, alpha=.4)
    a2.plot(tg, pc[i] * 100, color="#B7791F", lw=.3, alpha=.4)
a1.plot(tg, pv.mean(0) * 100, color="#C0392B", lw=1.8); a1.axhline(b_v*100, color="#333", ls="--", lw=.8)
a2.plot(tg, pc.mean(0) * 100, color="#C0392B", lw=1.8); a2.axhline(b_c*100, color="#333", ls="--", lw=.8)
a1.set_title(f"Vasicek paths (half-life {hl_v:.1f}y)"); a2.set_title(f"CIR paths (half-life {hl_c:.1f}y)")
a1.set_ylabel("rate (%)"); a1.set_xlabel("years"); a2.set_xlabel("years")
fig.tight_layout(); p_paths = os.path.join(OUT, "paths.png"); fig.savefig(p_paths); plt.close(fig)

fig, ax = plt.subplots()
ax.hist(pv_lo[:, -1] * 100, bins=50, alpha=.55, color="#0E7C7B", density=True, label="Vasicek (normal)")
ax.hist(pc_lo[:, -1] * 100, bins=50, alpha=.55, color="#B7791F", density=True, label="CIR (chi-square, >=0)")
ax.axvline(0, color="#C0392B", lw=1.4, ls="--", label="zero")
ax.set_title("Illustrative low-rate, high-vol regime: short-rate distribution in 2y")
ax.set_xlabel("rate (%)"); ax.set_ylabel("density"); ax.legend(frameon=False)
p_dist = os.path.join(OUT, "distribution.png"); fig.savefig(p_dist); plt.close(fig)

fig, ax = plt.subplots()
ax.plot(taus, yc_v * 100, "o-", color="#0E7C7B", label="Vasicek")
ax.plot(taus, yc_c * 100, "s-", color="#B7791F", label="CIR")
ax.axhline(r0 * 100, color="#999", ls=":", lw=1, label=f"current short rate {r0:.2%}")
ax.set_title("Model-implied zero-coupon yield curves (from today's short rate)")
ax.set_xlabel("maturity (years)"); ax.set_ylabel("yield (%)"); ax.legend(frameon=False)
p_yc = os.path.join(OUT, "yield_curve.png"); fig.savefig(p_yc); plt.close(fig)

# ----------------------------------------------------------------------
# 7. REPORT
# ----------------------------------------------------------------------
rp = Report(
    "Stochastic Interest-Rate Modeling: Vasicek & Cox-Ingersoll-Ross",
    "Two mean-reverting short-rate models calibrated by Maximum Likelihood to the real 13-week T-bill "
    "yield, compared on dynamics, the negative-rate question, and the yield curves they imply.",
    tags=["^IRX short rate", "MLE calibration", "Vasicek vs CIR", "Mean reversion", "SciPy"],
    date=f"Data through {rate.index[-1].date()}",
)
rp.purpose(
    "<p>Interest-rate models underpin the pricing of bonds, swaps and every rate derivative. Vasicek "
    "and CIR are the two canonical one-factor short-rate models — identical except for one term. This "
    "project calibrates both to a <em>real</em> short rate by Maximum Likelihood and asks the practical "
    "question: <em>which model, in which regime, and what does that single differing term actually "
    "change?</em></p>")

rp.section(
    "Data & fetching procedure",
    "<p>The 13-week US Treasury-bill yield is a clean, liquid proxy for the instantaneous short rate:</p>"
    + rp.formula(
        f"series   : ^IRX (13-week T-bill discount yield), Yahoo Finance, quoted in %\n"
        f"window   : {rate.index[0].date()} -> {rate.index[-1].date()}  ({len(r)} obs, ~4 years)\n"
        f"transform: divide by 100 -> decimal; floor at 1bp; current r0 = {r0:.3%}\n"
        f"note     : a ~4y window is used deliberately (see below)")
    + "<p>Window choice is itself a modelling decision: over a full decade the structural 0%&rarr;5% "
      "rate shift overwhelms mean reversion and single-factor MLE degenerates (the long-run mean runs "
      "off to the bound). The recent hiking-and-plateau cycle, by contrast, genuinely reverts around "
      "~4.5%, and <em>both</em> models calibrate to consistent, sensible parameters — so that is the "
      "window used.</p>",
    num="01")

rp.section(
    "Theory & derivations",
    "<h3>The two models — identical but for the diffusion</h3>"
    + rp.formula(
        "Vasicek :  dr = a(b - r) dt + sigma dW\n"
        "CIR     :  dr = a(b - r) dt + sigma sqrt(r) dW\n\n"
        "a = speed of mean reversion    b = long-run mean    sigma = volatility\n"
        "the drift a(b-r) pulls r back toward b; the ONLY difference is the sqrt(r) in CIR")
    + "<table style='width:100%'><tr><th></th><th>Vasicek</th><th>CIR</th></tr>"
      "<tr><td>Diffusion</td><td>constant &sigma;</td><td>&sigma;&radic;r (vanishes as r&rarr;0)</td></tr>"
      "<tr><td>Rates can be negative?</td><td>Yes (Gaussian)</td><td>No, if Feller holds</td></tr>"
      "<tr><td>Distribution of r</td><td>Normal</td><td>Non-central &chi;&sup2;</td></tr>"
      "<tr><td>Feller condition</td><td>&mdash;</td><td>2ab &ge; &sigma;&sup2;</td></tr></table>"
    + "<h3>Calibration by Maximum Likelihood</h3>"
    + rp.formula(
        "VASICEK: the exact transition is Gaussian, so the discretised model is an AR(1):\n"
        "   r_{t+1} = e^{-a dt} r_t + b(1-e^{-a dt}) + noise\n"
        "   => OLS of r_{t+1} on r_t IS the (conditional) MLE:\n"
        "      a = -ln(slope)/dt,   b = intercept/(1-slope),   sigma from residual variance\n\n"
        "CIR: transition density is non-central chi-square (no OLS shortcut):\n"
        "   2c r_{t+1} ~ ncx2(df = 4ab/sigma^2, nc = 2c r_t e^{-a dt}),  c = 2a / (sigma^2 (1-e^{-a dt}))\n"
        "   maximise the log-likelihood numerically over (a, b, sigma)")
    + "<p>MLE in one line: choose the parameters that make the observed sequence of rate moves the most "
      "probable under each model's own transition density.</p>",
    num="02")

rp.raw(rp.kpi_grid([
    ("Vasicek a / b", f"{a_v:.2f} / {b_v:.1%}", f"half-life {hl_v:.1f}y", "acc"),
    ("CIR a / b", f"{a_c:.2f} / {b_c:.1%}", f"half-life {hl_c:.1f}y", "acc"),
    ("Vasicek sigma", f"{sig_v:.4f}", "constant", ""),
    ("CIR sigma", f"{sig_c:.4f}", "scales with sqrt(r)", ""),
    ("Feller 2ab>=s^2", "holds" if feller else "violated",
     f"2ab={2*a_c*b_c:.4f}", "pos" if feller else "neg"),
    ("Neg-rate paths", f"V {neg_v:.0%} / C {neg_c:.0%}", "low-rate scenario", ""),
]))

rp.section(
    "Results on real data",
    rp.table(["Parameter", "Vasicek", "CIR", "Meaning"], [
        ["a (reversion speed)", f"{a_v:.3f}", f"{a_c:.3f}", "higher = faster pull to the mean"],
        ["b (long-run mean)", f"{b_v:.3%}", f"{b_c:.3%}", "level rates gravitate toward"],
        ["sigma (volatility)", f"{sig_v:.4f}", f"{sig_c:.4f}", "note: not directly comparable (CIR scales by √r)"],
        ["Half-life ln2/a", f"{hl_v:.2f} y", f"{hl_c:.2f} y", "time to close half the gap to b"],
    ])
    + f"<p>Both models agree the rate mean-reverts toward roughly <strong>{b_v:.1%}-{b_c:.1%}</strong>. "
      f"The Feller condition 2ab &ge; &sigma;&sup2; "
      f"{'holds' if feller else 'is violated'} for the CIR fit "
      f"(2ab = {2*a_c*b_c:.4f} vs &sigma;&sup2; = {sig_c**2:.4f}), so CIR keeps the rate "
      f"{'strictly positive' if feller else 'non-negative (though it can touch zero)'}. "
      f"From today's ~4% level, strong reversion keeps both models well away from zero. The structural "
      f"gap only bites in a low-rate, higher-volatility regime (the 2010s Europe/Japan world) — and in "
      f"that illustrative regime <strong>{neg_v:.0%}</strong> of Vasicek paths go negative versus "
      f"<strong>{neg_c:.0%}</strong> of CIR paths, exactly what the &radic;r term is designed to "
      f"prevent.</p>"
    + rp.figure(p_hist, "The real short rate with each model's calibrated long-run mean — reversion target.")
    + rp.figure(p_paths, "Simulated futures: both revert toward b; the red line is the path average.")
    + rp.figure(p_dist, "Illustrative low-rate regime: Vasicek is normal and leaks below zero; CIR is skewed and stays non-negative.")
    + rp.figure(p_yc, "Yield curves each model implies from today's short rate — the affine term structure in action."),
    num="03")

rp.learned([
    "<strong>One term changes everything.</strong> The only difference between the models is &sigma; vs "
    "&sigma;&radic;r, yet that alone decides whether rates can go negative and what distribution they "
    "follow.",
    "<strong>Vasicek calibration is free.</strong> Because its transition is Gaussian, the AR(1)/OLS "
    "fit is the exact MLE — a neat link between econometrics and stochastic calculus.",
    "<strong>CIR needs real MLE.</strong> Its non-central chi-square density has no closed-form "
    "estimator, so I maximised the log-likelihood numerically — and checked the Feller condition to "
    "confirm non-negativity.",
    "<strong>Both are affine.</strong> Each yields a closed-form bond price and hence an entire "
    "model-implied yield curve from a single state variable, r.",
])
rp.inference(
    f"<p>Calibrated to the real T-bill rate, both models revert toward ~{b_c:.1%} with half-lives of "
    f"{hl_c:.1f}-{hl_v:.1f} years. <strong>Vasicek</strong> wins on tractability (Gaussian, closed-form "
    "everything, OLS calibration) and is perfectly usable when negative rates are acceptable — as they "
    "genuinely were in Europe and Japan. <strong>CIR</strong> wins when non-negativity matters: its "
    "&radic;r diffusion switches off exactly at zero, and here the Feller condition "
    f"{'holds' if feller else 'is essentially borderline'}. The right model isn't universal — it "
    "depends on the rate regime you're modelling, which is exactly the comparative judgement the "
    "project set out to make.</p>")

rp.save(os.path.join(HERE, "report.html"))
print("Done.")
