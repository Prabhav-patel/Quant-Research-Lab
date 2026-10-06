# Quant research lab — Prabhav Jatin Patel

Seven reproducible Python studies and a separate [market-microstructure system](https://github.com/Prabhav-patel/HFT) document the projects on my résumé. Start with the **[single illustrated report](report.html)** for the question, method, result and limitation of each study. The code and generated figures remain in their numbered folders; no raw market-data dumps or personal files are published.

The résumé dates refer to when I first completed the underlying studies (including earlier Excel/notebook work). These Python implementations and the public presentation were refreshed in **October 2026**; the publication date is not the original study date. Earlier public notebooks remain available in [Black-Scholes](https://github.com/Prabhav-patel/Black-Scholes-Model), [binomial trees](https://github.com/Prabhav-patel/Binomial-Tree-Model), [Monte Carlo](https://github.com/Prabhav-patel/Monte-Carlo-Simulation), and [portfolio risk](https://github.com/Prabhav-patel/Portfolio-Optimization-Risk-Analysis).

| Original study | Research question | Reproducible implementation | Current finding |
|---|---|---|---|
| Ongoing | Can a sequenced order book feed features into paper trading? | [HFT source](https://github.com/Prabhav-patel/HFT) | Book tests pass; trading edge and full-stack latency not established. |
| Jan 2026 | How sensitive are NIFTY option prices to volatility assumptions? | [02 · volatility](02_StochVol_MonteCarlo_NIFTY/main.py) | GARCH and fixed-volatility scenarios differ materially; this is not an arbitrage claim. |
| May 2025 | Does a two-bank portfolio diversify risk? | [03 · portfolio](03_Portfolio_Risk_JPM_MS/main.py) | JPM–MS return correlation is 0.74 in the refreshed window. |
| Mar 2025 | Does a rolling VaR model calibrate out of sample? | [04 · tail risk](04_VaR_ExpectedShortfall/main.py) | Kupiec p = 0.008: reject the 99% model over the full test window. |
| Jan 2025 | How do Vasicek and CIR behave in low-rate scenarios? | [05 · rates](05_Vasicek_CIR_RateModels/main.py) | CIR paths stay nonnegative in the illustrative experiment; ^IRX is only a rate proxy. |
| Oct 2024 | Can path-dependent options be priced and cross-checked? | [06 · Monte Carlo](06_MonteCarlo_European_Asian_Barrier/main.py) | European/Asian benchmarks and barrier in–out parity agree; a control variate cuts standard error. |
| Sep 2024 | Does ARIMA beat a naive NIFTY forecast? | [07 · time series](07_ARIMA_TimeSeries/main.py) | No: held-out RMSE 966.2 vs 960.5 for the random walk. |
| May 2024 | Do analytical, lattice and simulation methods agree? | [08 · option pricing](08_OptionPricing_BS_Binomial_MC/main.py) | European-call estimates agree within Monte Carlo uncertainty. |

## Reproduce

Install Python 3.11+ and `pip install -r requirements.txt`. From a numbered folder, run `python main.py`. Each script fetches adjusted public prices, writes local `data/` CSV files, plots under `outputs/`, and a detailed local `report.html`. The scripts were last exercised on **6 October 2026** with the package versions in `requirements.txt`. Quotes and model results will change on a later run; a failed or empty download must not be treated as evidence.

The root `report.html` is the **only report HTML committed**. Its figures are frozen outputs from the stated refresh and are not a live dashboard. Data downloads use yfinance/Yahoo for personal research; raw quote files are intentionally not redistributed. The HFT repository likewise excludes raw tick captures. No strategy here is presented as live-trading performance or investment advice.

## What I would test next

For research rather than another pricing demo: (1) walk-forward VaR calibration with volatility-aware alternatives and exception-dependence tests; (2) a microstructure signal study with non-overlapping horizons, spread/fee/slippage assumptions and held-out days; (3) scenario and parameter-sensitivity analysis for option pricing under risk-neutral calibration. Negative results are worth reporting. This is a research agenda, not a claim that these experiments are complete.
