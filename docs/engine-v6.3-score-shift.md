# Engine v6.3: wiring the checks the methodology describes, score-shift evidence

Date: 2026-10-17. Compared: `main` (v6.2, after the seed/crash fix in 03aca02) vs v6.3.
Harness: every sample run through `QuantProofValidator(...).run()` with no benchmark and
n_trials=10, a synthetic random benchmark and n_trials=10, and no benchmark and n_trials=1.
"Uncapped" is the weighted category score before the engine's score caps.

## What changed in run()
- Scored checks: 29 → 37. Overfitting +3 (autocorrelation, information ratio, beta),
  Execution +4 (0.3% slippage, commission drag, partial fills, live gap), Compliance +1 (5-gate quick check).
- Plausibility: 4 checks now actually run. They feed `audit_flags`, `plausibility_summary`
  and the existing "Manual Audit Required" cap (35). They are not weighted, and the API
  `checks` list still excludes them, as before.
- Information ratio and beta return `scored=False` (informational, excluded from the weighted
  score) when no benchmark aligned 1:1 with the trades is available. That covers no dates,
  non-US/IN assets, a market-data fetch failure, length mismatch, a constant benchmark or NaNs.
- Partial fills were a constant (always 94% fill, always score 70). They now simulate an 80% fill
  rate on the strategy's own returns (500 seeded runs).
- Bugs found while wiring:
  - `benchmark_returns` was accepted but never stored.
  - Frequency-return plausibility compared a fraction to a percent threshold, so it could never fire.
  - The quick gate's name matched the per-firm soft-cap filter, so a failed gate would have capped scores at 75.
  - The review count ignored "Review Recommended".
- API: additive only. `CheckResultResponse.scored` (default true). No field removed or renamed
  (OpenAPI diff vs main: one added property).

## Score shifts (final score, then uncapped weighted score)
```
sample | benchmark | n_trials                      final      uncapped       Overfit          Exec         Compl
A_repo_sample.csv|bm=none|n=10                  30→30      84.8→81.9    72.7→64.6    88.1→78.1   100.0→100.0 
A_repo_sample.csv|bm=synthetic|n=10             30→30      84.8→83.3    72.7→71.1    88.1→78.1   100.0→100.0 
A_repo_sample.csv|bm=none|n=1                   55→55      86.7→83.6    81.5→72.4    88.1→78.1   100.0→100.0 
B_good_daily_2019_2021.csv|bm=none|n=10         55→55      67.6→70.5    81.6→83.6    51.4→50.6    50.0→66.7  
B_good_daily_2019_2021.csv|bm=synthetic|n=10    55→55      67.6→71.2    81.6→86.6    51.4→50.6    50.0→66.7  
B_good_daily_2019_2021.csv|bm=none|n=1          55→55      69.0→71.7    87.8→89.2    51.4→50.6    50.0→66.7  
C_thin_edge.csv|bm=none|n=10                    30→30      46.0→44.1    47.8→45.8    50.0→44.5    16.6→11.1  
C_thin_edge.csv|bm=synthetic|n=10               30→30      46.0→45.6    47.8→52.5    50.0→44.5    16.6→11.1  
C_thin_edge.csv|bm=none|n=1                     30→30      46.0→44.1    47.8→45.8    50.0→44.5    16.6→11.1  
D_losing.csv|bm=none|n=10                     18.6→20      18.6→21.4    23.1→31.7     0.0→8.5      0.0→0.0   
D_losing.csv|bm=synthetic|n=10                18.6→20      18.6→22.2    23.1→35.0     0.0→8.5      0.0→0.0   
D_losing.csv|bm=none|n=1                      19.6→20      19.6→22.3    27.5→35.6     0.0→8.5      0.0→0.0   
E_no_dates.csv|bm=none|n=10                   18.3→20      18.3→22.5    12.7→22.4     0.0→18.4     0.0→0.0   
E_no_dates.csv|bm=synthetic|n=10              18.3→20      18.3→23.6    12.7→27.4     0.0→18.4     0.0→0.0   
E_no_dates.csv|bm=none|n=1                    19.1→20      19.1→23.2    16.4→25.7     0.0→18.4     0.0→0.0   
F_implausibly_smooth.csv|bm=none|n=10           30→30      90.9→88.9    87.5→88.9    87.2→66.2   100.0→100.0 
F_implausibly_smooth.csv|bm=synthetic|n=10      30→30      90.9→89.3    87.5→90.9    87.2→66.2   100.0→100.0 
F_implausibly_smooth.csv|bm=none|n=1            30→30      90.9→88.9    87.5→88.9    87.2→66.2   100.0→100.0 
G_intraday_hf.csv|bm=none|n=10                  30→30      75.3→76.0    69.1→72.6    50.0→50.0   100.0→100.0 
G_intraday_hf.csv|bm=synthetic|n=10             30→30      75.3→77.1    69.1→77.5    50.0→50.0   100.0→100.0 
G_intraday_hf.csv|bm=none|n=1                   30→30      75.3→76.0    69.1→72.6    50.0→50.0   100.0→100.0 
H_autocorrelated.csv|bm=none|n=10               55→55      70.8→70.2    88.2→78.4    72.9→63.7    50.0→66.7  
H_autocorrelated.csv|bm=synthetic|n=10          55→55      70.8→71.0    88.2→82.3    72.9→63.7    50.0→66.7  
H_autocorrelated.csv|bm=none|n=1                55→55      71.4→70.7    90.7→80.6    72.9→63.7    50.0→66.7  
```
Samples:
- **A**: the repo's `backend/test_sample.csv` (30 dated trades).
- **B**: good daily, 2019-21.
- **C**: thin edge.
- **D**: losing.
- **E**: no dates.
- **F**: implausibly smooth.
- **G**: 15-minute intraday, 3,000 trades.
- **H**: AR(1)=0.5.

## Are the shifts correct?
- **Final scores** change in 2 of 8 samples (D and E), by at most 1.7 points. Both land on the existing Sharpe < 0 cap (20). In the other samples an existing cap already decided the result (DSR → 30, single regime → 55, plausibility), so the new checks move only the uncapped score. That is the intended design: hard failures dominate.
- **H (autocorrelated)**: Overfitting drops 88 → 78 because the Ljung-Box check fails (AR(1)=0.46, p<0.001). This is correct. Serial correlation inflates the DSR, CI and bootstrap results, which is exactly what the check exists to catch.
- **A (repo sample)**: Overfitting drops 73 → 65. The sample has AR(1) = −0.58, a real and significant result. Execution drops 88 → 78 because 0.3% slippage leaves only 28% of the edge. Correct.
- **F (smooth)**: the Equity Curve and Sharpe plausibility flags now fire and add audit flags, and the final score is held at the 35/30 caps. Execution drops 87 → 66 under realistic costs. Correct. Before, this sample returned HTTP 500 (see 03aca02).
- **D/E (losing, no dates)**: the uncapped score rises about 3–5 points, because the strategy does pass autocorrelation (no serial correlation) and commission drag. The Sharpe < 0 cap keeps the final score at 20. That's acceptable: each check measures one property, and the cap stops a losing strategy from looking fundable.
- **B/H (Compliance 50 → 66.7)**: the 5-gate check passes at 4/5 gates, which is its documented behaviour in standard mode.
