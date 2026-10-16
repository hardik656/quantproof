# QuantProof — Institutional Trading Strategy Validator

**Live at [quantproof.net](https://quantproof.net)** · Built solo · Paying users · Launched Product Hunt March 2026

> 44 institutional checks. 5 crash simulations. One Fundable Score.  
> Know if your edge is real — before you pay for a prop firm challenge.

---

## What It Does

QuantProof takes a trader's backtest or live trade history and runs it through the same statistical checks that hedge funds and prop firms use internally — things retail traders never see until they fail a challenge and lose ₹20,000.

Upload a CSV or connect a live broker. Get a 0–100 Fundable Score in under 30 seconds, with every failure ranked, explained, and tied to a specific fix.

**The problem it solves:** 92% of prop firm challenges fail. Most from drawdown violations and overfitted backtests. QuantProof catches both before you pay.

---

## Architecture

```
Frontend          Vanilla JS + HTML/CSS — single page, no framework dependency
Backend           Python 3 / FastAPI — async, Pydantic v2, Uvicorn
Validation        Custom QuantProofValidator engine — 44 checks, 5 crash sims
AI Fix Plans      Groq (Llama 3.3-70B) → Anthropic Claude → OpenAI fallback chain
Payments          Razorpay (India) · Lemon Squeezy (International)
Deployment        Render — backend + frontend separate services
Cache             OrderedDict LRU (50 entries) — validation results for PDF download
```

---

## The 44 Checks — What Gets Tested

**Overfitting Detection (8 checks)**
- Deflated Sharpe Ratio (Bailey & Lopez de Prado 2012) — adjusts for multiple testing bias
- Combinatorially Purged Cross-Validation with embargo gaps — the fund standard for walk-forward testing
- Monte Carlo robustness — n=1 to n=50 bootstrap stability
- Ljung-Box autocorrelation test — detects serial dependence in returns
- Hurst exponent — distinguishes genuine edge from random walk
- Bootstrap Sharpe stability — how much does Sharpe vary across resamples?
- HMM regime detection — does the strategy behave consistently across market regimes?
- Backtest overfitting probability

**Risk Management (8 checks)**
- Max drawdown vs prop firm thresholds (FTMO 10%, Topstep 6%, etc.)
- Daily loss limit compliance
- CVaR / Expected Shortfall at 95% and 99%
- Fractional Kelly criterion + ruin probability
- Sortino ratio
- Calmar ratio
- Profit factor
- Recovery factor

**Regime Robustness (8 checks)**
- Performance split across BULL / BEAR / CONSOLIDATION regimes
- Regime auto-detection via SPY 20-day SMA (yfinance) with 3-day confirmation
- Nifty 50 benchmark for Indian strategies (auto-detected by currency=INR or .NS/.BO tickers)
- Strategy vs benchmark: Information Ratio
- Strategy vs benchmark: Beta Exposure
- Regime consistency score
- Drawdown regime analysis
- Capacity analysis (Almgren-Chriss market impact model)

**Execution Reality (8 checks)**
- Slippage sensitivity
- Market impact estimation
- Trade frequency sustainability
- Hold time analysis
- Commission drag
- Spread cost modelling
- Position sizing consistency
- Scale warning detection

**Prop Firm Compliance (8 checks)**
- FTMO rules — 10% max DD, 5% daily loss, profit target
- Topstep rules — 6% max DD, trailing drawdown
- The5ers rules
- Blue Guardian rules
- Earn2Trade rules
- CTA / institutional fund standards
- SEBI algo framework checks (Indian strategies) — OPS threshold, session logout
- Plausibility checks — flags suspicious return profiles

**5 Crash Simulations**
- 2008 Global Financial Crisis
- 2020 COVID crash
- 2022 rate hike bear market
- 2010 Flash Crash
- 1998 LTCM / Russia default

---

## Broker Integrations

| Source | Method | Details |
|--------|--------|---------|
| Zerodha Kite TradeBook | CSV upload | FIFO lot-matching, orphan sell detection, segment-aware (EQ/FO) |
| Zerodha Kite Orders | CSV upload | COMPLETE orders only, FIFO pairing |
| Alpaca REST API | Live connection | OAuth read-only, keys discarded post-fetch |
| Alpaca UI Export | CSV upload | Tab-separated, human-readable dates |
| IBKR Flex API | Live connection | XML parsing, fifoPnlRealized, openCloseIndicator |
| TradingView | Webhook session | 7-day session, 2000 alerts, real-time BUY/SELL pairing |
| MT5 | CSV upload | Deals report + Positions report, forex scale warning |
| Options CSV | CSV upload | Single-leg only, premium normalisation |
| Generic PnL CSV | CSV upload | Auto-detects pnl/profit/return columns |

---

## Payment System

```
Tier         Price    Token Prefix    What It Unlocks
─────────────────────────────────────────────────────────────────────
Free         ₹0       —               3 validations/day, all 44 checks visible
Lite         ₹99      LITE-           AI top-3 fix plan in browser, full failed check details
Full PDF     ₹299     RPT-            PDF report, n=20 forced, full ranked AI plan
Portfolio    ₹199     PORT-           2-3 strategy correlation matrix, diversification score
Pro/30d      ₹999     PRO-            Unlimited, n=50, all PDFs, 30-day expiry
```

Payment flow: Razorpay order → HMAC-verified webhook → token minted → added to `REPORT_ACCESS_TOKENS` set → persisted to `/tmp`. Pro tokens stored separately in `_PRO_TOKENS` dict with Unix expiry timestamp.

---

## AI Fix Plan

When a strategy fails checks, QuantProof generates a personalised fix plan referencing the trader's actual numbers — not generic advice.

```python
# Provider fallback chain
if GROQ_API_KEY:
    # llama-3.3-70b-versatile — fastest, cheapest
elif ANTHROPIC_API_KEY:
    # claude-sonnet — fallback
elif OPENAI_API_KEY:
    # gpt-4o-mini — final fallback
```

Output is structured JSON: ranked fixes with `why_it_matters`, `specific_fix`, `score_impact`, and `effort` level. Deterministic rule-based fallback if all LLMs unavailable.

---

## Validation Cache

```python
VALIDATION_RESULT_CACHE = collections.OrderedDict()  # LRU, 50 entries max
```

After every validation, results are cached by `validation_hash`. PDF generation and AI fix endpoints pull from cache — no re-running the validator on payment. Cache evicts LRU when full.

---

## Key Technical Decisions

**Why no database?** Validation results are ephemeral — traders run a validation, download a PDF, move on. An in-memory LRU cache covers the window between validation and PDF download without the complexity and cost of a DB. Token persistence uses `/tmp` JSON files — sufficient for a single-instance Render deployment.

**Why vanilla JS frontend?** Zero build step, zero dependency drift, instant load. The validation UI is stateful but not complex enough to need a framework. This also means anyone reading the code can follow the logic without knowing React internals.

**Why FastAPI?** Async-native, Pydantic validation out of the box, automatic OpenAPI docs, fast enough for the compute profile. The PDF semaphore (`asyncio.Semaphore(2)`) prevents concurrent PDF generation from blocking the event loop.

**Why multiple AI providers?** Groq's Llama is free and fast but rate-limited. Claude and OpenAI are fallbacks. The structured JSON prompt works across all three with identical output shape — provider is transparent to the caller.

---

## Running Locally

```bash
git clone https://github.com/hardik656/quantproof
cd quantproof

pip install -r requirements.txt

# Set environment variables
export RAZORPAY_KEY_ID=your_key
export RAZORPAY_KEY_SECRET=your_secret
export GROQ_API_KEY=your_key
export TEST_REPORT_TOKEN=dev-test-token

uvicorn main:app --reload --port 8000
```

Frontend: open `index.html` in browser or serve with any static server.

---

## Environment Variables

```
RAZORPAY_KEY_ID          Razorpay payment key
RAZORPAY_KEY_SECRET      Razorpay webhook verification
GROQ_API_KEY             Primary LLM (Llama 3.3-70B)
ANTHROPIC_API_KEY        Fallback LLM (Claude)
OPENAI_API_KEY           Fallback LLM (GPT-4o-mini)
LEMONSQUEEZY_API_KEY     International payments
LEMONSQUEEZY_STORE_ID    LS store identifier
LEMONSQUEEZY_VARIANT_ID  LS product variant
LEMONSQUEEZY_WEBHOOK_SECRET  LS webhook verification
TEST_REPORT_TOKEN        Dev override — grants pro access
```

---

## Project Structure

```
quantproof/
├── main.py              FastAPI app — all endpoints, parsers, payment webhooks
├── validator.py         QuantProofValidator — 44 checks, crash sims, scoring
├── generate_report.py   PDF generation from validation results
├── prop_firm_compliance.py  FTMO/Topstep/The5ers/Blue Guardian/Earn2Trade rules
├── index.html           Frontend — single page application
├── methodology.html     Public methodology documentation
├── requirements.txt     Python dependencies
```

---

## API Endpoints

```
POST /validate                    CSV upload validation
POST /validate/kite               Zerodha Kite CSV validation
POST /validate/alpaca             Live Alpaca broker validation
POST /validate/portfolio          Multi-strategy portfolio analysis
POST /connect/ibkr                IBKR Flex API validation
POST /webhook/tradingview/{id}    TradingView alert ingestion
POST /webhook/tradingview/{id}/validate  Validate accumulated TV alerts
POST /validate/ai-fixes           AI fix plan (₹99+ token required)
POST /report/generate             PDF generation (₹299+ token required)
POST /report/generate-broker      PDF from cached broker validation
POST /report/create-checkout      Razorpay order creation
POST /report/webhook              Razorpay payment verification
GET  /health                      Health check
GET  /ping                        Keep-alive (cron-job.org)
GET  /debug/token                 Token tier inspection (dev)
```

---

## Built By

Hardik Gumber — Delhi, India  
Algo trader since 2021. MSc Information System Management, University of Huddersfield.  
[quantproof.net](https://quantproof.net) · [LinkedIn](https://linkedin.com/in/hardik-gumber-93507b98/) · hgumber43@gmail.com

---

*Not financial advice. QuantProof validates statistical properties of trade data — it does not predict future performance.*
