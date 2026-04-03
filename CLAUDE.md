# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python-based algorithmic trading bot for **Upbit** (Korean exchange, KRW markets). Features a backtesting engine with anti-lookahead enforcement (inspired by Jesse), strategy framework (inspired by Freqtrade's IStrategy), and risk management system. Spot trading only.

## Commands

```bash
# Install
pip install -e ".[dev]"

# Docker
docker build -t trading-bot .
docker-compose up -d        # Start paper trading
docker-compose logs -f       # Tail logs
docker-compose down          # Stop

# Dashboard
pip install -e ".[dashboard]"
tradingbot dashboard          # http://localhost:8501

# Run tests
pytest tests/ -v

# Lint & format
ruff check src/ tests/
ruff format src/ tests/

# Type check
mypy src/

# CLI commands
tradingbot download --symbol BTC/KRW --timeframe 1h --since 2024-01-01
tradingbot backtest --strategy sma_cross --symbol BTC/KRW
tradingbot optimize --strategy sma_cross --symbol BTC/KRW --top 10
tradingbot optimize --strategy sma_cross --param-grid '{"fast_period":[10,20],"slow_period":[40,50]}'
tradingbot walk-forward --strategy sma_cross --symbol BTC/KRW --train-months 3 --test-months 1
tradingbot paper --strategy sma_cross --symbol BTC/KRW --balance 1000000
tradingbot paper --strategy sma_cross --symbol BTC/KRW --websocket  # Real-time prices via WebSocket
tradingbot live --strategy sma_cross --symbol BTC/KRW --max-order 500000 --daily-loss-limit 200000
tradingbot status
tradingbot balance
tradingbot data-list
tradingbot symbols

# Multi-symbol backtest (uses all symbols from config/default.yaml)
tradingbot backtest --strategy sma_cross
# Single symbol override
tradingbot backtest --strategy sma_cross --symbol BTC/KRW

# Scan all strategy × symbol × timeframe combinations
tradingbot scan --top 15
tradingbot scan --sort-by total_return --top 20

# Combine filters (no-code strategy building)
tradingbot combine --entry "trend_up:4 + rsi_oversold:30" --exit "rsi_overbought:70" --symbol BTC/KRW
tradingbot combine-scan --top 15  # Scan 36 predefined filter templates
tradingbot combine --entry "trend_up:4 + rsi_oversold:30 + lgbm_prob:0.55" --exit "rsi_overbought:70"  # ML + Rule

# ML strategy (LightGBM)
pip install -e ".[ml]"
tradingbot ml-train --symbol BTC/KRW --timeframe 1h --train-months 3 --test-months 1
tradingbot ml-train-all                    # Train all downloaded symbol × timeframe pairs
tradingbot ml-train-all --timeframe 1h     # Train specific timeframe only
tradingbot ml-backtest --symbol BTC/KRW --timeframe 1h
tradingbot backtest --strategy lgbm --symbol BTC/KRW
```

## Architecture

### Anti-Lookahead Design (Core Principle)
The backtest engine iterates candle-by-candle, only passing confirmed (closed) candles to strategy methods. The strategy can never access the current incomplete candle or future data. This is enforced structurally in `backtest/engine.py` — not by convention.

### Key Flow: Backtest Loop (Multi-Symbol)
```
Pre-compute: strategy.indicators(full_df) per symbol  ← O(N), once
  (strategies with supports_precompute=False use per-iteration fallback)

Build unified timeline from all symbols' timestamps

For each timestamp ts:
  Phase 1 — fills:
    fill_candle = symbol_candles[idx]
    Check stop losses, fill pending orders

  Phase 2 — strategy evaluation:
    visible_df = indicator_df[0..idx-1]  ← strategy sees ONLY past candles
    strategy.should_exit(visible_df)
    strategy.should_entry(visible_df) — blocked if stop loss fired this candle
    Risk manager validates → fill at fill_candle's open + slippage

  Phase 3 — update prices & record equity
```

### Module Responsibilities
- `core/models.py` — Frozen dataclasses: Candle, Signal, Order, Trade, Position, PortfolioState
- `strategy/base.py` — Abstract `Strategy` class with `indicators()`, `should_entry()`, `should_exit()`, `supports_precompute`
- `backtest/engine.py` — Core backtest loop, multi-symbol support, indicator pre-computation (most critical file)
- `backtest/simulator.py` — Order fill simulation with slippage and fees (Upbit: 0.05%)
- `backtest/report.py` — Performance metrics: Sharpe, Sortino, max drawdown, win rate, profit factor
- `risk/manager.py` — Position sizing (fixed-fractional), drawdown circuit breaker, stop loss
- `data/fetcher.py` — CCXT-based OHLCV download with Upbit rate limiting
- `data/storage.py` — Parquet file I/O with auto-merge on save
- `data/indicators.py` — 19 technical indicator wrappers (SMA, EMA, RSI, MACD, BB, ATR, Stochastic, ADX, Ichimoku, Aroon, CCI, ROC, MFI, OBV, Keltner, Donchian, Z-score, PctFromMA, Volume SMA)
- `backtest/optimizer.py` — Grid search parameter optimization with parallel execution, optional `progress` parameter
- `backtest/walk_forward.py` — Walk-forward validation (train/test window rolling), optional `progress` parameter
- `strategy/filters/` — 31 reusable filters with role tagging (entry/trend/volatility/volume/exit)
  - `base.py` — BaseFilter ABC with `role` field + `check_exit(df, entry_index)` for trailing exits
  - `trend.py` — TrendUp/Down, AdxStrong, IchimokuAbove, AroonUp
  - `momentum.py` — RsiOversold, RsiOverbought, MacdCrossUp, StochOversold, CciOversold, RocPositive, MfiOversold
  - `price.py` — PriceBreakout, EmaAbove, BbUpperBreak, EmaCrossUp, DonchianBreak
  - `volatility.py` — AtrBreakout, KeltnerBreak, BbSqueeze, BbBandwidthLow
  - `volume.py` — VolumeSpike, ObvRising, MfiConfirm
  - `exit.py` — StochOverbought, CciOverbought, MfiOverbought, ZscoreExtreme, PctFromMaExit, AtrTrailingExit
  - `ml.py` — LgbmProbFilter: ML probability as entry veto + Half-Kelly strength for position sizing
  - `registry.py` — 31 filters registered, parse_filter_spec/parse_filter_string
- `strategy/combined.py` — CombinedStrategy: AND entry (role-aware skip, ML strength collection) + OR exit with entry_index for trailing stops
- `strategy/lgbm_strategy.py` — LGBMStrategy: LightGBM model inference + Half-Kelly position sizing
- `ml/features.py` — 36 features from 19 indicators (raw + derived + rolling stats + temporal)
- `ml/targets.py` — 4h forward return binary classification target (offline only)
- `ml/trainer.py` — LGBMTrainer: train, evaluate, save/load (.lgb + _meta.json)
- `ml/walk_forward.py` — MLWalkForwardTrainer: purged expanding window + embargo
- `exchange/base.py` — Abstract exchange interface (BaseExchange ABC)
- `exchange/ccxt_exchange.py` — CCXT async implementation for Upbit (retry + rate limiting)
- `exchange/paper.py` — Paper trading exchange (simulated fills, portfolio tracking)
- `live/engine.py` — Async live/paper trading loop (polling, candle detection, signal execution)
- `live/state.py` — JSON-based state persistence (positions, equity history, crash recovery)
- `live/order_manager.py` — Order lifecycle (submit, poll, timeout cancel, market re-order)
- `risk/validators.py` — Pre-trade safety (max order size, daily loss limit, cooldown)
- `notifications/telegram.py` — Telegram Bot API notifications
- `exchange/ws_client.py` — Upbit WebSocket client (real-time ticker, auto-reconnect)
- `dashboard/app.py` — Streamlit web dashboard (Live Monitor + Backtest Viewer)
- `config.py` — Pydantic settings from YAML + .env override
- `utils/logging.py` — Console + JSON file logging with daily rotation (LOG_DIR env)

### Built-in Strategies (7종)
- `sma_cross` — SMA golden/dead cross (fast_period, slow_period)
- `rsi_mean_reversion` — RSI oversold entry / overbought exit (rsi_period, oversold, overbought)
- `macd_momentum` — MACD histogram zero-cross (fast, slow, signal)
- `bollinger_breakout` — Price breaks upper BB / drops below middle BB (period, std)
- `multi_tf` — Higher TF trend filter (SMA) + base TF entry (RSI), anti-lookahead resample, `supports_precompute=False`
- `volume_breakout` — Volume spike (avg × N) + price breakout above recent high
- `lgbm` — LightGBM meta-model: 36 features from indicators → binary classification → Half-Kelly sizing

### Strategy Interface
Strategies inherit from `Strategy` and implement three methods:
- `indicators(df)` — Append indicator columns to DataFrame
- `should_entry(df, symbol)` → `Signal | None`
- `should_exit(df, symbol, position)` → `Signal | None`

### Data Format
Candle data stored as Parquet: `data/{SYMBOL}_{QUOTE}/{timeframe}.parquet`
(e.g., `data/BTC_KRW/1h.parquet`)

## Key Dependencies
- `ccxt` for exchange abstraction (Upbit supported)
- `ta` for technical indicators (not pandas-ta — incompatible with Python 3.11+)
- `pydantic` + `pydantic-settings` for config validation
- `typer` + `rich` for CLI
- `pyarrow` for Parquet I/O
- `websockets` for Upbit real-time data (auto-reconnect, interruptible cooldown)
- `streamlit` + `plotly` for web dashboard (optional: `pip install -e ".[dashboard]"`)
- `lightgbm` + `scikit-learn` for ML strategy (optional: `pip install -e ".[ml]"`)
