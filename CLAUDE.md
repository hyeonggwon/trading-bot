# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python-based algorithmic trading bot for **Upbit** (Korean exchange, KRW markets). Features a backtesting engine with anti-lookahead enforcement (inspired by Jesse), strategy framework (inspired by Freqtrade's IStrategy), and risk management system. Spot trading only.

## Commands

```bash
# Install (always use venv)
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint & format
ruff check src/ tests/
ruff format src/ tests/

# Type check
mypy src/

# Docker
docker build -t trading-bot .
docker-compose up -d

# Dashboard (optional)
pip install -e ".[dashboard]"
tradingbot dashboard

# ML (optional)
pip install -e ".[ml]"
```

### CLI Reference

```bash
# Data
tradingbot download --symbol BTC/KRW --timeframe 1h --since 2024-01-01
tradingbot data-list
tradingbot symbols

# Backtest (default: holdout-only window — last 20% of data, comparable to ml-backtest)
tradingbot backtest --strategy sma_cross --symbol BTC/KRW
tradingbot backtest --strategy sma_cross                      # Multi-symbol (uses config)
tradingbot backtest --strategy sma_cross --include-train      # Full data range
tradingbot backtest --strategy sma_cross --start 2024-06-01 --end 2024-12-31

# Scan & Optimize (scan/combine-scan also default to per-batch holdout)
tradingbot scan --top 15
tradingbot scan --sort-by total_return --top 20
tradingbot scan --include-train                               # Full data range
tradingbot scan --start 2024-06-01 --end 2024-12-31
tradingbot optimize --strategy sma_cross --symbol BTC/KRW --top 10
tradingbot optimize --strategy sma_cross --param-grid '{"fast_period":[10,20],"slow_period":[40,50]}'
tradingbot walk-forward --strategy sma_cross --symbol BTC/KRW --train-months 3 --test-months 1

# Filter combinations (no-code strategy building)
tradingbot combine --entry "trend_up:4 + rsi_oversold:30" --exit "rsi_overbought:70" --symbol BTC/KRW
tradingbot combine --entry "trend_up:4 + rsi_oversold:30 + lgbm_prob:0.45" --exit "rsi_overbought:70"
tradingbot combine-scan --top 15
tradingbot combine-scan --verify-top 5                        # Re-verify top N with full engine

# External data (for ML features: kimchi premium, funding rate, FNG, USD/KRW)
tradingbot download-external --since 2024-01-01

# ML
tradingbot ml-train --symbol BTC/KRW --timeframe 1h --train-months 3 --test-months 1
tradingbot ml-train-all --workers 4
tradingbot ml-backtest --symbol BTC/KRW --timeframe 1h        # Default: holdout window from meta.json
tradingbot ml-backtest --symbol BTC/KRW --timeframe 1h --include-train
tradingbot ml-walk-forward --symbol BTC/KRW --timeframe 1h    # Time-honest LGBM (fresh model per window)
tradingbot ml-tune --symbol BTC/KRW --timeframe 4h            # Optuna hyperparameter search
tradingbot ml-tune-thresholds --symbol BTC/KRW --timeframe 4h # Per-model entry/exit threshold sweep
tradingbot ml-tune-thresholds-all                             # Threshold sweep across every saved model

# Paper / Live trading
tradingbot paper --strategy sma_cross --symbol BTC/KRW --balance 1000000
tradingbot paper --strategy sma_cross --symbol BTC/KRW --websocket
tradingbot paper --entry "trend_up:4 + rsi_oversold:30" --exit "rsi_overbought:70" --symbol BTC/KRW
tradingbot live --strategy sma_cross --symbol BTC/KRW --max-order 500000 --daily-loss-limit 200000
tradingbot live --strategy ML+ADXTrend --symbol BTC/KRW --websocket
tradingbot status
tradingbot balance
```

## Architecture

### Anti-Lookahead Design (Core Principle)
The backtest engine iterates candle-by-candle, only passing confirmed (closed) candles to strategy methods. The strategy can never access the current incomplete candle or future data. This is enforced structurally in `backtest/engine.py` — not by convention.

### Backtest Loop (Multi-Symbol)
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

### Strategy Interface
Strategies inherit from `Strategy` and implement three methods:
- `indicators(df)` — Append indicator columns to DataFrame
- `should_entry(df, symbol)` → `Signal | None`
- `should_exit(df, symbol, position)` → `Signal | None`

7 built-in strategies registered via `strategy/registry.py`: `sma_cross`, `rsi_mean_reversion`, `macd_momentum`, `bollinger_breakout`, `multi_tf`, `volume_breakout`, `lgbm`

### Module Responsibilities
- `core/models.py` — Dataclasses: Candle, Signal (frozen), Order, Trade, Position, PortfolioState (mutable)
- `core/enums.py` — OrderSide, SignalType, PositionSide, OrderStatus
- `strategy/base.py` — Abstract `Strategy` class with `indicators()`, `should_entry()`, `should_exit()`, `supports_precompute`
- `strategy/registry.py` — Strategy name → class lookup, loads built-in + custom strategies from `strategies/`
- `strategy/combined.py` — CombinedStrategy: AND entry (role-aware skip, ML strength collection) + OR exit with entry_index for trailing stops
- `strategy/lgbm_strategy.py` — LGBMStrategy: LightGBM model inference + Half-Kelly position sizing (empirical avg_win/avg_loss ratio from training meta). Default thresholds: entry 0.45 / exit 0.30 (calibrated probabilities cluster below 0.50 due to isotonic squashing toward base rate). `set_model()` injects pre-trained models for time-honest walk-forward.
- `strategy/examples/` — 6 built-in strategy files (sma_cross, rsi_mean_reversion, macd_momentum, bollinger_breakout, multi_timeframe, volume_breakout)
- `strategy/filters/` — 31 reusable filters with role tagging (entry/trend/volatility/volume/exit)
  - `base.py` — BaseFilter ABC with `role` field + `check_exit(df, entry_index)` for trailing exits + `vectorized_entry/exit()` for screening
  - `trend.py` — TrendUp/Down, AdxStrong, IchimokuAbove, AroonUp
  - `momentum.py` — RsiOversold, RsiOverbought, MacdCrossUp, StochOversold, CciOversold, RocPositive, MfiOversold
  - `price.py` — PriceBreakout, EmaAbove, BbUpperBreak, EmaCrossUp, DonchianBreak
  - `volatility.py` — AtrBreakout, KeltnerBreak, BbSqueeze, BbBandwidthLow
  - `volume.py` — VolumeSpike, ObvRising, MfiConfirm
  - `exit.py` — StochOverbought, CciOverbought, MfiOverbought, ZscoreExtreme, PctFromMaExit, AtrTrailingExit
  - `ml.py` — LgbmProbFilter: ML probability as entry veto + Half-Kelly strength for position sizing
  - `registry.py` — 31 filters registered, 48 combine templates, parse_filter_spec/parse_filter_string
- `backtest/engine.py` — Core backtest loop, multi-symbol support, indicator pre-computation (most critical file). Reindexes precomputed indicators to match config-date-sliced data so iloc lookups stay aligned.
- `backtest/vectorized.py` — Vectorized screening engine for fast combine-scan (~100x vs candle-by-candle). NOT for live/paper — screening only. Computes indicators on the full df, then slices the indicator frame by timestamp so warmup is preserved.
- `backtest/holdout.py` — `resolve_holdout_window()` shared helper. Default behavior for backtest/combine/scan/combine-scan slices to the data's last 20% so rule-based runs are directly comparable to `ml-backtest` (which uses meta.json `holdout_start`). Precedence: `--start`/`--end` > `--include-train` > auto holdout. Multi-symbol picks the cutoff in the symbols' common timestamp window.
- `backtest/parallel.py` — Spawn-safe parallel workers for scan/combine-scan. Routes combined templates to vectorized engine (rule-based) or full engine (ML templates). Imports `holdout.resolve_holdout_window` directly (avoids pulling Typer/Console).
- `backtest/simulator.py` — Order fill simulation with slippage and fees (Upbit: 0.05%)
- `backtest/report.py` — Performance metrics: Sharpe, Sortino, max drawdown, win rate, profit factor
- `backtest/optimizer.py` — Grid search parameter optimization with parallel execution, optional `progress` parameter
- `backtest/walk_forward.py` — Walk-forward validation (train/test window rolling), optional `progress` parameter
- `ml/features.py` — 10 technical features + 6 optional external features (kimchi premium, funding rate, FNG, USD/KRW)
- `ml/targets.py` — 4h forward return binary classification target (offline only)
- `ml/trainer.py` — LGBMTrainer: train, evaluate, calibrate (isotonic), save/load (.lgb + _meta.json + _cal.json)
- `ml/walk_forward.py` — MLWalkForwardTrainer: purged expanding window + embargo (150 candles) + 20% holdout (half eval, half calibrator fit). Persists `train_end` / `holdout_start` / `holdout_end` / `data_end` + `avg_win_loss_ratio` to meta.json so downstream commands can slice and Kelly-size correctly.
- `ml/strategy_walk_forward.py` — MLStrategyWalkForward: time-honest LGBM walk-forward (`ml-walk-forward` CLI). Trains a fresh model per expanding window (Path B: single training with inner train/val split for early stopping), injects via `LGBMStrategy.set_model()`, then runs the standard backtest engine on the test window — no in-sample inference contamination.
- `ml/parallel.py` — Spawn-safe parallel training worker (ProcessPoolExecutor); forwards holdout AUC/precision so `ml-train-all` summary is accurate.
- `ml/utils.py` — Shared ML helpers (Half-Kelly position sizing with empirical avg_win/avg_loss ratio).
- `data/fetcher.py` — CCXT-based OHLCV download with Upbit rate limiting
- `data/external_fetcher.py` — External data: Binance OHLCV, funding rate, FRED USD/KRW, Fear & Greed Index
- `data/storage.py` — Parquet file I/O with auto-merge on save
- `data/indicators.py` — 19 technical indicator wrappers (SMA, EMA, RSI, MACD, BB, ATR, Stochastic, ADX, Ichimoku, Aroon, CCI, ROC, MFI, OBV, Keltner, Donchian, Z-score, PctFromMA, Volume SMA)
- `exchange/base.py` — Abstract exchange interface (BaseExchange ABC)
- `exchange/ccxt_exchange.py` — CCXT async implementation for Upbit (retry + rate limiting)
- `exchange/paper.py` — Paper trading exchange (simulated fills, portfolio tracking)
- `exchange/ws_client.py` — Upbit WebSocket client (real-time ticker, auto-reconnect, interruptible cooldown)
- `live/engine.py` — Async live/paper trading loop (polling, candle detection, signal execution). Syncs WebSocket prices to `PaperExchange` each tick, recalculates stop loss from filled price, enforces stop in `_tick_symbol` with notification, uses slippage-adjusted expected price for sizing/validation, records equity each tick.
- `live/state.py` — JSON-based state persistence with atomic write (positions, equity history, crash recovery)
- `live/order_manager.py` — Order lifecycle (submit, poll, timeout cancel, market re-order)
- `risk/manager.py` — Position sizing (fixed-fractional), drawdown circuit breaker, stop loss
- `risk/validators.py` — Pre-trade safety (max order size, daily loss limit, cooldown)
- `notifications/telegram.py` — Telegram Bot API notifications
- `dashboard/app.py` — Streamlit web dashboard (Live Monitor + Backtest Viewer)
- `config.py` — Pydantic settings from YAML + .env override
- `utils/logging.py` — Console + JSON file logging with daily rotation (LOG_DIR env)

### Data Format
Candle data stored as Parquet: `data/{SYMBOL}_{QUOTE}/{timeframe}.parquet`
(e.g., `data/BTC_KRW/1h.parquet`)
