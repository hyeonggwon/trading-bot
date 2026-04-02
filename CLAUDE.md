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
tradingbot live --strategy sma_cross --symbol BTC/KRW --max-order 500000 --daily-loss-limit 200000
tradingbot status
tradingbot balance
tradingbot data-list
tradingbot symbols

# Multi-symbol backtest (uses all symbols from config/default.yaml)
tradingbot backtest --strategy sma_cross
# Single symbol override
tradingbot backtest --strategy sma_cross --symbol BTC/KRW
```

## Architecture

### Anti-Lookahead Design (Core Principle)
The backtest engine iterates candle-by-candle, only passing confirmed (closed) candles to strategy methods. The strategy can never access the current incomplete candle or future data. This is enforced structurally in `backtest/engine.py` — not by convention.

### Key Flow: Backtest Loop (Multi-Symbol)
```
Build unified timeline from all symbols' timestamps

For each timestamp ts:
  For each symbol with data at ts:
    idx = position of ts in symbol's data
    visible_df = symbol_candles[0..idx-1]  ← strategy sees ONLY past candles
    fill_candle = symbol_candles[idx]       ← used for execution

    1. Check stop losses against fill_candle's OHLC (per symbol)
    2. Fill pending orders for this symbol only
    3. strategy.indicators(visible_df)
    4. strategy.should_exit(visible_df) for open positions
    5. strategy.should_entry(visible_df) — blocked if stop loss fired this candle
    6. Risk manager validates signals (max_open_positions across portfolio)
    7. Approved signals → fill at fill_candle's open + slippage

  Update last_known_prices & record portfolio equity snapshot
```

### Module Responsibilities
- `core/models.py` — Frozen dataclasses: Candle, Signal, Order, Trade, Position, PortfolioState
- `strategy/base.py` — Abstract `Strategy` class with `indicators()`, `should_entry()`, `should_exit()`
- `backtest/engine.py` — Core backtest loop, multi-symbol support (most critical file)
- `backtest/simulator.py` — Order fill simulation with slippage and fees (Upbit: 0.05%)
- `backtest/report.py` — Performance metrics: Sharpe, Sortino, max drawdown, win rate, profit factor
- `risk/manager.py` — Position sizing (fixed-fractional), drawdown circuit breaker, stop loss
- `data/fetcher.py` — CCXT-based OHLCV download with Upbit rate limiting
- `data/storage.py` — Parquet file I/O with auto-merge on save
- `data/indicators.py` — Technical indicator wrappers using `ta` library
- `backtest/optimizer.py` — Grid search parameter optimization with parallel execution
- `backtest/walk_forward.py` — Walk-forward validation (train/test window rolling)
- `exchange/base.py` — Abstract exchange interface (BaseExchange ABC)
- `exchange/ccxt_exchange.py` — CCXT async implementation for Upbit (retry + rate limiting)
- `exchange/paper.py` — Paper trading exchange (simulated fills, portfolio tracking)
- `live/engine.py` — Async live/paper trading loop (polling, candle detection, signal execution)
- `live/state.py` — JSON-based state persistence (positions, equity history, crash recovery)
- `live/order_manager.py` — Order lifecycle (submit, poll, timeout cancel, market re-order)
- `risk/validators.py` — Pre-trade safety (max order size, daily loss limit, cooldown)
- `notifications/telegram.py` — Telegram Bot API notifications
- `config.py` — Pydantic settings from YAML + .env override

### Built-in Strategies
- `sma_cross` — SMA golden/dead cross (fast_period, slow_period)
- `rsi_mean_reversion` — RSI oversold entry / overbought exit (rsi_period, oversold, overbought)
- `macd_momentum` — MACD histogram zero-cross (fast, slow, signal)
- `bollinger_breakout` — Price breaks upper BB / drops below middle BB (period, std)

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
