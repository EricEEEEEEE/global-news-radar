# Global News Radar

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An event-first, source-verified Telegram radar for financial breaking news.
It watches continuously, but sends only fresh events that are too time-sensitive
to wait for a daily market brief.

It is deliberately **not** a price dashboard, technical-analysis feed, portfolio
manager, or trading bot.

## What it does

- Polls central-bank and regulator RSS feeds every two minutes.
- Runs lightweight discovery through Google News RSS, with GDELT as fallback.
- Supports optional structured macro and earnings data from FMP.
- Applies deterministic P0/P1 rules before any LLM is called.
- Requires independent corroboration for high-impact claims from secondary media.
- Sends P0 immediately; batches P1 only when at least two distinct events exist.
- Enforces freshness, six-hour topic cooldowns, cross-day lineage, and a quiet window.
- Renders mobile-first Telegram HTML with clickable, per-event provenance.
- Stores SQLite state, JSONL delivery exports, heartbeat data, and auditable outbox files.

The optional LLM is only a Chinese translation/compression layer. It cannot choose
severity, invent numbers, change event keys, or create causal claims.

## Alert shape

```text
🚨 Event itself
Core fact
Immediate impact
Occurred 17:55 · Radar 18:00 SGT
Source · #category
```

P1 digests keep every fact bound to its own occurrence time and source. Telegram
HTML automatically degrades to complete plain text if entity parsing fails.

## Quick start

```bash
git clone https://github.com/EricEEEEEEE/global-news-radar.git
cd global-news-radar
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`:

```dotenv
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
TELEGRAM_NEWS_THREAD_ID=0
TELEGRAM_MONITOR_THREAD_ID=0

# Optional structured data
FMP_API_KEY=

# Optional OpenAI-compatible translation/compression endpoint
LLM_API_BASE=
LLM_API_KEY=
LLM_MODEL=
```

Use topic ID `0` when the target chat does not use Telegram forum topics.

Run one safe collection cycle without sending:

```bash
.venv/bin/python main.py once
```

The first run builds a silent baseline so old articles are never replayed as
breaking news. Enable sending explicitly only after inspecting the dry run:

```bash
.venv/bin/python main.py once --send
```

## Continuous operation

```bash
.venv/bin/python main.py run --send
```

`supervisor.conf.example` is a starting point. Replace its installation path and
service user before loading it into Supervisor.

Health check:

```bash
.venv/bin/python main.py health --max-age-seconds 600
```

Export confirmed deliveries:

```bash
.venv/bin/python main.py export
```

## Configuration

The default runtime uses `Asia/Singapore`, including its regional query rotation
and `21:00–21:30` quiet window. Edit `config/radar.yaml` to:

- add or remove official RSS feeds;
- choose discovery and structured-data intervals;
- tune only the documented deterministic thresholds;
- enable the optional LLM or FMP collectors;
- change the visible Telegram character budget.

Telegram credentials and routes belong in `.env`, never in tracked configuration.

## State and evidence

- `state/radar.sqlite3` — dedupe, delivery, source-health, P1, and lineage state.
- `state/heartbeat.json` — last completed cycle and provider health.
- `state/radar_events.jsonl` — sent-event export for downstream briefs.
- `outbox/latest.html` — last attempted Telegram HTML.
- `outbox/latest.json` — event keys, sources, VisualSpec, delivery state, and message ID.
- `logs/radar.log` — runtime log.

Delivery state is recorded only after Telegram confirms success. Failed sends enter
a bounded retry queue; dead letters make the health command fail.

## Visual contract

The renderer follows the
[TG Watch Visual System](https://github.com/EricEEEEEEE/TG-watch-skill):

1. answer first;
2. evidence second;
3. time and provenance last.

`config/visual_spec.json` records the medium decision. Every real outbox manifest
contains a source-bound VisualSpec for that message. The radar stays text-first:
short breaking events do not become decorative images.

## Tests

```bash
PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q main.py radar tests
```

The suite covers policy thresholds, freshness, corroboration, dedupe, cross-day
lineage, quiet-window delivery, long CJK rendering, HTML safety, outbox provenance,
and forced plain-text fallback.

## Safety

- Read-only public data sources.
- No broker, exchange, wallet, private key, order, or transaction integration.
- No automated trading.
- No analyst forecasts or long-term outlook generation.
- No claim of observed market reaction without supplied evidence.

This project is research and notification infrastructure, not financial advice.

## License

[MIT](LICENSE)
