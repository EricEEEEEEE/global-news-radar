# Global News Radar

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An event-first, source-verified Telegram radar for financial breaking news.
It watches continuously, but sends only fresh events that are too time-sensitive
to wait for a daily market brief.

It is deliberately **not** a price dashboard, technical-analysis feed, portfolio
manager, or trading bot.

中文说明：[README.zh.md](README.zh.md)

## What it does

- Polls central-bank and regulator RSS feeds every two minutes.
- Runs discovery across a configurable set of commercial news RSS feeds plus
  Google News, falling back to GDELT only when the batch comes back thin.
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

# Optional structured macro/earnings data
RADAR_FMP_ENABLED=false
FMP_API_KEY=

# Optional OpenAI-compatible translation/compression endpoint
RADAR_LLM_ENABLED=false
LLM_API_BASE=
LLM_API_KEY=
LLM_MODEL=
```

Use topic ID `0` when the target chat does not use Telegram forum topics. A key
left blank means "not set" and falls back to `config/radar.yaml`, so you can copy
the template and fill in only the lines you need.

Inspect one cycle without touching state — nothing is marked seen, no heartbeat
is written, and the conditional-GET cache is left alone, so this is safe to run
against a live deployment:

```bash
.venv/bin/python main.py once --readonly
```

Run one full cycle without sending. This does not send Telegram messages, but it
*does* consume state: items are marked seen and topic cooldowns advance. Running
it against a running daemon's state directory makes that daemon skip those items.

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

## Liveness watchdog

The daemon reports its own source failures, but a process that has died cannot
report that it died. `watchdog` closes that gap: it reads only the heartbeat,
keeps its own small state file, and never opens the SQLite database or the
outbox — a locked database is one of the failures it has to be able to report.

```bash
.venv/bin/python main.py watchdog --max-age-seconds 900 --send
```

Run it from cron rather than from the Supervisor that owns the daemon, so
Supervisor going down does not take the alarm with it:

```cron
*/10 * * * * cd /path/to/global-news-radar && ./.venv/bin/python main.py --root . watchdog --send >> logs/watchdog.log 2>&1
```

It alerts on the transition, not on every run: the first failure sends one
message to the monitor topic, `watchdog_alert_cooldown_hours` covers the rest of
the outage, and a `✅ recovered` notice with the outage duration closes it. An
outage that was never announced produces no lone recovery notice. Without
`--send` the check runs and prints only. Exit status is `0` healthy, `1` not, so
cron mail or an external ping service can key off it too.

This cannot report a host that is entirely down — nothing running on that host
can. It covers the daemon, not the machine.

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
- change the visible Telegram character budget.

Telegram credentials and routes belong in `.env`, never in tracked configuration.
The optional LLM and FMP collectors ship off, so a fresh clone cannot spend
someone's API quota by accident. Turn one on with `RADAR_LLM_ENABLED` or
`RADAR_FMP_ENABLED` in `.env`, next to the credential it needs — an operational
switch does not belong in a file every deployment would then have to keep out of
its own commits.

## Source-health alerts

A consecutive-failure threshold is a count, and the wall-clock time it buys
depends on how often the source is polled, so the threshold is split by tier:

- official feeds and FMP use `source_failure_alert_after` (3). Official feeds run
  every cycle, so that is about six minutes;
- discovery feeds use `source_failure_alert_after_discovery` (6). They only run
  once per `discovery.interval_cycles`, so three failures was half an hour of a
  commercial CDN hiccup — a poor reason to interrupt someone when ten other
  discovery feeds are still reporting.

An announced outage is closed by a `✅ source recovered` notice carrying its
duration, so a warning never hangs around with no way to learn it ended. Closing
the incident also lets a genuinely new outage alert again rather than waiting out
`source_error_cooldown_hours`, but it must first clear `source_realert_minutes`
so a flapping source cannot alert on every bounce.

## State and evidence

- `state/radar.sqlite3` — dedupe, delivery, source-health, P1, and lineage state.
- `state/heartbeat.json` — last completed cycle and provider health.
- `state/radar_events.jsonl` — sent-event export for downstream briefs.
- `outbox/latest.html` — last attempted Telegram HTML.
- `outbox/latest.json` — event keys, sources, VisualSpec, delivery state, and message ID.
- `logs/radar.log` — runtime log, self-rotating at 8 MB × 5 files.

Delivery state is recorded only after Telegram confirms success. Failed sends enter
a bounded retry queue; dead letters make the health command fail.

Every log line, alert body, and heartbeat passes through a redaction filter: URL
query credentials and any token loaded from `.env` are replaced with `***`, so a
failing request cannot write an API key into a file or a chat message.
`item_retention_days` and `delivery_retention_days` bound SQLite and outbox growth.

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
