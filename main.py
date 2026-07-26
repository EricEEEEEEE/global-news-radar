#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from radar.config import load_config
from radar.service import RadarService, check_health
from radar.util import RedactingFormatter, load_env, register_secrets
from radar.watchdog import run_watchdog


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Global real-time financial news radar"
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    once = sub.add_parser("once", help="run one collection cycle")
    once.add_argument("--send", action="store_true")
    once.add_argument(
        "--readonly",
        action="store_true",
        help="inspect a cycle without consuming state: nothing is marked seen",
    )

    run = sub.add_parser("run", help="run Supervisor-owned loop")
    run.add_argument("--send", action="store_true")

    health = sub.add_parser("health", help="check heartbeat freshness")
    health.add_argument("--max-age-seconds", type=int, default=600)

    watchdog = sub.add_parser(
        "watchdog", help="alert to Telegram when the daemon stops reporting"
    )
    watchdog.add_argument("--max-age-seconds", type=int, default=900)
    watchdog.add_argument(
        "--send",
        action="store_true",
        help="actually send; without it the check runs and prints only",
    )

    export = sub.add_parser("export", help="export sent events for Daily Brief")
    export.add_argument("--output", type=Path)
    return parser


def configure_logging(root: Path, verbose: bool) -> None:
    root.joinpath("logs").mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler(
            root / "logs" / "radar.log",
            maxBytes=8 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        ),
    ]
    # A third-party library logging a failed request would otherwise write the
    # full URL, API key included, into a file Supervisor keeps for weeks.
    formatter = RedactingFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    for handler in handlers:
        handler.setFormatter(formatter)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        handlers=handlers,
    )


def main() -> int:
    args = build_parser().parse_args()
    root = args.root.resolve()
    config_path = args.config or root / "config" / "radar.yaml"
    env_path = args.env_file or root / ".env"
    config = load_config(config_path)
    env = load_env(env_path)
    # Register before the first log line, not after the service is built, so
    # nothing can leak in the window between startup and construction.
    register_secrets(
        env.get(name, "")
        for name in (
            "FMP_API_KEY",
            "TELEGRAM_BOT_TOKEN",
            "LLM_API_KEY",
            "FE_LLM_API_KEY",
        )
    )
    configure_logging(root, args.verbose)

    if args.command == "health":
        ok, payload = check_health(root, args.max_age_seconds)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if ok else 1

    # The watchdog runs from cron while the daemon owns the process, so it is
    # built without RadarService: it must not open the store the daemon writes.
    if args.command == "watchdog":
        ok, report = run_watchdog(
            root,
            config,
            env,
            max_age_seconds=args.max_age_seconds,
            send_enabled=bool(args.send),
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if ok else 1

    send_enabled = bool(getattr(args, "send", False))
    service = RadarService(
        root=root,
        config=config,
        env=env,
        send_enabled=send_enabled,
    )
    try:
        if args.command == "once":
            stats = service.run_once(readonly=bool(args.readonly))
            print(json.dumps(stats, ensure_ascii=False, indent=2))
            return 0
        if args.command == "run":
            service.run_forever()
            return 0
        if args.command == "export":
            output = args.output or root / "state" / "radar_events.jsonl"
            count = service.store.export_deliveries(output)
            print(json.dumps({"exported": count, "output": str(output)}))
            return 0
    finally:
        service.close()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
