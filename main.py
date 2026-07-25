#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from radar.config import load_config
from radar.service import RadarService, check_health
from radar.util import load_env


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

    run = sub.add_parser("run", help="run Supervisor-owned loop")
    run.add_argument("--send", action="store_true")

    health = sub.add_parser("health", help="check heartbeat freshness")
    health.add_argument("--max-age-seconds", type=int, default=600)

    export = sub.add_parser("export", help="export sent events for Daily Brief")
    export.add_argument("--output", type=Path)
    return parser


def configure_logging(root: Path, verbose: bool) -> None:
    root.joinpath("logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(root / "logs" / "radar.log", encoding="utf-8"),
        ],
    )


def main() -> int:
    args = build_parser().parse_args()
    root = args.root.resolve()
    config_path = args.config or root / "config" / "radar.yaml"
    env_path = args.env_file or root / ".env"
    config = load_config(config_path)
    env = load_env(env_path)
    configure_logging(root, args.verbose)

    if args.command == "health":
        ok, payload = check_health(root, args.max_age_seconds)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
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
            print(json.dumps(service.run_once(), ensure_ascii=False, indent=2))
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
