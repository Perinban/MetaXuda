#!/usr/bin/env python3
"""Collect GitHub repository traffic and generate branch-hosted public stats."""

from __future__ import annotations

import html
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "analytics" / "traffic.json"
SUMMARY_PATH = ROOT / "TRAFFIC.md"
SVG_PATH = ROOT / "traffic.svg"


def github_get(url: str, token: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "metaxuda-traffic-collector",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"GitHub API request failed ({error.code}): {detail}"
        ) from error


def load_history() -> dict[str, Any]:
    if not DATA_PATH.exists():
        return {"tracked_from": None, "updated_at": None, "days": {}}
    with DATA_PATH.open(encoding="utf-8") as file:
        data = json.load(file)
    data.setdefault("tracked_from", None)
    data.setdefault("updated_at", None)
    data.setdefault("days", {})
    return data


def merge_daily(history: dict[str, Any], key: str, entries: list[dict[str, Any]]) -> None:
    days = history["days"]
    for entry in entries:
        date = entry["timestamp"][:10]
        record = days.setdefault(
            date,
            {
                "views": 0,
                "unique_visitors": 0,
                "clones": 0,
                "unique_cloners": 0,
            },
        )
        if key == "views":
            record["views"] = int(entry.get("count", 0))
            record["unique_visitors"] = int(entry.get("uniques", 0))
        else:
            record["clones"] = int(entry.get("count", 0))
            record["unique_cloners"] = int(entry.get("uniques", 0))


def totals(history: dict[str, Any]) -> dict[str, int]:
    days = history["days"]
    return {
        key: sum(int(record.get(key, 0)) for record in days.values())
        for key in ("views", "unique_visitors", "clones", "unique_cloners")
    }


def tracked_label(history: dict[str, Any]) -> str:
    tracked_from = history.get("tracked_from")
    if not tracked_from:
        return "Not started"
    return datetime.strptime(tracked_from, "%Y-%m-%d").strftime("%B %-d, %Y")


def render_markdown(history: dict[str, Any]) -> str:
    count = totals(history)
    return f"""# MetaXuda Repository Traffic

| Views | Daily unique visitors | Clones | Daily unique cloners |
|---:|---:|---:|---:|
| {count['views']:,} | {count['unique_visitors']:,} | {count['clones']:,} | {count['unique_cloners']:,} |

Tracked from: **{tracked_label(history)}**
"""


def render_svg(history: dict[str, Any]) -> str:
    count = totals(history)
    labels = ["Views", "Daily unique visitors", "Clones", "Daily unique cloners"]
    values = [
        count["views"],
        count["unique_visitors"],
        count["clones"],
        count["unique_cloners"],
    ]
    cells: list[str] = []
    width = 220
    for index, (label, value) in enumerate(zip(labels, values, strict=True)):
        x = index * width
        cells.append(
            f'<rect x="{x}" y="0" width="{width}" height="96" fill="#ffffff" '
            f'stroke="#d0d7de"/>'
            f'<text x="{x + width / 2}" y="34" text-anchor="middle" '
            f'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif" '
            f'font-size="14" fill="#57606a">{html.escape(label)}</text>'
            f'<text x="{x + width / 2}" y="68" text-anchor="middle" '
            f'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif" '
            f'font-size="24" font-weight="600" fill="#24292f">{value:,}</text>'
        )
    tracked = html.escape(tracked_label(history))
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="880" height="120" '
        'viewBox="0 0 880 120" role="img" aria-label="MetaXuda repository traffic">'
        + "".join(cells)
        + '<rect x="0" y="96" width="880" height="24" fill="#f6f8fa" stroke="#d0d7de"/>'
        + f'<text x="440" y="113" text-anchor="middle" '
        + 'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif" '
        + f'font-size="12" fill="#57606a">Tracked from {tracked}</text>'
        + "</svg>\n"
    )


def main() -> int:
    token = os.environ.get("TRAFFIC_TOKEN") or os.environ.get("GH_TOKEN")
    repository = os.environ.get("GITHUB_REPOSITORY")
    if not token or not repository:
        print("TRAFFIC_TOKEN/GH_TOKEN and GITHUB_REPOSITORY are required", file=sys.stderr)
        return 2

    base = f"https://api.github.com/repos/{repository}/traffic"
    views = github_get(f"{base}/views?per=day", token)
    clones = github_get(f"{base}/clones?per=day", token)

    history = load_history()
    merge_daily(history, "views", views.get("views", []))
    merge_daily(history, "clones", clones.get("clones", []))

    ordered_dates = sorted(history["days"])
    if ordered_dates and not history.get("tracked_from"):
        history["tracked_from"] = ordered_dates[0]
    history["updated_at"] = datetime.now(timezone.utc).isoformat()

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    SUMMARY_PATH.write_text(render_markdown(history), encoding="utf-8")
    SVG_PATH.write_text(render_svg(history), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
