"""Slack notifier for unified_requirements.txt."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import requests

REQ_FILE = Path("unified_requirements.txt")
PREVIEW_LINES = 20
PREVIEW_CHAR_LIMIT = 1200
PROD_CHANNEL = "C09NQQBQ28G"
DEV_CHANNEL = "C09NQQGNVTJ"


@dataclass(slots=True)
class Context:
    run_url: str
    token: str
    env_label: str


def detect_env() -> str:
    event = os.environ.get("GITHUB_EVENT_NAME", "").lower()
    return "PROD" if event == "schedule" else "DEV"


def preview_requirements() -> tuple[str, str]:
    """Return (preview_text, status_icon) for unified_requirements.txt."""

    if not REQ_FILE.exists():
        return "unified_requirements.txt missing", ":x:"

    lines = REQ_FILE.read_text(encoding="utf-8").splitlines()
    if not lines:
        return "unified_requirements.txt is empty", ":warning:"

    tail = len(lines) - PREVIEW_LINES
    body = lines[:PREVIEW_LINES]
    if tail > 0:
        body.append(f"... ({tail} more lines)")

    snippet = "\n".join(body)
    if len(snippet) > PREVIEW_CHAR_LIMIT:
        snippet = f"{snippet[:PREVIEW_CHAR_LIMIT].rstrip()}\n... (truncated)"

    return snippet, ":white_check_mark:"


def build_blocks(
    ctx: Context, preview: str, status_icon: str
) -> list[dict[str, object]]:
    link = f"<{ctx.run_url}|link>" if ctx.run_url else "not provided"
    return [
        {"type": "header", "text": {"type": "plain_text", "text": "Deps Requirements"}},
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Environment:* {ctx.env_label}"},
                {"type": "mrkdwn", "text": f"*Workflow:* {link}"},
                {"type": "mrkdwn", "text": f"*Status:* {status_icon}"},
            ],
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*unified_requirements.txt*\n```{preview}```",
            },
        },
    ]


def slack_channel(env_label: str) -> str:
    return DEV_CHANNEL if env_label == "DEV" else PROD_CHANNEL


def send_to_slack(ctx: Context, blocks: list[dict[str, object]]) -> None:
    payload = {"channel": slack_channel(ctx.env_label), "blocks": blocks}
    resp = requests.post(
        "https://slack.com/api/chat.postMessage",
        headers={"Authorization": f"Bearer {ctx.token}"},
        json=payload,
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    if not data.get("ok", False):
        raise RuntimeError(f"Slack API error: {data.get('error', 'unknown error')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--github_action_url",
        default="",
        help="Workflow run URL for the Slack link",
    )
    parser.add_argument(
        "--slack_bot_token",
        required=True,
        help="Slack bot token with chat:write",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ctx = Context(
        run_url=args.github_action_url,
        token=args.slack_bot_token,
        env_label=detect_env(),
    )
    preview, status_icon = preview_requirements()
    blocks = build_blocks(ctx, preview, status_icon)
    send_to_slack(ctx, blocks)


if __name__ == "__main__":
    main()
