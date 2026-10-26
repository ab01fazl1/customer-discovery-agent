"""Message sources. The pipeline only needs `.messages()` -> iterator of Message.

CsvSource / JsonlSource are for tests. TelegramSource is the real thing and is
the ONLY file that knows Telegram exists.
"""
from __future__ import annotations

import csv
import json
from typing import Iterator, Optional, Protocol

from .models import Message


class MessageSource(Protocol):
    def messages(self) -> Iterator[Message]: ...


def _to_ref(v) -> Optional[str]:
    return None if v in (None, "") else str(v)


def _to_label(v) -> Optional[int]:
    if v in (None, ""):
        return None
    return int(v)


class CsvSource:
    """Columns: id, text (required); timestamp, author, chat, label, reply_to (optional)."""

    def __init__(self, path: str, limit: Optional[int] = None):
        self.path, self.limit = path, limit

    def messages(self) -> Iterator[Message]:
        with open(self.path, newline="", encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f)):
                if self.limit and i >= self.limit:
                    return
                yield Message(
                    id=str(row.get("id") or i),
                    text=row["text"],
                    author=row.get("author", ""),
                    timestamp=row.get("timestamp", ""),
                    chat=row.get("chat", ""),
                    label=_to_label(row.get("label")),
                    reply_to=_to_ref(row.get("reply_to")),
                )


class JsonlSource:
    """One JSON object per line with the same fields as CsvSource."""

    def __init__(self, path: str, limit: Optional[int] = None):
        self.path, self.limit = path, limit

    def messages(self) -> Iterator[Message]:
        with open(self.path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if self.limit and i >= self.limit:
                    return
                if not line.strip():
                    continue
                d = json.loads(line)
                yield Message(
                    id=str(d.get("id", i)),
                    text=d["text"],
                    author=d.get("author", ""),
                    timestamp=d.get("timestamp", ""),
                    chat=d.get("chat", ""),
                    label=_to_label(d.get("label")),
                    reply_to=_to_ref(d.get("reply_to")),
                )


class TelegramSource:
    """Reads a group's history through a Telethon user account (batch mode).

    pip install telethon; get api_id/api_hash from https://my.telegram.org.
    First run asks for your phone number and login code, then caches the session.
    """

    def __init__(self, group, api_id: int, api_hash: str, session: str = "discovery",
                 limit: Optional[int] = None, since=None):
        self.group, self.api_id, self.api_hash = group, api_id, api_hash
        self.session, self.limit, self.since = session, limit, since

    def messages(self) -> Iterator[Message]:
        from telethon.sync import TelegramClient  # imported lazily: optional dependency

        with TelegramClient(self.session, self.api_id, self.api_hash) as client:
            # reverse=True + offset_date => oldest-first, starting at `since`
            for m in client.iter_messages(self.group, limit=self.limit,
                                          reverse=True, offset_date=self.since):
                if not m.message:
                    continue
                yield Message(
                    id=str(m.id),
                    text=m.message,
                    author=str(m.sender_id or ""),
                    timestamp=m.date.isoformat(),
                    chat=str(self.group),
                    reply_to=_to_ref(getattr(m, "reply_to_msg_id", None)),
                )
