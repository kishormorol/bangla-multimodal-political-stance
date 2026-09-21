"""Collect new candidate items into an annotation pool.

    bmpb collect --days 30 --limit 800

Writes `data/interim/candidates.csv`. Nothing here assigns a label — the output
is the pool that annotators work through, and it is deliberately kept separate
from `data/processed/corpus.csv` so an unannotated candidate can never leak into
a training split.

Politeness is not optional in this module: every request checks the outlet's
robots.txt, identifies itself, and waits `--delay` seconds. The defaults are
slow on purpose. If you raise them, you are the one talking to someone else's
webserver.
"""

from __future__ import annotations

import time
import urllib.robotparser
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pandas as pd

from bmpb.data.sources import (
    FETCHABLE,
    OUTLETS,
    USER_AGENT,
    Outlet,
    looks_political,
    parse_article,
)
from bmpb.paths import CORPUS, INTERIM, ensure_dirs
from bmpb.utils.logging import get_logger

log = get_logger(__name__)

CANDIDATES = INTERIM / "candidates.csv"


@dataclass
class Fetcher:
    """A rate-limited, robots-aware HTTP getter.

    `timeout` is urllib's socket timeout, which only fires when a server goes
    quiet — a host that dribbles bytes, or one whose TLS handshake stalls, can
    hold a request open indefinitely. `max_bytes` caps how much is read and
    `hard_deadline` caps the wall time of a single fetch, so one bad host costs
    seconds rather than the whole run. Both were added after a backfill hung for
    hours on 25 URLs.
    """

    delay: float = 1.5
    timeout: int = 15
    hard_deadline: float = 30.0
    max_bytes: int = 4_000_000
    _robots: dict[str, urllib.robotparser.RobotFileParser | None] = None  # type: ignore[assignment]
    _last_request: float = 0.0

    def __post_init__(self) -> None:
        self._robots = {}

    def allowed(self, url: str) -> bool:
        parts = urlparse(url)
        host = parts.netloc
        if host not in self._robots:
            self._robots[host] = self._read_robots(f"{parts.scheme}://{host}/robots.txt", host)
        parser = self._robots[host]
        return bool(parser and parser.can_fetch(USER_AGENT, url))

    def _read_robots(self, robots_url: str, host: str):
        """Fetch robots.txt with our own User-Agent, then parse it.

        RobotFileParser.read() uses urllib's default `Python-urllib/3.x` agent,
        which several of these outlets answer with 403. The parser treats that as
        "disallow everything", so every URL on the host looked forbidden. Fetching
        the file ourselves with the agent we actually identify as gives the real
        rules.
        """
        parser = urllib.robotparser.RobotFileParser()
        try:
            request = Request(robots_url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=self.timeout) as response:
                body = response.read(self.max_bytes).decode("utf-8", errors="ignore")
        except HTTPError as error:
            if error.code in (401, 403):
                log.warning(
                    "%s returned HTTP %s for robots.txt; treating the host as " "off-limits",
                    host,
                    error.code,
                )
                return None
            # 404 means no robots.txt, which conventionally means no restrictions.
            log.info("%s has no robots.txt (HTTP %s); proceeding", host, error.code)
            parser.parse([])
            return parser
        except (URLError, TimeoutError, OSError) as error:
            log.warning(
                "could not reach robots.txt for %s (%s); skipping the host",
                host,
                type(error).__name__,
            )
            return None

        parser.parse(body.splitlines())
        return parser

    def get(self, url: str) -> str | None:
        if not self.allowed(url):
            log.debug("robots.txt disallows %s", url)
            return None

        elapsed = time.monotonic() - self._last_request
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_request = time.monotonic()

        started = time.monotonic()
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=self.timeout) as response:
                chunks, total = [], 0
                while True:
                    if time.monotonic() - started > self.hard_deadline:
                        log.debug("%s -> exceeded %.0fs deadline", url, self.hard_deadline)
                        return None
                    chunk = response.read(65536)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > self.max_bytes:
                        log.debug("%s -> larger than %d bytes; truncating", url, self.max_bytes)
                        break
                    chunks.append(chunk)
                return b"".join(chunks).decode("utf-8", errors="ignore")
        except HTTPError as error:
            log.debug("%s -> HTTP %s", url, error.code)
        except (URLError, TimeoutError, OSError, ValueError) as error:
            log.debug("%s -> %s", url, type(error).__name__)
        return None


def _urls_from_sitemap(page: str) -> list[str]:
    import re

    return re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", page)


def _urls_from_feed(page: str) -> list[str]:
    import re

    links = re.findall(r"<link[^>]*>\s*([^<\s]+)\s*</link>", page)
    links += re.findall(r'<link[^>]+href=["\']([^"\']+)["\']', page)
    return [link for link in links if link.startswith("http")]


def discover(outlet: Outlet, fetcher: Fetcher, days: int) -> list[str]:
    """Article URLs for one outlet, from its sitemaps or feeds."""
    found: list[str] = []

    for sitemap in outlet.sitemaps_for(days):
        page = fetcher.get(sitemap)
        if not page:
            continue
        urls = [u for u in _urls_from_sitemap(page) if outlet.wants(u)]
        log.info("  %s -> %d urls", sitemap.rsplit("/", 1)[-1], len(urls))
        found.extend(urls)

    for feed in outlet.feeds:
        page = fetcher.get(feed)
        if not page:
            continue
        urls = [u for u in _urls_from_feed(page) if outlet.wants(u)]
        log.info("  %s -> %d urls", feed.rsplit("/", 1)[-1], len(urls))
        found.extend(urls)

    # Preserve discovery order while dropping repeats.
    return list(dict.fromkeys(found))


def known_urls() -> set[str]:
    """URLs already in the corpus, so collection never re-offers them."""
    if not CORPUS.exists():
        return set()
    corpus = pd.read_csv(CORPUS)
    if "source_url" not in corpus:
        return set()
    return {str(u).strip() for u in corpus["source_url"].dropna()}


def collect(
    days: int = 30,
    limit: int = 800,
    delay: float = 1.5,
    outlets: list[str] | None = None,
    require_political: bool = True,
    out: Path = CANDIDATES,
) -> pd.DataFrame:
    ensure_dirs()
    fetcher = Fetcher(delay=delay)
    seen = known_urls()

    existing = pd.read_csv(out) if out.exists() else pd.DataFrame()
    if len(existing):
        seen |= {str(u).strip() for u in existing.get("source_url", pd.Series(dtype=str)).dropna()}
        log.info("resuming: %d candidates already collected", len(existing))

    selected = outlets or FETCHABLE
    rows: list[dict] = []

    for key in selected:
        outlet = OUTLETS.get(key)
        if outlet is None:
            log.warning("unknown outlet %r; known: %s", key, sorted(OUTLETS))
            continue
        if not (outlet.sitemap_daily or outlet.feeds):
            log.info("skipping %s (%s)", outlet.name, outlet.notes)
            continue

        log.info("discovering %s", outlet.name)
        urls = [u for u in discover(outlet, fetcher, days) if u not in seen]
        log.info("%s: %d new candidate urls", outlet.name, len(urls))

        for url in urls:
            if len(rows) + len(existing) >= limit:
                log.info("reached limit of %d", limit)
                break
            page = fetcher.get(url)
            if not page:
                continue
            record = parse_article(page, url, outlet.name)
            if record is None:
                continue
            if require_political and not looks_political(f"{record['title']} {record['summary']}"):
                continue
            rows.append(record)
            seen.add(url)
            if len(rows) % 25 == 0:
                log.info("  %d collected", len(rows))
        if len(rows) + len(existing) >= limit:
            break

    fresh = pd.DataFrame(rows)
    combined = pd.concat([existing, fresh], ignore_index=True) if len(existing) else fresh
    if len(combined):
        combined = combined.drop_duplicates(subset="source_url").reset_index(drop=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out, index=False)
        log.info("wrote %s (%d candidates, %d new)", out, len(combined), len(fresh))
        if len(fresh):
            log.info("median body length: %d chars", int(fresh["body_chars"].median()))
            log.info("by outlet: %s", fresh["outlet"].value_counts().to_dict())
    else:
        log.warning("collected nothing — check connectivity and the outlet adapters")
    return combined


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--limit", type=int, default=800)
    parser.add_argument("--delay", type=float, default=1.5)
    parser.add_argument("--outlets", nargs="*", default=None)
    parser.add_argument("--all-topics", action="store_true", help="skip the political-term filter")
    args = parser.parse_args()
    collect(
        days=args.days,
        limit=args.limit,
        delay=args.delay,
        outlets=args.outlets,
        require_political=not args.all_topics,
    )
