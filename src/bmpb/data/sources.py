"""Per-outlet adapters for discovering and parsing article pages.

Discovery uses whatever archive interface an outlet publishes — a daily sitemap
where one exists, an RSS feed otherwise — rather than crawling link graphs. That
is both politer and more reproducible: a sitemap for a given date returns the
same set of URLs tomorrow.

Parsing goes through OpenGraph tags and schema.org JSON-LD, which every outlet
here emits, so one parser covers all of them and we are not reverse-engineering
each site's HTML structure.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from datetime import date, timedelta

USER_AGENT = (
    "bmpb-research/0.1 (academic dataset collection for Bangla political bias research; "
    "contact kishormorol.working@gmail.com)"
)


@dataclass
class Outlet:
    """One news source and how to enumerate its articles.

    `sitemap_daily` is a strftime template producing one sitemap URL per day.
    `feeds` are RSS URLs, used when an outlet publishes no dated sitemap — they
    only surface recent items, so a feed-only outlet has to be polled over
    several days to accumulate a batch.
    `path_sections` filters discovered URLs to the sections worth reading; an
    empty list keeps everything.
    """

    name: str
    base: str
    sitemap_daily: str | None = None
    feeds: list[str] = field(default_factory=list)
    path_sections: list[str] = field(default_factory=list)
    notes: str = ""

    def sitemaps_for(self, days: int, until: date | None = None) -> list[str]:
        if not self.sitemap_daily:
            return []
        until = until or date.today()
        return [
            (until - timedelta(days=offset)).strftime(self.sitemap_daily) for offset in range(days)
        ]

    def wants(self, url: str) -> bool:
        if not self.path_sections:
            return True
        path = url[len(self.base) :] if url.startswith(self.base) else url
        return any(path.lstrip("/").startswith(section) for section in self.path_sections)


# Verified reachable on 21 Sep 2026. Outlets that return 403 to a scripted
# request are listed with a note rather than silently dropped, so the set can be
# revisited without re-probing from scratch.
OUTLETS: dict[str, Outlet] = {
    "prothom_alo": Outlet(
        name="Prothom Alo",
        base="https://www.prothomalo.com",
        sitemap_daily="https://www.prothomalo.com/sitemap/sitemap-daily-%Y-%m-%d.xml",
        path_sections=["politics", "bangladesh", "opinion"],
        notes="~280 urls/day, of which ~7 politics and ~93 bangladesh",
    ),
    "bbc_bangla": Outlet(
        name="BBC Bangla",
        base="https://www.bbc.com",
        feeds=[
            "https://feeds.bbci.co.uk/bengali/rss.xml",
            "https://feeds.bbci.co.uk/bengali/topics/c907347rezkt/rss.xml",
        ],
        notes="feed only, ~14 items per fetch; poll daily to accumulate",
    ),
    "daily_star": Outlet(
        name="The Daily Star",
        base="https://www.thedailystar.net",
        feeds=["https://www.thedailystar.net/rss.xml"],
        notes="feed only, ~8 items per fetch; English-language",
    ),
    # These returned HTTP 403 to a scripted request with a descriptive
    # User-Agent. They are in the existing dataset, so if coverage matters they
    # need either a different access route or manual collection.
    "jugantor": Outlet(
        name="Jugantor",
        base="https://www.jugantor.com",
        notes="403 to scripted requests (feed and rss); news_sitemap.xml exists but is gated",
    ),
    "samakal": Outlet(name="Samakal", base="https://samakal.com", notes="403 to scripted requests"),
    "kaler_kantho": Outlet(
        name="Kaler Kantho", base="https://www.kalerkantho.com", notes="403 to scripted requests"
    ),
    "dhaka_post": Outlet(
        name="Dhaka Post", base="https://www.dhakapost.com", notes="403 to scripted requests"
    ),
    "bangla_tribune": Outlet(
        name="Bangla Tribune",
        base="https://www.banglatribune.com",
        notes="403 to scripted requests",
    ),
}

FETCHABLE = [key for key, outlet in OUTLETS.items() if outlet.sitemap_daily or outlet.feeds]

# Political salience filter for section feeds that mix topics (e.g. `bangladesh`,
# which carries both politics and crime/weather). A headline needs one of these
# to enter the annotation pool. Deliberately broad — precision comes from the
# annotators, and a recall miss here is an item we never even look at.
POLITICAL_TERMS = [
    "সরকার",
    "রাজনীতি",
    "রাজনৈতিক",
    "নির্বাচন",
    "ভোট",
    "সংসদ",
    "মন্ত্রী",
    "প্রধানমন্ত্রী",
    "রাষ্ট্রপতি",
    "উপদেষ্টা",
    "আওয়ামী",
    "বিএনপি",
    "জামায়াত",
    "জাতীয় পার্টি",
    "এনসিপি",
    "ছাত্রদল",
    "ছাত্রশিবির",
    "ছাত্রলীগ",
    "যুবদল",
    "হাসিনা",
    "খালেদা",
    "তারেক",
    "ইউনূস",
    "আন্দোলন",
    "বিক্ষোভ",
    "হরতাল",
    "মিছিল",
    "সমাবেশ",
    "দল",
    "নেতা",
    "নেত্রী",
    "এমপি",
    "সংবিধান",
    "তত্ত্বাবধায়ক",
    "অন্তর্বর্তী",
    "ট্রাইব্যুনাল",
    "দুর্নীতি",
    "মামলা",
    "গ্রেপ্তার",
    "রিমান্ড",
    "বিচার",
    "রায়",
    "কমিশন",
    "সংস্কার",
    "জুলাই",
    "গণঅভ্যুত্থান",
]

_POLITICAL_RE = re.compile("|".join(map(re.escape, POLITICAL_TERMS)))


def looks_political(text: str) -> bool:
    return bool(text) and bool(_POLITICAL_RE.search(text))


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

_TAG = re.compile(r"<[^>]+>")


def _meta(page: str, key: str) -> str | None:
    """Read an OpenGraph / meta tag, tolerating either attribute order."""
    quoted = re.escape(key)
    for pattern in (
        r'<meta[^>]+(?:property|name)=["\']' + quoted + r'["\'][^>]*?content=["\']([^"\']*)',
        r'<meta[^>]*?content=["\']([^"\']*)["\'][^>]+(?:property|name)=["\']' + quoted + r'["\']',
    ):
        match = re.search(pattern, page, re.I)
        if match:
            return html.unescape(match.group(1)).strip()
    return None


def _json_ld_article(page: str) -> dict:
    """The first Article/NewsArticle JSON-LD block, or {}."""
    blocks = re.findall(r"<script[^>]+application/ld\+json[^>]*>(.*?)</script>", page, re.S | re.I)
    for block in blocks:
        try:
            payload = json.loads(block.strip())
        except json.JSONDecodeError:
            continue
        candidates = payload if isinstance(payload, list) else [payload]
        graph = payload.get("@graph") if isinstance(payload, dict) else None
        if graph:
            candidates = graph
        for item in candidates:
            if isinstance(item, dict) and "Article" in str(item.get("@type", "")):
                return item
    return {}


_DROP_BLOCKS = re.compile(
    r"<(script|style|noscript|nav|header|footer|aside|form|figure)\b.*?</\1>", re.S | re.I
)
_PARA = re.compile(r"<p\b[^>]*>(.*?)</p>", re.S | re.I)


def body_from_html(page: str, min_chars: int = 200) -> str:
    """Fall back to the page's paragraphs when JSON-LD carries no articleBody.

    BBC Bangla, The Daily Star and several smaller outlets emit Article markup
    without the body in it, so JSON-LD alone recovered only the Prothom Alo
    items. This takes the `<p>` runs after stripping navigation and script
    blocks, and keeps the paragraphs long enough to be prose rather than
    captions, bylines or share prompts.
    """
    cleaned = _DROP_BLOCKS.sub(" ", page)
    paragraphs = []
    for raw in _PARA.findall(cleaned):
        text = html.unescape(_TAG.sub(" ", raw))
        text = re.sub(r"\s+", " ", text).strip()
        # Short fragments in a news page are captions, credits and UI chrome.
        if len(text) >= 60:
            paragraphs.append(text)
    body = " ".join(paragraphs).strip()
    return body if len(body) >= min_chars else ""


def parse_article(page: str, url: str, outlet: str) -> dict | None:
    """Extract one record from an article page.

    Returns None when there is no headline or no image: an item without both is
    useless for a multimodal dataset, and dropping it here keeps the annotation
    pool clean.
    """
    article = _json_ld_article(page)

    headline = _meta(page, "og:title") or article.get("headline")
    image = _meta(page, "og:image") or _image_from_json_ld(article)
    if not headline or not image:
        return None

    body = article.get("articleBody") or ""
    if body:
        # Prothom Alo double-escapes its body: &lt;p&gt; rather than <p>.
        body = _TAG.sub(" ", html.unescape(html.unescape(body)))
        body = re.sub(r"\s+", " ", body).strip()
    if not body:
        body = body_from_html(page)

    return {
        "outlet": outlet,
        "source_url": url,
        "title": html.unescape(str(headline)).strip(),
        "body": body,
        "summary": _meta(page, "og:description") or article.get("description") or "",
        "image_url": image,
        "published": article.get("datePublished") or _meta(page, "article:published_time") or "",
        "section": article.get("articleSection") or "",
        "body_chars": len(body),
    }


def _image_from_json_ld(article: dict) -> str | None:
    image = article.get("image")
    if isinstance(image, str):
        return image
    if isinstance(image, dict):
        return image.get("url")
    if isinstance(image, list) and image:
        first = image[0]
        return first if isinstance(first, str) else first.get("url")
    return None
