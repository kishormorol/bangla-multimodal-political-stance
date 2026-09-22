"""Scrape Bangla news headlines + images from major portals.

Collects recent political/national news from ~15 outlets into
data/raw/scraped/scraped_corpus.csv + data/raw/scraped/images/.

    python scripts/scrape_news.py              # default: collect ~1200 items
    python scripts/scrape_news.py --target 500 # stop after 500

Each outlet has its own scraper function. Articles are deduplicated by URL.
No stance labels — those come from human annotation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from urllib.parse import urljoin, urlparse

import feedparser
import requests
from bs4 import BeautifulSoup

# ── Paths ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
SCRAPE_DIR = ROOT / "data" / "raw" / "scraped"
IMAGE_DIR = SCRAPE_DIR / "images"
CSV_PATH = SCRAPE_DIR / "scraped_corpus.csv"
PROGRESS_PATH = SCRAPE_DIR / "progress.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "bn-BD,bn;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# ── Data model ───────────────────────────────────────────────────────────────

@dataclass
class Article:
    headline: str
    source_url: str
    outlet: str
    image_url: str = ""
    image_path: str = ""
    date: str = ""
    section: str = ""


# ── Helpers ──────────────────────────────────────────────────────────────────

def get(url: str, timeout: int = 20) -> requests.Response | None:
    """GET with retry."""
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
            if r.status_code == 200:
                return r
            if r.status_code == 429:
                wait = 5 * (attempt + 1)
                print(f"  429 rate-limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            if r.status_code in (403, 404):
                return None
        except requests.RequestException:
            if attempt < 2:
                time.sleep(2)
    return None


def soup(url: str) -> BeautifulSoup | None:
    r = get(url)
    if r is None:
        return None
    return BeautifulSoup(r.content, "lxml")


def download_image(url: str, article_id: str) -> str:
    """Download image, return relative path under scraped/images/."""
    if not url:
        return ""
    ext = Path(urlparse(url).path).suffix or ".jpg"
    if len(ext) > 6:
        ext = ".jpg"
    fname = f"{article_id}{ext}"
    dest = IMAGE_DIR / fname
    if dest.exists():
        return f"images/{fname}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15, stream=True)
        if r.status_code == 200 and int(r.headers.get("content-length", 999)) > 500:
            dest.write_bytes(r.content)
            return f"images/{fname}"
    except Exception:
        pass
    return ""


def article_id(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:12]


def clean_headline(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    # Remove trailing outlet name after |
    text = re.sub(r"\s*\|.*$", "", text)
    return text


# ── Scrapers ─────────────────────────────────────────────────────────────────

def scrape_prothomalo_rss() -> list[Article]:
    """Prothom Alo — RSS feed (main feed covers all sections)."""
    articles = []
    feed = feedparser.parse("https://www.prothomalo.com/feed")
    for entry in feed.entries:
        headline = clean_headline(entry.get("title", ""))
        link = entry.get("link", "")
        if not headline or not link:
            continue
        img = ""
        # Image from media_content or enclosure
        for mc in entry.get("media_content", []):
            if mc.get("url"):
                img = mc["url"]
                break
        if not img:
            for enc in entry.get("enclosures", []):
                if enc.get("href"):
                    img = enc["href"]
                    break
        # Try og:image from summary
        if not img and entry.get("summary"):
            m = re.search(r'src=["\']([^"\']+)', entry.summary)
            if m:
                img = m.group(1)
        articles.append(Article(
            headline=headline, source_url=link, outlet="Prothom Alo",
            image_url=img, date=entry.get("published", ""),
        ))
    print(f"  Prothom Alo RSS: {len(articles)} items")
    return articles


def scrape_daily_star_section(section: str, label: str) -> list[Article]:
    """The Daily Star Bangla — HTML scrape."""
    base = "https://bangla.thedailystar.net"
    articles = []
    for page in range(1, 16):
        url = f"{base}/news/{section}" if page == 1 else f"{base}/news/{section}?page={page}"
        s = soup(url)
        if not s:
            break
        # Find article links with headlines
        found = 0
        for a_tag in s.find_all("a", href=True):
            href = a_tag["href"]
            if "/news/" not in href or href.count("/") < 3:
                continue
            # Get headline text
            h = a_tag.find(["h2", "h3", "h4", "h5"])
            if not h:
                text = a_tag.get_text(strip=True)
                if len(text) < 10 or len(text) > 300:
                    continue
            else:
                text = h.get_text(strip=True)
            if not text or len(text) < 10:
                continue
            full_url = urljoin(base, href)
            # Find nearby image
            img_url = ""
            parent = a_tag.parent
            if parent:
                img = parent.find("img")
                if img:
                    img_url = img.get("data-src") or img.get("src") or ""
                    if img_url:
                        img_url = urljoin(base, img_url)
            articles.append(Article(
                headline=clean_headline(text), source_url=full_url,
                outlet="The Daily Star", image_url=img_url, section=section,
            ))
            found += 1
        if found == 0:
            break
        time.sleep(1)
    return articles


def scrape_daily_star() -> list[Article]:
    all_articles = []
    for section in ["bangladesh", "politics", "bangladesh/crime-justice"]:
        arts = scrape_daily_star_section(section, section)
        all_articles.extend(arts)
        time.sleep(1)
    # dedup
    seen = set()
    deduped = []
    for a in all_articles:
        if a.source_url not in seen:
            seen.add(a.source_url)
            deduped.append(a)
    print(f"  Daily Star: {len(deduped)} items")
    return deduped


def scrape_bdnews24() -> list[Article]:
    """bdnews24 Bangla — HTML scrape."""
    base = "https://bangla.bdnews24.com"
    articles = []
    for section in ["bangladesh", "politics"]:
        for page in range(1, 6):
            url = f"{base}/{section}" if page == 1 else f"{base}/{section}?page={page}"
            s = soup(url)
            if not s:
                break
            found = 0
            for a_tag in s.find_all("a", href=True):
                href = a_tag["href"]
                if not href.startswith(("http", "/")):
                    continue
                full_url = urljoin(base, href)
                if base not in full_url:
                    continue
                h = a_tag.find(["h2", "h3", "h4", "h5"])
                if not h:
                    continue
                text = h.get_text(strip=True)
                if not text or len(text) < 10:
                    continue
                img_url = ""
                parent = a_tag.parent
                if parent:
                    img = parent.find("img")
                    if not img:
                        img = a_tag.find("img")
                    if img:
                        img_url = img.get("data-src") or img.get("src") or ""
                        if img_url:
                            img_url = urljoin(base, img_url)
                articles.append(Article(
                    headline=clean_headline(text), source_url=full_url,
                    outlet="bdnews24", image_url=img_url, section=section,
                ))
                found += 1
            if found == 0:
                break
            time.sleep(1)
    seen = set()
    deduped = [a for a in articles if a.source_url not in seen and not seen.add(a.source_url)]
    print(f"  bdnews24: {len(deduped)} items")
    return deduped


def scrape_ntv() -> list[Article]:
    """NTV Online — HTML scrape."""
    base = "https://www.ntvbd.com"
    articles = []
    for section in ["bangladesh", "politics", "crime"]:
        for page in range(1, 6):
            url = f"{base}/{section}" if page == 1 else f"{base}/{section}?page={page}"
            s = soup(url)
            if not s:
                break
            found = 0
            for a_tag in s.find_all("a", href=True):
                href = a_tag["href"]
                if "news-" not in href:
                    continue
                full_url = urljoin(base, href)
                text = a_tag.get_text(strip=True)
                if not text or len(text) < 10 or len(text) > 300:
                    continue
                # skip nav/menu items
                if any(x in text.lower() for x in ["menu", "nav", "footer"]):
                    continue
                img_url = ""
                parent = a_tag.parent
                if parent:
                    img = parent.find("img")
                    if img:
                        img_url = img.get("data-src") or img.get("data-original") or img.get("src") or ""
                        if img_url:
                            img_url = urljoin(base, img_url)
                articles.append(Article(
                    headline=clean_headline(text), source_url=full_url,
                    outlet="NTV", image_url=img_url, section=section,
                ))
                found += 1
            if found == 0:
                break
            time.sleep(1)
    seen = set()
    deduped = [a for a in articles if a.source_url not in seen and not seen.add(a.source_url)]
    print(f"  NTV: {len(deduped)} items")
    return deduped


def scrape_bbc_bangla() -> list[Article]:
    """BBC Bangla — HTML scrape."""
    base = "https://www.bbc.com"
    articles = []
    for path in ["/bengali/topics/ckdxnwvwg9zt", "/bengali"]:
        s = soup(f"{base}{path}")
        if not s:
            continue
        for a_tag in s.find_all("a", href=True):
            href = a_tag["href"]
            if "/bengali/" not in href:
                continue
            if "/articles/" not in href and "/news/" not in href:
                continue
            full_url = urljoin(base, href)
            text = a_tag.get_text(strip=True)
            if not text or len(text) < 10 or len(text) > 300:
                continue
            img_url = ""
            parent = a_tag.parent
            for p in [a_tag, parent, parent.parent if parent else None]:
                if p:
                    img = p.find("img")
                    if img:
                        img_url = img.get("src") or ""
                        if img_url:
                            img_url = urljoin(base, img_url)
                        break
            articles.append(Article(
                headline=clean_headline(text), source_url=full_url,
                outlet="BBC Bangla", image_url=img_url,
            ))
    seen = set()
    deduped = [a for a in articles if a.source_url not in seen and not seen.add(a.source_url)]
    print(f"  BBC Bangla: {len(deduped)} items")
    return deduped


def scrape_dw_bangla() -> list[Article]:
    """DW Bangla — HTML scrape."""
    base = "https://www.dw.com"
    articles = []
    for path in ["/bn/latest-news/s-11929", "/bn/বাংলাদেশ/s-11903"]:
        s = soup(f"{base}{path}")
        if not s:
            continue
        for a_tag in s.find_all("a", href=True):
            href = a_tag["href"]
            if "/bn/" not in href or "/a-" not in href:
                continue
            full_url = urljoin(base, href)
            text = a_tag.get_text(strip=True)
            if not text or len(text) < 10 or len(text) > 300:
                continue
            img_url = ""
            parent = a_tag.parent
            if parent:
                img = parent.find("img")
                if img:
                    img_url = img.get("src") or img.get("data-src") or ""
                    if img_url:
                        img_url = urljoin(base, img_url)
            articles.append(Article(
                headline=clean_headline(text), source_url=full_url,
                outlet="DW Bangla", image_url=img_url,
            ))
    seen = set()
    deduped = [a for a in articles if a.source_url not in seen and not seen.add(a.source_url)]
    print(f"  DW Bangla: {len(deduped)} items")
    return deduped


def _scrape_generic_html(base_url: str, outlet: str, sections: list[str],
                          pages: int = 5, link_pattern: str = "") -> list[Article]:
    """Generic scraper for sites with standard article-link structure."""
    articles = []
    for section in sections:
        for page in range(1, pages + 1):
            if page == 1:
                url = f"{base_url}/{section}" if section else base_url
            else:
                url = f"{base_url}/{section}?page={page}" if section else f"{base_url}?page={page}"
            s = soup(url)
            if not s:
                break
            found = 0
            for a_tag in s.find_all("a", href=True):
                href = a_tag["href"]
                full_url = urljoin(base_url, href)
                if base_url.split("//")[1].split("/")[0] not in full_url:
                    continue
                if link_pattern and link_pattern not in href:
                    continue
                # Need some heuristic for headline links
                h = a_tag.find(["h1", "h2", "h3", "h4", "h5"])
                text = h.get_text(strip=True) if h else a_tag.get_text(strip=True)
                if not text or len(text) < 15 or len(text) > 300:
                    continue
                # Skip if no Bangla chars
                if not re.search(r"[\u0980-\u09FF]", text):
                    continue
                img_url = ""
                for parent in [a_tag, a_tag.parent, a_tag.parent.parent if a_tag.parent else None]:
                    if parent:
                        img = parent.find("img")
                        if img:
                            img_url = img.get("data-src") or img.get("data-original") or img.get("src") or ""
                            if img_url and not img_url.startswith("data:"):
                                img_url = urljoin(base_url, img_url)
                                break
                            img_url = ""
                articles.append(Article(
                    headline=clean_headline(text), source_url=full_url,
                    outlet=outlet, image_url=img_url, section=section,
                ))
                found += 1
            if found == 0:
                break
            time.sleep(1.5)
    seen = set()
    deduped = [a for a in articles if a.source_url not in seen and not seen.add(a.source_url)]
    print(f"  {outlet}: {len(deduped)} items")
    return deduped


def scrape_jugantor() -> list[Article]:
    return _scrape_generic_html(
        "https://www.jugantor.com", "Jugantor",
        ["national", "politics", "city"],
        link_pattern="/",
    )


def scrape_samakal() -> list[Article]:
    return _scrape_generic_html(
        "https://samakal.com", "Samakal",
        ["national", "politics"],
    )


def scrape_kaler_kantho() -> list[Article]:
    return _scrape_generic_html(
        "https://www.kalerkantho.com", "Kaler Kantho",
        ["online/national", "online/politics"],
    )


def scrape_bangla_tribune() -> list[Article]:
    return _scrape_generic_html(
        "https://www.banglatribune.com", "Bangla Tribune",
        ["national", "politics"],
    )


def scrape_dhaka_post() -> list[Article]:
    return _scrape_generic_html(
        "https://www.dhakapost.com", "Dhaka Post",
        ["national", "politics"],
    )


def scrape_ittefaq() -> list[Article]:
    return _scrape_generic_html(
        "https://www.ittefaq.com.bd", "Daily Ittefaq",
        ["national", "politics"],
    )


def scrape_manab_zamin() -> list[Article]:
    return _scrape_generic_html(
        "https://mzamin.com", "Manab Zamin",
        ["national", "politics"],
    )


def scrape_jago_news() -> list[Article]:
    return _scrape_generic_html(
        "https://www.jagonews24.com", "Jago News",
        ["politics", "national"],
    )


def scrape_naya_diganta() -> list[Article]:
    return _scrape_generic_html(
        "https://www.dailynayadiganta.com", "Naya Diganta",
        ["politics", "country-news"],
    )


def scrape_bangladesh_pratidin() -> list[Article]:
    return _scrape_generic_html(
        "https://www.bd-pratidin.com", "Bangladesh Pratidin",
        ["national", "politics"],
    )


def scrape_channel_i() -> list[Article]:
    return _scrape_generic_html(
        "https://www.channelionline.com", "Channel i",
        ["national", "politics"],
    )


def scrape_somoy_news() -> list[Article]:
    return _scrape_generic_html(
        "https://www.somoynews.tv", "Somoy News",
        ["national", "politics"],
    )


def scrape_google_news_rss() -> list[Article]:
    """Google News RSS — Bangla political news queries. Bypasses 403 sites."""
    queries = [
        "বাংলাদেশ রাজনীতি",
        "বাংলাদেশ সরকার",
        "বাংলাদেশ বিরোধীদল",
        "বাংলাদেশ নির্বাচন",
        "বাংলাদেশ আন্দোলন",
        "বাংলাদেশ সংসদ",
        "বাংলাদেশ মন্ত্রী",
        "বাংলাদেশ আইন",
        "বাংলাদেশ পুলিশ",
        "বাংলাদেশ বিচার",
        "ঢাকা রাজনীতি",
        "বাংলাদেশ অর্থনীতি সরকার",
        "বাংলাদেশ দুর্নীতি",
        "বাংলাদেশ প্রতিবাদ",
        "বাংলাদেশ মানবাধিকার",
        "বাংলাদেশ সংস্কার",
        "জাতীয় সংসদ বাংলাদেশ",
        "বাংলাদেশ পররাষ্ট্র",
        "বাংলাদেশ উন্নয়ন সরকার",
        "বাংলাদেশ ছাত্র আন্দোলন",
    ]
    articles = []
    base = "https://news.google.com/rss/search?q={}&hl=bn&gl=BD&ceid=BD:bn"
    for q in queries:
        url = base.format(requests.utils.quote(q))
        feed = feedparser.parse(url)
        for entry in feed.entries:
            title = clean_headline(entry.get("title", ""))
            link = entry.get("link", "")
            if not title or not link:
                continue
            # Skip non-Bangla
            if not re.search(r"[\u0980-\u09FF]", title):
                continue
            # Extract source from title (Google News appends " - Source")
            source = entry.get("source", {}).get("title", "")
            if not source:
                m = re.search(r"\s*[-–—]\s*([^-–—]+)$", title)
                if m:
                    source = m.group(1).strip()
                    title = title[:m.start()].strip()
            articles.append(Article(
                headline=title, source_url=link, outlet=source or "Unknown",
                date=entry.get("published", ""),
            ))
        time.sleep(1)
    seen = set()
    deduped = [a for a in articles if a.source_url not in seen and not seen.add(a.source_url)]
    print(f"  Google News RSS: {len(deduped)} items")
    return deduped


# Also try to get article-page images for items with missing images
def fetch_og_image(article: Article) -> str:
    """Try to get og:image from the article page itself."""
    if article.image_url:
        return article.image_url
    s = soup(article.source_url)
    if not s:
        return ""
    og = s.find("meta", property="og:image")
    if og and og.get("content"):
        return og["content"]
    return ""


# ── Main pipeline ────────────────────────────────────────────────────────────

SCRAPERS = [
    scrape_prothomalo_rss,
    scrape_daily_star,
    scrape_bdnews24,
    scrape_ntv,
    scrape_bbc_bangla,
    scrape_dw_bangla,
    scrape_jugantor,
    scrape_samakal,
    scrape_kaler_kantho,
    scrape_bangla_tribune,
    scrape_dhaka_post,
    scrape_ittefaq,
    scrape_manab_zamin,
    scrape_jago_news,
    scrape_naya_diganta,
    scrape_bangladesh_pratidin,
    scrape_channel_i,
    scrape_somoy_news,
    scrape_google_news_rss,   # catches items from 403-blocked sites
]


def load_existing() -> set[str]:
    """Load URLs already scraped."""
    if not CSV_PATH.exists():
        return set()
    urls = set()
    with open(CSV_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            urls.add(row.get("source_url", ""))
    return urls


def save_articles(articles: list[Article]) -> None:
    """Append articles to CSV."""
    file_exists = CSV_PATH.exists()
    fieldnames = ["item_id", "headline", "source_url", "outlet", "image_url",
                   "image_path", "date", "section"]
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for a in articles:
            writer.writerow({
                "item_id": article_id(a.source_url),
                "headline": a.headline,
                "source_url": a.source_url,
                "outlet": a.outlet,
                "image_url": a.image_url,
                "image_path": a.image_path,
                "date": a.date,
                "section": a.section,
            })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=1200,
                        help="Target number of articles")
    parser.add_argument("--skip-images", action="store_true",
                        help="Skip image downloads (collect metadata only)")
    parser.add_argument("--og-fallback", action="store_true",
                        help="Fetch og:image for items missing images (slow)")
    args = parser.parse_args()

    SCRAPE_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    existing_urls = load_existing()
    print(f"Already scraped: {len(existing_urls)} articles")

    all_articles: list[Article] = []
    for scraper in SCRAPERS:
        name = scraper.__name__
        print(f"\nRunning {name}...")
        try:
            batch = scraper()
            # Dedup against existing
            new = [a for a in batch if a.source_url not in existing_urls]
            all_articles.extend(new)
            for a in new:
                existing_urls.add(a.source_url)
            print(f"  -> {len(new)} new (total so far: {len(all_articles)})")
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()

        if len(all_articles) + len(load_existing()) >= args.target:
            print(f"\nReached target ({args.target}), stopping.")
            break

    # Deduplicate by headline similarity (exact match)
    seen_headlines = set()
    deduped = []
    for a in all_articles:
        norm = re.sub(r"\s+", " ", a.headline.strip().lower())
        if norm not in seen_headlines:
            seen_headlines.add(norm)
            deduped.append(a)
    print(f"\nAfter dedup: {len(deduped)} new articles")

    # Optionally fetch og:image for items without images
    if args.og_fallback:
        missing = [a for a in deduped if not a.image_url]
        print(f"\nFetching og:image for {len(missing)} items without images...")
        for i, a in enumerate(missing):
            a.image_url = fetch_og_image(a)
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(missing)}")
            time.sleep(0.5)

    # Download images
    if not args.skip_images:
        with_img = [a for a in deduped if a.image_url]
        print(f"\nDownloading images for {len(with_img)} articles...")
        for i, a in enumerate(with_img):
            a.image_path = download_image(a.image_url, article_id(a.source_url))
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(with_img)} images downloaded")
            time.sleep(0.2)

    # Save
    save_articles(deduped)

    total = len(load_existing())
    with_images = sum(1 for a in deduped if a.image_path)
    print(f"\n{'='*50}")
    print(f"Total articles in CSV: {total}")
    print(f"New articles this run: {len(deduped)}")
    print(f"New with images: {with_images}")
    print(f"Output: {CSV_PATH}")
    print(f"Images: {IMAGE_DIR}")

    if total < args.target:
        print(f"\nBelow target ({args.target}). Run again — some sites may "
              "return different pages, or use --og-fallback for more images.")


if __name__ == "__main__":
    main()
