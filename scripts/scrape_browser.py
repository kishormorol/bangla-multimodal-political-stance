"""Browser-based scraper for Cloudflare-protected Bangla news sites.

Uses Playwright (Chromium) to render pages that block plain requests.
Appends to data/raw/scraped/scraped_corpus.csv (same format as scrape_news.py).

    python scripts/scrape_browser.py
"""

from __future__ import annotations

import csv
import hashlib
import re
import time
import traceback
from pathlib import Path
from urllib.parse import urljoin

import requests
from playwright.sync_api import sync_playwright, Page, Browser

ROOT = Path(__file__).resolve().parent.parent
SCRAPE_DIR = ROOT / "data" / "raw" / "scraped"
IMAGE_DIR = SCRAPE_DIR / "images"
CSV_PATH = SCRAPE_DIR / "scraped_corpus.csv"

FIELDNAMES = ["item_id", "headline", "source_url", "outlet", "image_url",
              "image_path", "date", "section"]

DL_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}


def article_id(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:12]


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_existing_urls() -> set[str]:
    if not CSV_PATH.exists():
        return set()
    with open(CSV_PATH, encoding="utf-8") as f:
        return {r.get("source_url", "") for r in csv.DictReader(f)}


def download_image(url: str, aid: str) -> str:
    if not url:
        return ""
    ext = ".jpg"
    m = re.search(r"\.(jpg|jpeg|png|webp)", url.lower())
    if m:
        ext = "." + m.group(1)
    fname = f"{aid}{ext}"
    dest = IMAGE_DIR / fname
    if dest.exists():
        return f"images/{fname}"
    try:
        r = requests.get(url, headers=DL_HEADERS, timeout=15)
        if r.status_code == 200 and len(r.content) > 500:
            dest.write_bytes(r.content)
            return f"images/{fname}"
    except Exception:
        pass
    return ""


def extract_articles(page: Page, base_url: str, outlet: str, section: str,
                     existing: set[str]) -> list[dict]:
    """Extract articles from the current page."""
    articles = []
    # Get all links with Bangla text
    links = page.evaluate("""() => {
        const results = [];
        document.querySelectorAll('a[href]').forEach(a => {
            const text = a.innerText.trim();
            if (text.length < 15 || text.length > 400) return;
            // Must contain Bangla characters
            if (!/[\u0980-\u09FF]/.test(text)) return;
            const href = a.href;
            // Find nearby image
            let imgUrl = '';
            const img = a.querySelector('img') ||
                        (a.parentElement && a.parentElement.querySelector('img')) ||
                        (a.closest('article, .card, .news-item, .story, div') &&
                         a.closest('article, .card, .news-item, .story, div').querySelector('img'));
            if (img) {
                imgUrl = img.src || img.dataset.src || img.dataset.original || img.dataset.lazySrc || '';
            }
            results.push({text, href, imgUrl});
        });
        return results;
    }""")

    for item in links:
        url = item["href"]
        if url in existing:
            continue
        headline = clean(item["text"])
        # Remove outlet name suffix
        headline = re.sub(r"\s*[-–—|]\s*[^-–—|]+$", "", headline)
        if len(headline) < 10:
            continue

        img_url = item.get("imgUrl", "")
        if img_url and img_url.startswith("data:"):
            img_url = ""
        if img_url and not img_url.startswith("http"):
            img_url = urljoin(base_url, img_url)

        aid = article_id(url)
        img_path = download_image(img_url, aid)

        articles.append({
            "item_id": aid,
            "headline": headline,
            "source_url": url,
            "outlet": outlet,
            "image_url": img_url,
            "image_path": img_path,
            "date": "",
            "section": section,
        })
        existing.add(url)

    return articles


def save_articles(articles: list[dict]):
    file_exists = CSV_PATH.exists() and CSV_PATH.stat().st_size > 0
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerows(articles)


# ── Site configs ─────────────────────────────────────────────────────────────

SITES = [
    # (outlet, base_url, [section_paths], pages_per_section)
    ("Jugantor", "https://www.jugantor.com", [
        "national", "politics", "city", "economics",
        "international", "country-news",
    ], 8),
    ("Kaler Kantho", "https://www.kalerkantho.com", [
        "online/national", "online/politics", "online/country-news",
        "online/dhaka", "online/crime",
    ], 8),
    ("Jago News", "https://www.jagonews24.com", [
        "politics", "national", "economics", "crime",
    ], 8),
    ("Daily Ittefaq", "https://www.ittefaq.com.bd", [
        "national", "politics", "city",
    ], 8),
    ("Bangla Tribune", "https://www.banglatribune.com", [
        "national", "politics", "country",
    ], 8),
    ("Manab Zamin", "https://mzamin.com", [
        "national", "politics", "city",
    ], 5),
    ("Somoy News", "https://www.somoynews.tv", [
        "national", "politics",
    ], 5),
    ("Channel i", "https://www.channelionline.com", [
        "national", "politics",
    ], 5),
    ("Naya Diganta", "https://www.dailynayadiganta.com", [
        "politics", "country-news", "first-page",
    ], 5),
    ("Amader Shomoy", "https://www.dainikamadershomoy.com", [
        "national", "politics",
    ], 5),
    ("Bhorer Kagoj", "https://www.bhorerkagoj.net", [
        "national", "politics",
    ], 5),
    ("Inqilab", "https://www.dailyinqilab.com", [
        "national", "politics",
    ], 5),
    ("Janakantha", "https://www.dailyjanakantha.com", [
        "national", "politics",
    ], 5),
    ("Bangladesh Journal", "https://www.bangladeshjournal.net", [
        "national", "politics",
    ], 5),
    ("Risingbd", "https://www.risingbd.com", [
        "politics", "country",
    ], 5),
]


def scrape_site(browser: Browser, outlet: str, base_url: str,
                sections: list[str], pages: int, existing: set[str]) -> list[dict]:
    """Scrape one site using a browser context."""
    context = browser.new_context(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
        locale="bn-BD",
    )
    page = context.new_page()
    all_articles = []

    for section in sections:
        for pg in range(1, pages + 1):
            if pg == 1:
                url = f"{base_url}/{section}"
            else:
                # Try common pagination patterns
                url = f"{base_url}/{section}?page={pg}"

            try:
                page.goto(url, timeout=20000, wait_until="domcontentloaded")
                # Wait a bit for JS rendering
                page.wait_for_timeout(2000)
            except Exception:
                break

            articles = extract_articles(page, base_url, outlet, section, existing)
            all_articles.extend(articles)

            if len(articles) == 0 and pg > 1:
                break
            time.sleep(1)

    context.close()

    # Dedup by URL
    seen = set()
    deduped = []
    for a in all_articles:
        if a["source_url"] not in seen:
            seen.add(a["source_url"])
            deduped.append(a)

    return deduped


def fetch_og_images_browser(browser: Browser, articles: list[dict], existing: set[str]):
    """For articles without images, visit the page and grab og:image."""
    no_img = [a for a in articles if not a["image_path"]]
    if not no_img:
        return

    context = browser.new_context(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    )
    page = context.new_page()
    fetched = 0

    for i, a in enumerate(no_img):
        try:
            page.goto(a["source_url"], timeout=15000, wait_until="domcontentloaded")
            page.wait_for_timeout(1500)

            og_img = page.evaluate("""() => {
                const meta = document.querySelector('meta[property="og:image"]');
                return meta ? meta.content : '';
            }""")

            if og_img and not og_img.startswith("data:"):
                a["image_url"] = og_img
                a["image_path"] = download_image(og_img, a["item_id"])
                if a["image_path"]:
                    fetched += 1
        except Exception:
            pass

        if (i + 1) % 50 == 0:
            print(f"    og:image {i+1}/{len(no_img)}: {fetched} fetched")
        time.sleep(0.3)

    context.close()
    print(f"    og:image done: {fetched}/{len(no_img)} fetched")


def main():
    SCRAPE_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    existing = load_existing_urls()
    print(f"Already collected: {len(existing)} articles\n")

    total_new = 0

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)

        for outlet, base_url, sections, pages in SITES:
            print(f"Scraping {outlet} ({base_url})...")
            try:
                articles = scrape_site(browser, outlet, base_url, sections, pages, existing)
                print(f"  -> {len(articles)} new articles")

                if articles:
                    # Fetch og:image for items without images
                    no_img_count = sum(1 for a in articles if not a["image_path"])
                    if no_img_count > 0:
                        print(f"  -> Fetching og:image for {no_img_count} items...")
                        fetch_og_images_browser(browser, articles, existing)

                    save_articles(articles)
                    total_new += len(articles)

            except Exception as e:
                print(f"  FAILED: {e}")
                traceback.print_exc()

            print()

        browser.close()

    final_count = len(load_existing_urls())
    images_count = len(list(IMAGE_DIR.glob("*")))
    print(f"{'='*50}")
    print(f"New articles this run: {total_new}")
    print(f"Total articles in CSV: {final_count}")
    print(f"Total images on disk: {images_count}")


if __name__ == "__main__":
    main()
