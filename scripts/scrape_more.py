"""Deeper scraping from sites that work + expanded Google News queries.

Appends to data/raw/scraped/scraped_corpus.csv.
"""

from __future__ import annotations

import csv
import hashlib
import re
import time
import traceback
from pathlib import Path
from urllib.parse import urljoin

import feedparser
import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
SCRAPE_DIR = ROOT / "data" / "raw" / "scraped"
IMAGE_DIR = SCRAPE_DIR / "images"
CSV_PATH = SCRAPE_DIR / "scraped_corpus.csv"
FIELDNAMES = ["item_id", "headline", "source_url", "outlet", "image_url",
              "image_path", "date", "section"]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
}


def aid(url): return hashlib.md5(url.encode()).hexdigest()[:12]
def clean(t): return re.sub(r"\s+", " ", t).strip()


def load_existing():
    if not CSV_PATH.exists():
        return set()
    with open(CSV_PATH, encoding="utf-8") as f:
        return {r["source_url"] for r in csv.DictReader(f)}


def save(articles):
    exists = CSV_PATH.exists() and CSV_PATH.stat().st_size > 0
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            w.writeheader()
        w.writerows(articles)


def dl_img(url, article_id):
    if not url: return ""
    ext = ".jpg"
    m = re.search(r"\.(jpg|jpeg|png|webp)", url.lower())
    if m: ext = "." + m.group(1)
    fname = f"{article_id}{ext}"
    dest = IMAGE_DIR / fname
    if dest.exists(): return f"images/{fname}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code == 200 and len(r.content) > 500:
            dest.write_bytes(r.content)
            return f"images/{fname}"
    except Exception: pass
    return ""


def google_news_rss(existing):
    """Expanded Google News RSS with many more query terms."""
    queries = [
        # Politics
        "বাংলাদেশ রাজনীতি", "বাংলাদেশ সরকার", "বাংলাদেশ বিরোধীদল",
        "বাংলাদেশ নির্বাচন", "বাংলাদেশ আন্দোলন", "বাংলাদেশ সংসদ",
        "বাংলাদেশ মন্ত্রী", "বাংলাদেশ আইন", "বাংলাদেশ বিচার",
        "বাংলাদেশ দুর্নীতি", "বাংলাদেশ প্রতিবাদ", "বাংলাদেশ মানবাধিকার",
        "বাংলাদেশ সংস্কার", "বাংলাদেশ পররাষ্ট্র", "বাংলাদেশ ছাত্র আন্দোলন",
        # Government
        "প্রধানমন্ত্রী বাংলাদেশ", "অন্তর্বর্তী সরকার বাংলাদেশ",
        "উপদেষ্টা পরিষদ বাংলাদেশ", "ড. ইউনূস সরকার",
        "বাংলাদেশ মন্ত্রণালয়", "বাংলাদেশ প্রশাসন",
        # Economy & policy
        "বাংলাদেশ অর্থনীতি", "বাংলাদেশ বাজেট", "বাংলাদেশ জ্বালানি",
        "বাংলাদেশ ব্যাংক", "বাংলাদেশ বিদ্যুৎ", "বাংলাদেশ পোশাক শিল্প",
        # Security & law
        "বাংলাদেশ পুলিশ", "বাংলাদেশ র‌্যাব", "বাংলাদেশ সেনাবাহিনী",
        "বাংলাদেশ ট্রাইব্যুনাল", "বাংলাদেশ গ্রেপ্তার",
        # International
        "বাংলাদেশ ভারত সম্পর্ক", "বাংলাদেশ চীন", "বাংলাদেশ জাতিসংঘ",
        "রোহিঙ্গা বাংলাদেশ", "বাংলাদেশ মিয়ানমার",
        # Social
        "বাংলাদেশ শিক্ষা সংস্কার", "বাংলাদেশ স্বাস্থ্য",
        "বাংলাদেশ পরিবেশ", "বাংলাদেশ নারী অধিকার",
        # Party-specific
        "আওয়ামী লীগ", "বিএনপি বাংলাদেশ", "জামায়াত বাংলাদেশ",
        "জাতীয় পার্টি এরশাদ",
        # Events
        "হরতাল বাংলাদেশ", "মিছিল ঢাকা", "ধর্মঘট বাংলাদেশ",
        "গণতন্ত্র বাংলাদেশ", "সংবিধান সংশোধন বাংলাদেশ",
        # Quota/reform movement
        "কোটা আন্দোলন বাংলাদেশ", "সংস্কার কমিশন বাংলাদেশ",
        "গণঅভ্যুত্থান বাংলাদেশ",
    ]
    articles = []
    base = "https://news.google.com/rss/search?q={}&hl=bn&gl=BD&ceid=BD:bn"

    for i, q in enumerate(queries):
        url = base.format(requests.utils.quote(q))
        feed = feedparser.parse(url)
        new_in_query = 0
        for entry in feed.entries:
            title = clean(entry.get("title", ""))
            link = entry.get("link", "")
            if not title or not link or link in existing:
                continue
            if not re.search(r"[\u0980-\u09FF]", title):
                continue
            source = entry.get("source", {}).get("title", "")
            if not source:
                m = re.search(r"\s*[-–—]\s*([^-–—]+)$", title)
                if m:
                    source = m.group(1).strip()
                    title = title[:m.start()].strip()
            existing.add(link)
            articles.append({
                "item_id": aid(link),
                "headline": title,
                "source_url": link,
                "outlet": source or "Unknown",
                "image_url": "",
                "image_path": "",
                "date": entry.get("published", ""),
                "section": "politics",
            })
            new_in_query += 1
        if (i + 1) % 10 == 0:
            print(f"  Google News: {i+1}/{len(queries)} queries, {len(articles)} new so far")
        time.sleep(0.8)

    print(f"  Google News RSS total: {len(articles)} new items")
    return articles


def daily_star_deep(existing):
    """Scrape Daily Star with more sections and deeper pagination."""
    base = "https://bangla.thedailystar.net"
    sections = [
        "bangladesh", "politics", "bangladesh/crime-justice",
        "bangladesh/government-politics", "bangladesh/accident-fire",
        "opinion", "opinion/editorial", "opinion/columns",
        "international", "business",
    ]
    articles = []
    for section in sections:
        for page in range(1, 25):
            url = f"{base}/news/{section}" if page == 1 else f"{base}/news/{section}?page={page}"
            try:
                r = requests.get(url, headers=HEADERS, timeout=15)
                if r.status_code != 200:
                    break
            except Exception:
                break
            s = BeautifulSoup(r.content, "lxml")
            found = 0
            for a_tag in s.find_all("a", href=True):
                href = a_tag["href"]
                if "/news/" not in href or href.count("/") < 3:
                    continue
                full_url = urljoin(base, href)
                if full_url in existing:
                    continue
                h = a_tag.find(["h2", "h3", "h4", "h5"])
                text = h.get_text(strip=True) if h else a_tag.get_text(strip=True)
                if not text or len(text) < 10 or len(text) > 300:
                    continue
                if not re.search(r"[\u0980-\u09FF]", text):
                    continue
                img_url = ""
                parent = a_tag.parent
                if parent:
                    img = parent.find("img")
                    if img:
                        img_url = img.get("data-src") or img.get("src") or ""
                        if img_url:
                            img_url = urljoin(base, img_url)
                existing.add(full_url)
                a_id = aid(full_url)
                articles.append({
                    "item_id": a_id,
                    "headline": clean(text),
                    "source_url": full_url,
                    "outlet": "The Daily Star",
                    "image_url": img_url,
                    "image_path": dl_img(img_url, a_id),
                    "date": "",
                    "section": section,
                })
                found += 1
            if found == 0 and page > 2:
                break
            time.sleep(0.8)
    print(f"  Daily Star deep: {len(articles)} new items")
    return articles


def risingbd_jago_browser(existing):
    """Use browser for Risingbd and Jago News with deeper pagination."""
    articles = []
    sites = [
        ("Risingbd", "https://www.risingbd.com", ["politics", "country", "crime", "economics"], 15),
        ("Jago News", "https://www.jagonews24.com", ["politics", "national", "economics", "crime", "international"], 12),
        ("Channel i", "https://www.channelionline.com", ["national", "politics", "economics"], 10),
        ("Bhorer Kagoj", "https://www.bhorerkagoj.net", ["national", "politics", "economics"], 10),
    ]

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)

        for outlet, base_url, sections, max_pages in sites:
            ctx = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
            )
            page = ctx.new_page()
            site_count = 0

            for section in sections:
                for pg in range(1, max_pages + 1):
                    url = f"{base_url}/{section}" if pg == 1 else f"{base_url}/{section}?page={pg}"
                    try:
                        page.goto(url, timeout=20000, wait_until="domcontentloaded")
                        page.wait_for_timeout(2000)
                    except Exception:
                        break

                    links = page.evaluate("""() => {
                        const results = [];
                        document.querySelectorAll('a[href]').forEach(a => {
                            const text = a.innerText.trim();
                            if (text.length < 15 || text.length > 400) return;
                            if (!/[\u0980-\u09FF]/.test(text)) return;
                            let imgUrl = '';
                            const img = a.querySelector('img') ||
                                (a.parentElement && a.parentElement.querySelector('img'));
                            if (img) imgUrl = img.src || img.dataset.src || '';
                            results.push({text, href: a.href, imgUrl});
                        });
                        return results;
                    }""")

                    found = 0
                    for item in links:
                        link = item["href"]
                        if link in existing:
                            continue
                        headline = clean(item["text"])
                        headline = re.sub(r"\s*[-–—|]\s*[^-–—|]+$", "", headline)
                        if len(headline) < 10:
                            continue

                        img_url = item.get("imgUrl", "")
                        if img_url and img_url.startswith("data:"):
                            img_url = ""

                        existing.add(link)
                        a_id = aid(link)
                        articles.append({
                            "item_id": a_id,
                            "headline": headline,
                            "source_url": link,
                            "outlet": outlet,
                            "image_url": img_url,
                            "image_path": dl_img(img_url, a_id),
                            "date": "",
                            "section": section,
                        })
                        found += 1
                        site_count += 1

                    if found == 0 and pg > 2:
                        break
                    time.sleep(1)

            ctx.close()
            print(f"  {outlet}: {site_count} new items")

        browser.close()

    return articles


def fetch_og_images_batch(articles):
    """Fetch og:image for items without images using requests."""
    no_img = [a for a in articles if not a["image_path"]]
    if not no_img:
        return
    print(f"  Fetching og:image for {len(no_img)} items...")
    fetched = 0
    for i, a in enumerate(no_img):
        try:
            r = requests.get(a["source_url"], headers=HEADERS, timeout=10, allow_redirects=True)
            if r.status_code != 200:
                continue
            s = BeautifulSoup(r.content, "lxml")
            og = s.find("meta", property="og:image")
            if og and og.get("content"):
                img_url = og["content"]
                a["image_url"] = img_url
                a["image_path"] = dl_img(img_url, a["item_id"])
                if a["image_path"]:
                    fetched += 1
        except Exception:
            pass
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(no_img)}: {fetched} images fetched")
        time.sleep(0.3)
    print(f"    og:image done: {fetched}/{len(no_img)}")


def main():
    SCRAPE_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    existing = load_existing()
    print(f"Already collected: {len(existing)} articles\n")

    all_new = []

    # 1. Deep Daily Star
    print("Phase 1: Daily Star deep crawl...")
    ds = daily_star_deep(existing)
    all_new.extend(ds)

    # 2. Browser-based sites
    print("\nPhase 2: Browser-based sites...")
    br = risingbd_jago_browser(existing)
    all_new.extend(br)

    # 3. Google News expanded
    print("\nPhase 3: Google News RSS expanded queries...")
    gn = google_news_rss(existing)
    all_new.extend(gn)

    # 4. Fetch og:image for items without images
    print(f"\nPhase 4: Fetching images for {sum(1 for a in all_new if not a['image_path'])} items...")
    fetch_og_images_batch(all_new)

    # Dedup by headline
    seen = set()
    deduped = []
    for a in all_new:
        norm = re.sub(r"\s+", " ", a["headline"].strip().lower())
        if norm not in seen:
            seen.add(norm)
            deduped.append(a)

    save(deduped)

    total = len(load_existing())
    imgs = len(list(IMAGE_DIR.glob("*")))
    print(f"\n{'='*50}")
    print(f"New this run: {len(deduped)}")
    print(f"Total articles: {total}")
    print(f"Total images: {imgs}")


if __name__ == "__main__":
    main()
