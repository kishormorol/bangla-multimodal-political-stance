"""Push toward 10k articles. More Google News queries + deeper site crawls.

Appends to data/raw/scraped/scraped_corpus.csv.
"""

from __future__ import annotations

import csv
import hashlib
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import feedparser
import requests
from bs4 import BeautifulSoup

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
    if not CSV_PATH.exists(): return set(), set()
    urls, headlines = set(), set()
    with open(CSV_PATH, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            urls.add(r.get("source_url", ""))
            headlines.add(re.sub(r"\s+", " ", r.get("headline", "").strip().lower()))
    return urls, headlines

def save(articles):
    exists = CSV_PATH.exists() and CSV_PATH.stat().st_size > 0
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists: w.writeheader()
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

def fetch_og(url, a_id):
    """Fetch og:image from article page."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=10, allow_redirects=True)
        if r.status_code != 200: return "", ""
        s = BeautifulSoup(r.content, "lxml")
        og = s.find("meta", property="og:image")
        if og and og.get("content"):
            img_url = og["content"]
            return img_url, dl_img(img_url, a_id)
    except Exception: pass
    return "", ""


# ── Google News with extensive queries ───────────────────────────────────────

QUERIES = [
    # Governance & state
    "বাংলাদেশ সরকার নীতি", "বাংলাদেশ মন্ত্রিসভা", "বাংলাদেশ প্রশাসন সংস্কার",
    "বাংলাদেশ স্থানীয় সরকার", "বাংলাদেশ উপজেলা নির্বাচন", "বাংলাদেশ ইউনিয়ন পরিষদ",
    "বাংলাদেশ সিটি কর্পোরেশন", "বাংলাদেশ সচিবালয়", "বাংলাদেশ মন্ত্রণালয় সিদ্ধান্ত",
    # Parliament & law
    "জাতীয় সংসদ অধিবেশন", "বাংলাদেশ আইন প্রণয়ন", "বাংলাদেশ সংবিধান",
    "বাংলাদেশ অধ্যাদেশ", "বাংলাদেশ আদালত রায়", "বাংলাদেশ হাইকোর্ট",
    "বাংলাদেশ সুপ্রিম কোর্ট", "বাংলাদেশ বিচার বিভাগ সংস্কার",
    # Political parties
    "আওয়ামী লীগ নিষিদ্ধ", "বিএনপি সমাবেশ", "জামায়াতে ইসলামী বাংলাদেশ",
    "জাতীয় পার্টি রওশন", "গণতন্ত্র মঞ্চ বাংলাদেশ", "বাম দল বাংলাদেশ",
    "ছাত্রদল বাংলাদেশ", "ছাত্রলীগ বাংলাদেশ", "জাসদ বাংলাদেশ",
    # Economy
    "বাংলাদেশ মূল্যস্ফীতি", "বাংলাদেশ ডলার সংকট", "বাংলাদেশ রিজার্ভ",
    "বাংলাদেশ রপ্তানি আয়", "বাংলাদেশ প্রবাসী রেমিট্যান্স", "বাংলাদেশ শেয়ার বাজার",
    "বাংলাদেশ করনীতি", "বাংলাদেশ ভ্যাট", "বাংলাদেশ জ্বালানি দাম",
    "বাংলাদেশ বিদ্যুৎ সংকট", "বাংলাদেশ গ্যাস সংকট",
    # Security & crime
    "বাংলাদেশ সন্ত্রাস", "বাংলাদেশ জঙ্গিবাদ", "বাংলাদেশ গুম",
    "বাংলাদেশ বিচারবহির্ভূত হত্যা", "বাংলাদেশ পুলিশ সংস্কার",
    "বাংলাদেশ র্যাব সংস্কার", "বাংলাদেশ কারাগার",
    # International relations
    "বাংলাদেশ ভারত সীমান্ত", "বাংলাদেশ চীন বিনিয়োগ", "বাংলাদেশ আমেরিকা",
    "বাংলাদেশ রাশিয়া", "বাংলাদেশ জাপান", "বাংলাদেশ সৌদি আরব",
    "রোহিঙ্গা প্রত্যাবাসন", "বাংলাদেশ মিয়ানমার সীমান্ত সংঘর্ষ",
    "তিস্তা চুক্তি বাংলাদেশ", "বাংলাদেশ বঙ্গোপসাগর",
    # Social issues
    "বাংলাদেশ শিক্ষা সংকট", "বাংলাদেশ বিশ্ববিদ্যালয়", "বাংলাদেশ চিকিৎসা",
    "বাংলাদেশ দারিদ্র্য", "বাংলাদেশ বেকারত্ব", "বাংলাদেশ নারী নির্যাতন",
    "বাংলাদেশ শিশু শ্রম", "বাংলাদেশ পরিবেশ দূষণ",
    "বাংলাদেশ বন্যা ত্রাণ", "বাংলাদেশ ঘূর্ণিঝড়",
    # Reform & movements
    "কোটা সংস্কার আন্দোলন", "গণঅভ্যুত্থান জুলাই", "বৈষম্যবিরোধী আন্দোলন",
    "সংস্কার কমিশন সুপারিশ", "বাংলাদেশ গণমাধ্যম স্বাধীনতা",
    "বাংলাদেশ মত প্রকাশের স্বাধীনতা", "ডিজিটাল নিরাপত্তা আইন বাংলাদেশ",
    # Tribunals & accountability
    "আন্তর্জাতিক অপরাধ ট্রাইব্যুনাল বাংলাদেশ", "বাংলাদেশ দুদক",
    "বাংলাদেশ সম্পদের হিসাব", "বাংলাদেশ পাচার অর্থ",
    # Regional
    "ঢাকা মহানগর", "চট্টগ্রাম রাজনীতি", "সিলেট রাজনীতি",
    "রাজশাহী রাজনীতি", "খুলনা রাজনীতি", "রংপুর রাজনীতি",
    # Specific issues
    "বাংলাদেশ পদ্মা সেতু", "বাংলাদেশ মেট্রোরেল", "বাংলাদেশ পারমাণবিক বিদ্যুৎ",
    "বাংলাদেশ গার্মেন্টস শ্রমিক", "বাংলাদেশ কৃষক আন্দোলন",
    # Misc political
    "বাংলাদেশ সংবাদ সম্মেলন", "বাংলাদেশ মিটিং মিছিল",
    "বাংলাদেশ ধর্মঘট", "বাংলাদেশ অবরোধ",
    "বাংলাদেশ সাম্প্রদায়িক সম্প্রীতি", "বাংলাদেশ সংখ্যালঘু",
]


def scrape_google_news(existing_urls, existing_headlines):
    articles = []
    base = "https://news.google.com/rss/search?q={}&hl=bn&gl=BD&ceid=BD:bn"

    for i, q in enumerate(QUERIES):
        url = base.format(requests.utils.quote(q))
        feed = feedparser.parse(url)
        for entry in feed.entries:
            title = clean(entry.get("title", ""))
            link = entry.get("link", "")
            if not title or not link or link in existing_urls:
                continue
            if not re.search(r"[\u0980-\u09FF]", title):
                continue
            # Dedup by headline
            norm = re.sub(r"\s+", " ", title.strip().lower())
            if norm in existing_headlines:
                continue

            source = entry.get("source", {}).get("title", "")
            if not source:
                m = re.search(r"\s*[-–—]\s*([^-–—]+)$", title)
                if m:
                    source = m.group(1).strip()
                    title = title[:m.start()].strip()

            existing_urls.add(link)
            existing_headlines.add(norm)
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
        if (i + 1) % 10 == 0:
            print(f"  Queries: {i+1}/{len(QUERIES)}, new: {len(articles)}")
        time.sleep(0.7)

    print(f"  Google News total: {len(articles)} new")
    return articles


def scrape_daily_star_deeper(existing_urls, existing_headlines):
    """Even deeper Daily Star crawl — up to 40 pages per section."""
    base = "https://bangla.thedailystar.net"
    sections = [
        "bangladesh", "politics", "bangladesh/crime-justice",
        "bangladesh/government-politics", "bangladesh/accident-fire",
        "opinion", "opinion/editorial", "opinion/columns",
        "international", "business", "sports",
        "bangladesh/development", "bangladesh/education",
    ]
    articles = []
    for section in sections:
        for page in range(1, 40):
            url = f"{base}/news/{section}" if page == 1 else f"{base}/news/{section}?page={page}"
            try:
                r = requests.get(url, headers=HEADERS, timeout=15)
                if r.status_code != 200: break
            except Exception: break

            s = BeautifulSoup(r.content, "lxml")
            found = 0
            for a_tag in s.find_all("a", href=True):
                href = a_tag["href"]
                if "/news/" not in href or href.count("/") < 3: continue
                full_url = urljoin(base, href)
                if full_url in existing_urls: continue

                h = a_tag.find(["h2", "h3", "h4", "h5"])
                text = h.get_text(strip=True) if h else a_tag.get_text(strip=True)
                if not text or len(text) < 10 or len(text) > 300: continue
                if not re.search(r"[\u0980-\u09FF]", text): continue

                norm = re.sub(r"\s+", " ", text.strip().lower())
                if norm in existing_headlines: continue

                img_url = ""
                parent = a_tag.parent
                if parent:
                    img = parent.find("img")
                    if img:
                        img_url = img.get("data-src") or img.get("src") or ""
                        if img_url: img_url = urljoin(base, img_url)

                existing_urls.add(full_url)
                existing_headlines.add(norm)
                a_id = aid(full_url)
                articles.append({
                    "item_id": a_id,
                    "headline": clean(text),
                    "source_url": full_url,
                    "outlet": "The Daily Star",
                    "image_url": img_url,
                    "image_path": dl_img(img_url, a_id),
                    "date": "", "section": section,
                })
                found += 1
            if found == 0 and page > 3: break
            time.sleep(0.8)

    print(f"  Daily Star deep: {len(articles)} new")
    return articles


def fetch_og_batch(articles):
    """Fetch og:image for items missing images."""
    no_img = [a for a in articles if not a["image_path"]]
    print(f"  Fetching og:image for {len(no_img)} items...")
    fetched = 0
    for i, a in enumerate(no_img):
        img_url, img_path = fetch_og(a["source_url"], a["item_id"])
        if img_path:
            a["image_url"] = img_url
            a["image_path"] = img_path
            fetched += 1
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(no_img)}: {fetched} images")
        time.sleep(0.3)
    print(f"    Done: {fetched}/{len(no_img)} images fetched")


def main():
    SCRAPE_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    existing_urls, existing_headlines = load_existing()
    print(f"Already collected: {len(existing_urls)} articles")
    print(f"Target: 10,000 (need {max(0, 10000 - len(existing_urls))} more)\n")

    all_new = []

    # Phase 1: Daily Star deep
    print("Phase 1: Daily Star deeper crawl...")
    ds = scrape_daily_star_deeper(existing_urls, existing_headlines)
    all_new.extend(ds)
    print(f"  Running total: {len(existing_urls)} articles\n")

    # Phase 2: Google News expanded
    print("Phase 2: Google News RSS ({} queries)...".format(len(QUERIES)))
    gn = scrape_google_news(existing_urls, existing_headlines)
    all_new.extend(gn)
    print(f"  Running total: {len(existing_urls)} articles\n")

    # Phase 3: Fetch images
    print("Phase 3: Fetching images...")
    fetch_og_batch(all_new)

    # Final dedup
    seen = set()
    deduped = []
    for a in all_new:
        norm = re.sub(r"\s+", " ", a["headline"].strip().lower())
        if norm not in seen:
            seen.add(norm)
            deduped.append(a)

    save(deduped)

    total = len(load_existing()[0])
    imgs = len(list(IMAGE_DIR.glob("*")))
    print(f"\n{'='*50}")
    print(f"New this run: {len(deduped)}")
    print(f"Total articles: {total}")
    print(f"Total images: {imgs}")
    if total >= 10000:
        print("TARGET REACHED!")
    else:
        print(f"Still need {10000 - total} more. Run again for fresh results.")


if __name__ == "__main__":
    main()
